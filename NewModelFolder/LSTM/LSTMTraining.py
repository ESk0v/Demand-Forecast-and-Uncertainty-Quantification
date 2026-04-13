import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


def load_and_split_dataset(dataset_path, val_ratio=0.15, cal_ratio=0.10, test_ratio=0.15):
    """
    Chronological 4-way split — no data leakage.

    Default proportions:  train=60%  val=15%  cal=10%  test=15%

    val  → early stopping / model selection only
    cal  → conformal calibration (never seen during training)
    test → final held-out evaluation

    Returned order:
        train_dataset, val_dataset, cal_dataset, test_dataset,
        train_size, val_size, cal_size, test_size
    """
    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total    = len(full_dataset)
    val_size   = int(n_total * val_ratio)
    cal_size   = int(n_total * cal_ratio)
    test_size  = int(n_total * test_ratio)
    train_size = n_total - val_size - cal_size - test_size

    i0 = 0
    i1 = train_size
    i2 = i1 + val_size
    i3 = i2 + cal_size
    i4 = i3 + test_size  # == n_total

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset   = Subset(full_dataset, range(i1, i2))
    cal_dataset   = Subset(full_dataset, range(i2, i3))
    test_dataset  = Subset(full_dataset, range(i3, i4))

    print(
        f"Split sizes — train: {train_size}  val: {val_size}  "
        f"cal: {cal_size}  test: {test_size}  (total: {n_total})"
    )

    return (train_dataset, val_dataset, cal_dataset, test_dataset,
            train_size, val_size, cal_size, test_size)


def get_lr(epoch, warmup_epochs=3, base_lr=1e-4):
    """
    Linear warmup for the first warmup_epochs epochs, then constant.

    Warmup prevents the model from making large destabilising updates
    in epoch 1 before it has any sense of the loss landscape — this was
    causing epoch 1 to be suspiciously good or bad, making early stopping
    select a poor checkpoint.
    """
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


def train_epoch(model, train_loader, optimizer, device, train_size):
    """
    One full pass over the training set.

    Lag feature noise (encoder indices 8–10: lag_1h, lag_24h, lag_168h):
        Small Gaussian noise prevents the model from over-relying on exact
        lag values, which would collapse early-horizon uncertainty. std=0.05
        is ~5% of one normalised demand std — small but effective.

    Loss weights (0.35 / 0.40 / 0.25):
        q10 weighted higher than q90 to correct systematic lower-bound
        underestimation seen in earlier versions.

    Crossing penalty:
        Soft penalty for q10 > q50 or q50 > q90. Pushes the model toward
        well-ordered quantiles without hard constraints.
    """
    model.train()
    epoch_loss = 0
    q10_le_q50 = []
    q50_le_q90 = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        # ── Lag feature noise — training only ─────────────────────────────────
        # Encoder layout:
        #   0: abvaerk  1: toutdoor  2-7: time features
        #   8: lag_1h   9: lag_24h   10: lag_168h
        noise = torch.randn_like(enc[:, :, 8:]) * 0.05
        enc = enc.clone()
        enc[:, :, 8:] += noise

        optimizer.zero_grad()
        q10, q50, q90 = model(enc, dec)

        loss_q10 = quantile_loss(q10, tgt, 0.1)
        loss_q50 = quantile_loss(q50, tgt, 0.5)
        loss_q90 = quantile_loss(q90, tgt, 0.9)

        # Soft crossing penalty
        crossing_penalty = (
            torch.mean(torch.relu(q10 - q50)) +
            torch.mean(torch.relu(q50 - q90))
        ) * 0.1

        loss = 0.35 * loss_q10 + 0.4 * loss_q50 + 0.25 * loss_q90 + crossing_penalty
        loss.backward()

        # Relaxed gradient clipping — 0.5 was too tight for hidden=768
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * enc.size(0)

        q10_le_q50.append((q10 <= q50).float().mean().item())
        q50_le_q90.append((q50 <= q90).float().mean().item())

    print("q10<=q50 avg:", np.mean(q10_le_q50))
    print("q50<=q90 avg:", np.mean(q50_le_q90))

    return epoch_loss / train_size


def validate_epoch(model, val_loader, device, val_size):
    """
    Validation loss using the same weights as training so early stopping
    selects checkpoints on a consistent objective.
    No noise, no dropout — pure model evaluation.
    """
    model.eval()
    val_loss_epoch = 0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

            q10, q50, q90 = model(enc, dec)

            loss_q10 = quantile_loss(q10, tgt, 0.1)
            loss_q50 = quantile_loss(q50, tgt, 0.5)
            loss_q90 = quantile_loss(q90, tgt, 0.9)

            # Match train weights exactly
            loss = 0.35 * loss_q10 + 0.4 * loss_q50 + 0.25 * loss_q90
            val_loss_epoch += loss.item() * enc.size(0)

    return val_loss_epoch / val_size


def collect_predictions(model, loader, device):
    """Run inference over a DataLoader and return (q10, q50, q90, targets) as numpy arrays."""
    model.eval()
    all_q10, all_q50, all_q90, all_tgt = [], [], [], []

    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            q10, q50, q90 = model(enc, dec)
            all_q10.append(q10.cpu().numpy())
            all_q50.append(q50.cpu().numpy())
            all_q90.append(q90.cpu().numpy())
            all_tgt.append(tgt.cpu().numpy())

    return (
        np.concatenate(all_q10),
        np.concatenate(all_q50),
        np.concatenate(all_q90),
        np.concatenate(all_tgt),
    )


def conformal_calibration(q10, q50, q90, targets, alpha=0.1):
    """
    Per-horizon conformal calibration with monotone accumulation.

    Score function:
        s(t) = max(q10_t - y_t, y_t - q90_t, 0)
        Zero when y is inside [q10, q90], positive otherwise.

    Alpha ramp:
        Decreases from alpha at hour 1 to alpha*0.35 at hour 168.
        Lower alpha → higher quantile level → wider correction.
        Greedier at late horizons where the model is less accurate.
        alpha*0.35 targets ~93-94% at late horizons (was 0.2 → 97-98%,
        too conservative and wasted sharpness).

    Monotone accumulate:
        Band can only grow with horizon — never shrinks due to noise.
    """
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    scores = np.maximum(q10 - targets, targets - q90)
    scores = np.maximum(scores, 0)  # shape: (N, 168)

    n_horizons = scores.shape[1]

    # Ramp alpha down → greedier at late horizons
    # 0.35 endpoint targets ~93-94% coverage at hour 168
    alpha_per_horizon = np.linspace(alpha, alpha, n_horizons)

    u_alpha_t = np.array([
        np.quantile(scores[:, t], 1.0 - alpha_per_horizon[t])
        for t in range(n_horizons)
    ])

    # Monotone — band can only grow, never shrink
    u_alpha_t = np.maximum.accumulate(u_alpha_t)

    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    empirical_coverage = float(np.mean((targets >= q10_cal) & (targets <= q90_cal)))
    print(f"Empirical coverage on cal set: {empirical_coverage:.3f} (target >= {1-alpha:.2f})")

    return q10_cal, q90_cal, u_alpha_t


def save_checkpoint(model, optimizer, config, epoch, val_loss,
                    train_losses, val_losses, model_save_path):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config':               vars(config),
        'epoch':                epoch,
        'val_loss':             val_loss,
        'train_losses':         train_losses.copy(),
        'val_losses':           val_losses.copy(),
    }, model_save_path)


def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path):
    """Plot training vs validation loss curves and save to disk."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss",      linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses,   label="Validation Loss", linewidth=1.5, marker='o', markersize=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Quantile Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss (Quantile)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if (len(all_losses) > 0
            and min(all_losses) > 0
            and max(all_losses) / min(all_losses) > 10):
        ax.set_yscale('log')
        ax.set_ylabel("Quantile Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(config, train_loader, val_loader, cal_loader,
                train_size, val_size, cal_size,
                model_save_path, logger=None, patience=5, conformal_alpha=0.1):
    """
    Full training loop with warmup, early stopping, and conformal calibration.

      val set  → early stopping / model selection only
      cal set  → conformal calibration (never seen during training)
      test set → final held-out evaluation (handled outside this function)

    Args:
        patience        : epochs without val improvement before stopping
        conformal_alpha : miscoverage rate (default 0.1 → 90% coverage target)
    """
    model     = LSTMForecast(config).to(config.device)

    # Reduced weight decay — dropout=0.2 already regularises,
    # 1e-3 was over-regularising in combination
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    # Scheduler fires after patience epochs without improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    best_val_loss     = np.inf
    best_epoch        = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    if logger is not None:
        logger.info(f"Starting training for {config.epochs} epochs...")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):

        # ── Linear warmup — overrides scheduler for first 3 epochs ────────────
        warmup_lr = get_lr(epoch, warmup_epochs=3, base_lr=config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

        train_loss = train_epoch(model, train_loader, optimizer, config.device, train_size)
        train_losses.append(train_loss)

        val_loss = validate_epoch(model, val_loader, config.device, val_size)
        val_losses.append(val_loss)

        # Only step scheduler after warmup is complete
        if epoch > 3:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_epoch        = epoch
            logger.info(
                f"Epoch {epoch}: Train={train_loss:.4f}  Val={val_loss:.4f}  LR={current_lr:.2e}  "
                f"[best — saving checkpoint]"
            )
            save_checkpoint(model, optimizer, config, epoch, val_loss,
                            train_losses, val_losses, model_save_path)
        else:
            logger.info(
                f"Epoch {epoch}: Train={train_loss:.4f}  Val={val_loss:.4f}  LR={current_lr:.2e}  "
                f"(best epoch: {best_epoch})"
            )
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── Persist final loss curves into checkpoint ──────────────────────────────
    checkpoint = torch.load(model_save_path, weights_only=False)
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses']   = val_losses

    # ── Conformal calibration ──────────────────────────────────────────────────
    # Reload best weights before calibration
    model.load_state_dict(checkpoint['model_state_dict'])

    if logger is not None:
        logger.info(f"Running conformal calibration on cal set ({cal_size} samples) …")

    q10_cal, q50_cal, q90_cal, tgt_cal = collect_predictions(model, cal_loader, config.device)
    _, _, u_alpha_t = conformal_calibration(
        q10_cal, q50_cal, q90_cal, tgt_cal, alpha=conformal_alpha
    )
    checkpoint['conformal_u_alpha'] = u_alpha_t  # shape: (forecast_length,)
    checkpoint['conformal_alpha']   = conformal_alpha
    torch.save(checkpoint, model_save_path)

    if logger is not None:
        logger.success(
            f"Training complete. Best checkpoint: epoch {checkpoint['epoch']}  "
            f"val_loss={checkpoint['val_loss']:.4f}  "
            f"conformal_u_alpha shape={u_alpha_t.shape}"
        )

    # ── Loss curve plot ────────────────────────────────────────────────────────
    run_dir        = os.path.dirname(model_save_path)
    plot_dir       = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    loss_plot_path = os.path.join(plot_dir, "train_val_loss.png")
    plot_train_val_loss(train_losses, val_losses, checkpoint['epoch'], loss_plot_path)

    if logger is not None:
        logger.info(f"Loss curve saved to {loss_plot_path}")

    return best_val_loss, train_losses, val_losses


def apply_conformal(q10, q90, u_alpha):
    """
    Apply saved conformal calibration to new predictions.

    Supports:
      - scalar u_alpha → same expansion for all horizons
      - vector u_alpha (per-horizon) → one expansion per forecast step

    Args:
        q10     : np.ndarray or torch.Tensor of shape (N, horizon)
        q90     : np.ndarray or torch.Tensor of shape (N, horizon)
        u_alpha : float or np.ndarray of shape (horizon,)

    Returns:
        q10_cal, q90_cal : calibrated bounds, same type as inputs
    """
    is_tensor = False
    device    = None
    if isinstance(q10, torch.Tensor):
        is_tensor = True
        device    = q10.device  # save before converting
        q10 = q10.detach().cpu().numpy()
        q90 = q90.detach().cpu().numpy()

    if np.isscalar(u_alpha):
        q10_cal = q10 - u_alpha
        q90_cal = q90 + u_alpha
    else:
        q10_cal = q10 - u_alpha[np.newaxis, :]
        q90_cal = q90 + u_alpha[np.newaxis, :]

    if is_tensor:
        q10_cal = torch.from_numpy(q10_cal).to(device)
        q90_cal = torch.from_numpy(q90_cal).to(device)

    return q10_cal, q90_cal


def quantile_loss(pred, target, q):
    """Pinball / quantile loss. Standard formulation."""
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))