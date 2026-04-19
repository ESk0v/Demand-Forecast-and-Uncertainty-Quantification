import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


def load_and_split_dataset(dataset_path, val_ratio=0.20, cal_ratio=0.10, test_ratio=0.10):
    """
    Chronological 4-way split — no data leakage.

    CHANGE: val and cal are now drawn from the same combined pool (30% of data).
    The first 66% of that pool is used for early stopping (val),
    the last 34% is used for conformal calibration (cal).
    Neither set ever touches model weights, so merging them is safe and gives
    a larger, more representative calibration set.

    Effective proportions:  train=70%  val=20%  cal=10%  test=10%
    """
    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total     = len(full_dataset)
    test_size   = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size  = n_total - valcal_size - test_size

    i0 = 0
    i1 = train_size
    i2 = i1 + valcal_size
    i3 = i2 + test_size   # == n_total

    # val and cal both use the full valcal pool.
    # Neither set touches model weights so there is no leakage.
    val_size = valcal_size
    cal_size = valcal_size

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset   = Subset(full_dataset, range(i1, i2))
    cal_dataset   = Subset(full_dataset, range(i1, i2))
    test_dataset  = Subset(full_dataset, range(i2, i3))

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
    in epoch 1 before it has any sense of the loss landscape.
    """
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


def train_epoch(model, train_loader, optimizer, device, train_size):
    """
    One full pass over the training set.

    Lag feature noise (encoder indices 8–10: lag_1h, lag_24h, lag_168h):
        Small Gaussian noise prevents the model from over-relying on exact
        lag values, which would collapse early-horizon uncertainty.

    Loss weights (0.30 / 0.40 / 0.30):
        Symmetric tail weights so the model is equally penalised for being
        wrong on both sides.

    Crossing penalty:
        Soft penalty for q10 > q50 or q50 > q90.
    """
    model.train()
    epoch_loss = 0
    q10_le_q50 = []
    q50_le_q90 = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        # ── Lag feature noise — training only ─────────────────────────────────
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

        loss = 0.30 * loss_q10 + 0.40 * loss_q50 + 0.30 * loss_q90 + crossing_penalty
        loss.backward()

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
    Validation loss — same weights as training for consistent early stopping.
    No noise, no dropout.
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

            loss = 0.30 * loss_q10 + 0.40 * loss_q50 + 0.30 * loss_q90
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
    Per-horizon conformal calibration with a horizon-aware alpha ramp.

    Early horizons are already well-covered so need less expansion.
    Late horizons are under-covered so need aggressive expansion.
    The ramp on alpha achieves this: lower alpha late = larger u_alpha_t late.
    """

    # 1. Fix crossing quantiles
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    # 2. Miss distance — 0 if inside [q10, q90], else distance to nearest bound
    scores = np.maximum(
        np.maximum(q10 - targets, targets - q90),
        0
    )

    n_horizons = scores.shape[1]

    # Horizon-aware alpha ramp: tight early (less expansion), loose late (more expansion)
    alpha_per_horizon = np.linspace(alpha * 1.5, alpha * 0.5, n_horizons)
    alpha_per_horizon = np.clip(alpha_per_horizon, 0.01, 0.25)

    u_alpha_t = np.array([
        np.quantile(scores[:, t], 1.0 - alpha_per_horizon[t])
        for t in range(n_horizons)
    ])

    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    empirical_coverage = float(
        np.mean((targets >= q10_cal) & (targets <= q90_cal))
    )

    # Per-horizon coverage diagnostic printed at calibration time
    per_horizon_coverage = np.mean(
        (targets >= q10_cal) & (targets <= q90_cal), axis=0
    )
    print(
        f"Empirical coverage on cal set: {empirical_coverage:.3f} "
        f"(target ≈ {1 - alpha:.2f})"
    )
    print(
        f"Per-horizon coverage — "
        f"h1: {per_horizon_coverage[0]:.2f}  "
        f"h24: {per_horizon_coverage[23]:.2f}  "
        f"h72: {per_horizon_coverage[71]:.2f}  "
        f"h168: {per_horizon_coverage[-1]:.2f}"
    )

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


def plot_coverage_per_horizon(q10, q90, targets, u_alpha_t, save_path, alpha=0.1):
    """
    Plot raw vs calibrated per-horizon coverage.
    Generated automatically after every training run so you can immediately
    see whether the conformal ramp is working as intended.
    """
    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0)
    cal_coverage = np.mean((targets >= q10_cal) & (targets <= q90_cal), axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    horizons = np.arange(1, len(raw_coverage) + 1)
    ax.plot(horizons, raw_coverage * 100,  label="Raw q10–q90 coverage",           color='steelblue')
    ax.plot(horizons, cal_coverage * 100,  label="Per-horizon calibrated coverage", color='green')
    ax.axhline(y=(1 - alpha) * 100, color='orange', linestyle='--',
               label=f'Nominal {(1-alpha)*100:.0f}% (calibrated target)')
    ax.axhline(y=80, color='black', linestyle='--',
               label='Nominal 80% (raw interval target)')
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Prediction Interval Coverage per Horizon\nRaw q10–q90 vs Conformal-Calibrated", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Coverage plot saved to {save_path}")


def train_model(config, train_loader, val_loader, cal_loader,
                train_size, val_size, cal_size,
                model_save_path, logger=None, patience=20, conformal_alpha=0.1):
    """
    Full training loop with warmup, early stopping, and conformal calibration.

      val set  → early stopping / model selection only
      cal set  → conformal calibration (never seen during training)
      test set → final held-out evaluation (handled outside this function)

    patience default raised to 20 — patience=5 was stopping the model after
    only ~2 real epochs of full-LR training (warmup consumes the first 3).
    Scheduler patience is set independently to 10.
    """
    model = LSTMForecast(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    # Scheduler patience independent of early stopping patience.
    # 10 epochs without improvement before halving LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss     = np.inf
    best_epoch        = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    if logger is not None:
        logger.info(f"Starting training for {config.epochs} epochs  patience={patience}...")

    for epoch in range(1, config.epochs + 1):

        # Linear warmup overrides scheduler for first 3 epochs
        warmup_lr = get_lr(epoch, warmup_epochs=3, base_lr=config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

        train_loss = train_epoch(model, train_loader, optimizer, config.device, train_size)
        train_losses.append(train_loss)

        val_loss = validate_epoch(model, val_loader, config.device, val_size)
        val_losses.append(val_loss)

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
    model.load_state_dict(checkpoint['model_state_dict'])

    if logger is not None:
        logger.info(f"Running conformal calibration on cal set ({cal_size} samples) …")

    q10_raw, q50_raw, q90_raw, tgt_cal = collect_predictions(model, cal_loader, config.device)
    _, _, u_alpha_t = conformal_calibration(
        q10_raw, q50_raw, q90_raw, tgt_cal, alpha=conformal_alpha
    )
    checkpoint['conformal_u_alpha'] = u_alpha_t
    checkpoint['conformal_alpha']   = conformal_alpha
    torch.save(checkpoint, model_save_path)

    if logger is not None:
        logger.success(
            f"Training complete. Best checkpoint: epoch {checkpoint['epoch']}  "
            f"val_loss={checkpoint['val_loss']:.4f}  "
            f"conformal_u_alpha shape={u_alpha_t.shape}"
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    run_dir        = os.path.dirname(model_save_path)
    plot_dir       = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    loss_plot_path     = os.path.join(plot_dir, "train_val_loss.png")
    coverage_plot_path = os.path.join(plot_dir, "coverage_per_horizon.png")

    plot_train_val_loss(train_losses, val_losses, checkpoint['epoch'], loss_plot_path)
    plot_coverage_per_horizon(
        q10_raw, q90_raw, tgt_cal, u_alpha_t,
        coverage_plot_path, alpha=conformal_alpha
    )

    if logger is not None:
        logger.info(f"Loss curve saved to {loss_plot_path}")
        logger.info(f"Coverage plot saved to {coverage_plot_path}")

    return best_val_loss, train_losses, val_losses


def apply_conformal(q10, q90, u_alpha):
    """
    Apply saved conformal calibration to new predictions.

    Supports:
      - scalar u_alpha → same expansion for all horizons
      - vector u_alpha (per-horizon) → one expansion per forecast step
    """
    is_tensor = False
    device    = None
    if isinstance(q10, torch.Tensor):
        is_tensor = True
        device    = q10.device
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
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))