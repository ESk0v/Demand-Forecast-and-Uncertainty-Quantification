import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


# ─── Dataset ──────────────────────────────────────────────────────────────────

def _split_ratios_for_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path)
    if dataset_name == "dataset.pt":
        return 1 / 10, 1 / 10, 1 / 10
    return 1 / 12, 1 / 12, 1 / 6


def load_and_split_dataset(dataset_path, val_ratio=None, cal_ratio=None, test_ratio=None):
    """
    Chronological 4-way split — no data leakage.

    val and cal draw from the same pool so neither touches model weights,
    giving a larger, more representative calibration set.

    Effective proportions by dataset:
      dataset.pt       -> train=70%  val=20%  cal=20%(same rows as val)  test=10%
      dataset_syn*.pt  -> train=66.7%  val=16.7%  cal=16.7%(same rows as val)  test=16.7%
    """
    if val_ratio is None or cal_ratio is None or test_ratio is None:
        val_ratio, cal_ratio, test_ratio = _split_ratios_for_dataset(dataset_path)

    dataset = torch.load(dataset_path, weights_only=False)

    full_dataset = TensorDataset(
        dataset['encoder'],
        dataset['decoder'],
        dataset['target'],
    )

    n_total     = len(full_dataset)
    test_size   = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size  = n_total - valcal_size - test_size

    i0, i1 = 0,          train_size
    i2     =             i1 + valcal_size
    i3     = i2 + test_size   # == n_total

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset   = Subset(full_dataset, range(i1, i2))
    cal_dataset   = Subset(full_dataset, range(i1, i2))   # same window as val
    test_dataset  = Subset(full_dataset, range(i2, i3))

    print(
        f"Split sizes — train: {train_size}  val: {valcal_size}  "
        f"cal: {valcal_size}  test: {test_size}  (total: {n_total})"
    )

    return (train_dataset, val_dataset, cal_dataset, test_dataset,
            train_size, valcal_size, valcal_size, test_size)


# ─── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(epoch, warmup_epochs=3, base_lr=1e-4):
    """Linear warmup for the first `warmup_epochs` epochs, then constant."""
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


# ─── Loss functions ───────────────────────────────────────────────────────────

def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width        = q_high - q_low
    penalty_low  = (2 / alpha) * torch.clamp(q_low  - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)


# ─── Phase 1: train median path ───────────────────────────────────────────────

def train_epoch_median(model, train_loader, optimizer, device, train_size):
    """
    One pass training the encoder + decoder + fc_decoderMedian only.
    The spread head's parameters are frozen (requires_grad=False).
    """
    model.train()
    epoch_loss = 0.0

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer.zero_grad()
        _, q50, _ = model(enc, dec)

        loss = quantile_loss(q50, tgt, q=0.5)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.encoderLstm.parameters(),      max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.decoderLstm.parameters(),      max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_decoderMedian.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item() * enc.size(0)

    return epoch_loss / train_size


def validate_epoch_median(model, val_loader, device, val_size):
    """Validation for phase 1 — median loss only."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            _, q50, _ = model(enc, dec)
            val_loss += quantile_loss(q50, tgt, q=0.5).item() * enc.size(0)

    return val_loss / val_size


# ─── Phase 2: train spread path ───────────────────────────────────────────────

def train_epoch_spread(model, train_loader, optimizer, device, train_size, conformal_alpha=0.2):
    model.train()
    epoch_loss = 0.0
    q10_le_q50 = []
    q50_le_q90 = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer.zero_grad()
        q10, q50, q90 = model(enc, dec)

        loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)

        # -----------------------------
        # Horizon-aware auxiliary losses
        # -----------------------------

        H = q10.shape[1]  # forecast length

        horizon_weight = torch.linspace(2, 0.7, H, device=device)

        # expand to batch shape
        hw = horizon_weight.unsqueeze(0)  # (1, H)

        # width penalty (prevents inflation, especially early horizons)
        width = q90 - q10
        width_loss = torch.mean(hw * width)

        # symmetry loss (stabilizes structure)
        symmetry_loss = torch.mean(hw * torch.abs((q50 - q10) - (q90 - q50)))

        # coverage loss (IMPORTANT FIX: horizon-aware)
        inside = ((tgt >= q10) & (tgt <= q90)).float()
        coverage = torch.mean(hw * inside)

        coverage_loss = (coverage - (1 - conformal_alpha)) ** 2

        aux_loss = (
            0.6 * width_loss +
            0.2 * symmetry_loss +
            0.2 * coverage_loss
        )

        aux_weight = 0.05

        loss = loss_interval + aux_weight * aux_loss
        loss.backward()

        # gradients
        torch.nn.utils.clip_grad_norm_(model.uncertaintyDecoderLstm.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderLow.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderHigh.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.proj_hidden.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.proj_cell.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item() * enc.size(0)

        q10_le_q50.append((q10 <= q50).float().mean().item())
        q50_le_q90.append((q50 <= q90).float().mean().item())

    print("q10<=q50 avg:", np.mean(q10_le_q50))
    print("q50<=q90 avg:", np.mean(q50_le_q90))

    return epoch_loss / train_size


def validate_epoch_spread(model, val_loader, device, val_size, conformal_alpha=0.2):
    """Validation for phase 2 — interval loss only."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            q10, _, q90   = model(enc, dec)
            val_loss += interval_score_loss(q10, q90, tgt, alpha=conformal_alpha).item() * enc.size(0)

    return val_loss / val_size


# ─── Inference helper ─────────────────────────────────────────────────────────

def collect_predictions(model, loader, device):
    """Run inference and return (q10, q50, q90, targets) as numpy arrays."""
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


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizers, config, epoch, val_loss,
                    train_losses, val_losses, model_save_path, phase):
    """
    Save a checkpoint.  `optimizers` is a dict keyed by name so both
    phase-1 and phase-2 optimizers can be stored independently.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_states': {k: v.state_dict() for k, v in optimizers.items()},
        'config':           vars(config),
        'epoch':            epoch,
        'phase':            phase,
        'val_loss':         val_loss,
        'train_losses':     train_losses.copy(),
        'val_losses':       val_losses.copy(),
    }, model_save_path)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss",      linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses,   label="Validation Loss", linewidth=1.5, marker='o', markersize=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Train vs Validation Loss{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if all_losses and min(all_losses) > 0 and max(all_losses) / min(all_losses) > 10:
        ax.set_yscale('log')
        ax.set_ylabel("Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_coverage_per_horizon(q10, q90, targets, save_path, alpha=0.1):
    """Raw per-horizon coverage — no conformal adjustment."""
    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    horizons = np.arange(1, len(raw_coverage) + 1)
    ax.plot(horizons, raw_coverage * 100, label="Raw interval coverage", color='steelblue')
    ax.axhline(y=(1 - alpha) * 100, color='orange', linestyle='--',
               label=f'Nominal {(1 - alpha) * 100:.0f}% target')
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Prediction Interval Coverage per Horizon", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Coverage plot saved to {save_path}")


# ─── Phase runner ─────────────────────────────────────────────────────────────

def _run_phase(
    phase, model, train_loader, val_loader,
    train_size, val_size, config,
    model_save_path, logger, patience, conformal_alpha,
):
    """
    Generic training loop for one phase.

    phase=1 → trains median path, uses quantile loss for early stopping
    phase=2 → trains spread path, uses interval score for early stopping
    """
    model.freeze_for_phase(phase)

    if phase == 1:
        optimizer = torch.optim.AdamW([
            {'params': model.encoderLstm.parameters()},
            {'params': model.decoderLstm.parameters()},
            {'params': model.fc_decoderMedian.parameters()},
        ], lr=config.learning_rate, weight_decay=1e-4)

        def _train(epoch):
            return train_epoch_median(model, train_loader, optimizer, config.device, train_size)

        def _validate():
            return validate_epoch_median(model, val_loader, config.device, val_size)

    else:  # phase == 2
        optimizer = torch.optim.AdamW([
            {'params': model.uncertaintyDecoderLstm.parameters()},
            {'params': model.proj_hidden.parameters()},
            {'params': model.proj_cell.parameters()},
            {'params': model.fc_uncertaintyDecoderLow.parameters()},
            {'params': model.fc_uncertaintyDecoderHigh.parameters()},
        ], lr=config.learning_rate, weight_decay=1e-4)

        def _train(epoch):
            return train_epoch_spread(
                model, train_loader, optimizer, config.device, train_size, conformal_alpha
            )

        def _validate():
            return validate_epoch_spread(
                model, val_loader, config.device, val_size, conformal_alpha
            )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss     = np.inf
    best_epoch        = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    phase_tag = f"Phase {phase}/2"

    for epoch in range(1, config.epochs + 1):

        # Linear warmup
        warmup_lr = get_lr(epoch, warmup_epochs=3, base_lr=config.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

        train_loss = _train(epoch)
        val_loss   = _validate()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch > 3:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            logger.info(
                f"[{phase_tag}] Epoch {epoch}: "
                f"Train={train_loss:.4f}  Val={val_loss:.4f}  LR={current_lr:.2e}  "
                f"[best — saving checkpoint]"
            )
            save_checkpoint(
                model, {f'optimizer_phase{phase}': optimizer},
                config, epoch, val_loss, train_losses, val_losses,
                model_save_path, phase,
            )
        else:
            logger.info(
                f"[{phase_tag}] Epoch {epoch}: "
                f"Train={train_loss:.4f}  Val={val_loss:.4f}  LR={current_lr:.2e}  "
                f"(best epoch: {best_epoch})"
            )
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"[{phase_tag}] Early stopping at epoch {epoch}")
                break

    return train_losses, val_losses, best_epoch, best_val_loss


# ─── Public API ───────────────────────────────────────────────────────────────

def train_model(config, train_loader, val_loader, cal_loader,
                train_size, val_size, cal_size,
                model_save_path, logger=None, patience=20, conformal_alpha=0.2):
    """
    Sequential two-phase training.

    Phase 1 — median: trains encoder + decoder + fc_decoderMedian until
               the median validation loss converges.

    Phase 2 — spread: loads the best phase-1 checkpoint, freezes the
               median path, then trains the uncertainty head until the
               interval-score validation loss converges.

    The final checkpoint contains both phases' weights and is fully
    compatible with Plot.py and collect_predictions().

    conformal_alpha controls the interval training objective in phase 2.
    Conformal calibration has been removed from this function; apply it
    externally if required.
    """
    model = LSTMForecast(config).to(config.device)

    if logger is not None:
        logger.info(
            f"Sequential training — "
            f"epochs={config.epochs}  patience={patience}  "
            f"conformal_alpha={conformal_alpha}  "
            f"(target coverage={(1 - conformal_alpha) * 100:.0f}%)"
        )

    # ── Phase 1: median ───────────────────────────────────────────────────────
    if logger is not None:
        logger.info("── Phase 1: training median path ──")

    p1_train, p1_val, p1_best_epoch, p1_best_val = _run_phase(
        phase           = 1,
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        train_size      = train_size,
        val_size        = val_size,
        config          = config,
        model_save_path = model_save_path,
        logger          = logger,
        patience        = patience,
        conformal_alpha = conformal_alpha,
    )

    # Reload best phase-1 weights before entering phase 2
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if logger is not None:
        logger.info(
            f"Phase 1 complete — best epoch {p1_best_epoch}  "
            f"val_loss={p1_best_val:.4f}"
        )

    # ── Phase 2: spread ───────────────────────────────────────────────────────
    if logger is not None:
        logger.info("── Phase 2: training spread path (median frozen) ──")

    p2_train, p2_val, p2_best_epoch, p2_best_val = _run_phase(
        phase           = 2,
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        train_size      = train_size,
        val_size        = val_size,
        config          = config,
        model_save_path = model_save_path,
        logger          = logger,
        patience        = patience,
        conformal_alpha = conformal_alpha,
    )

    # Reload best phase-2 weights (both phases' weights are in the model now)
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Persist full loss history for both phases into final checkpoint
    checkpoint['p1_train_losses'] = p1_train
    checkpoint['p1_val_losses']   = p1_val
    checkpoint['p2_train_losses'] = p2_train
    checkpoint['p2_val_losses']   = p2_val
    # Keep flat lists too so existing Plot.py keys don't break
    checkpoint['train_losses']    = p1_train + p2_train
    checkpoint['val_losses']      = p1_val   + p2_val
    checkpoint['conformal_alpha'] = conformal_alpha
    torch.save(checkpoint, model_save_path)

    if logger is not None:
        logger.success(
            f"Training complete — "
            f"Phase 1 best epoch {p1_best_epoch} (val={p1_best_val:.4f})  |  "
            f"Phase 2 best epoch {p2_best_epoch} (val={p2_best_val:.4f})"
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    run_dir  = os.path.dirname(model_save_path)
    plot_dir = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_train_val_loss(
        p1_train, p1_val, p1_best_epoch,
        os.path.join(plot_dir, "train_val_loss_phase1.png"),
        title_suffix=" — Phase 1 (Median)",
    )
    plot_train_val_loss(
        p2_train, p2_val, p2_best_epoch,
        os.path.join(plot_dir, "train_val_loss_phase2.png"),
        title_suffix=" — Phase 2 (Spread)",
    )

    # Coverage plot on the cal set
    model.freeze_for_phase(2)   # ensure no dropout, all paths active
    q10_raw, _, q90_raw, tgt_cal = collect_predictions(model, cal_loader, config.device)
    plot_coverage_per_horizon(
        q10_raw, q90_raw, tgt_cal,
        os.path.join(plot_dir, "coverage_per_horizon.png"),
        alpha=conformal_alpha,
    )

    if logger is not None:
        logger.info(f"Plots saved to {plot_dir}")

    return p2_best_val, p1_train + p2_train, p1_val + p2_val
