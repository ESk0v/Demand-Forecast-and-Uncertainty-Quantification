import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from LSTMModel import LSTMForecast


def _split_ratios_for_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path)
    if dataset_name == "dataset.pt":
        return 1 / 10, 1 / 10, 1 / 10
    return 1 / 12, 1 / 12, 1 / 6


def load_and_split_dataset(dataset_path, val_ratio=None, cal_ratio=None, test_ratio=None):
    """
    Chronological 4-way split — no data leakage.

    val and cal intentionally use the same valcal pool (overlap).
    Neither set updates model weights.

    Effective proportions by dataset:
      dataset.pt       -> train=70%  val=20%  cal=20%(same rows as val)  test=10%
      dataset_syn*.pt  -> train=66.7%  val=16.7%  cal=16.7%(same rows as val)  test=16.7%
    """
    if val_ratio is None or cal_ratio is None or test_ratio is None:
        val_ratio, cal_ratio, test_ratio = _split_ratios_for_dataset(dataset_path)

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


def _flatten_autograd_grads(grad_tensors):
    vectors = []
    for grad in grad_tensors:
        if grad is None:
            continue
        vectors.append(grad.reshape(-1))
    if not vectors:
        return None
    return torch.cat(vectors)


def compute_encoder_cosine_similarity(model, loss_median, loss_interval):
    encoder_params = [p for p in model.encoderLstm.parameters() if p.requires_grad]

    g1_raw = torch.autograd.grad(
        loss_median,
        encoder_params,
        retain_graph=True,
        allow_unused=True,
    )
    g2_raw = torch.autograd.grad(
        loss_interval,
        encoder_params,
        retain_graph=True,
        allow_unused=True,
    )

    g1 = _flatten_autograd_grads(g1_raw)
    g2 = _flatten_autograd_grads(g2_raw)

    if g1 is None or g2 is None:
        return 0.0

    g1_norm = g1.norm().item()
    g2_norm = g2.norm().item()
    if g1_norm == 0.0 or g2_norm == 0.0:
        return 0.0

    return torch.nn.functional.cosine_similarity(
        g1.unsqueeze(0), g2.unsqueeze(0)
    ).item()


def train_epoch(model, train_loader, optimizer_point, optimizer_spread, device, train_size, conformal_alpha=0.2):
    """
    One full pass over the training set.

    Lag feature noise (encoder indices 8–10: lag_1h, lag_24h, lag_168h):
        Small Gaussian noise prevents the model from over-relying on exact
        lag values, which would collapse early-horizon uncertainty.

    Crossing penalty:
        Soft penalty for q_low > q50 or q50 > q_high.
    """
    model.train()
    epoch_loss = 0
    step_cos_sims = []
    q10_le_q50 = []
    q50_le_q90 = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer_point.zero_grad()
        optimizer_spread.zero_grad()
        q10, q50, q90 = model(enc, dec)

        loss_median   = quantile_loss(q50, tgt, q=0.5)
        loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)
        cos_sim = compute_encoder_cosine_similarity(model, loss_median, loss_interval)
        step_cos_sims.append(cos_sim)
        q50_for_cross = q50.detach() if getattr(model, "training_variant", "decoupled") == "decoupled" else q50
        crossing_penalty = (
            torch.mean(torch.relu(q10 - q50_for_cross)) + torch.mean(torch.relu(q50_for_cross - q90))
        ) * 0.1

        loss_median.backward(retain_graph=True)
        loss_spread = loss_interval + crossing_penalty
        loss_spread.backward()

        torch.nn.utils.clip_grad_norm_(model.encoderLstm.parameters(),         max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.decoderLstm.parameters(),         max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_decoderMedian.parameters(),    max_norm=1.0)
        optimizer_point.step()

        torch.nn.utils.clip_grad_norm_(model.encoderLstm.parameters(),         max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.uncertainty_cell_proj.parameters(),       max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.uncertaintyDecoderLstm.parameters(),      max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderLow.parameters(),    max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderHigh.parameters(),   max_norm=1.0)
        optimizer_spread.step()

        epoch_loss += (loss_median + loss_interval).item() * enc.size(0)

        q10_le_q50.append((q10 <= q50).float().mean().item())
        q50_le_q90.append((q50 <= q90).float().mean().item())

    print("q10<=q50 avg:", np.mean(q10_le_q50))
    print("q50<=q90 avg:", np.mean(q50_le_q90))
    avg_cos_sim = float(np.mean(step_cos_sims)) if step_cos_sims else 0.0
    print(f"Avg Cosine Sim (encoderLstm): {avg_cos_sim:.4f}")

    return epoch_loss / train_size, avg_cos_sim


def validate_epoch(model, val_loader, device, val_size, conformal_alpha=0.2):
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

            loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)
            loss_median   = quantile_loss(q50, tgt, q=0.5)
            loss = 0.6 * loss_interval + 0.4 * loss_median

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
    # 1. Fix crossing quantiles
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    # 2. Non-conformity scores — 0 if inside [q10, q90], else distance to nearest bound
    scores = np.maximum(
        np.maximum(q10 - targets, targets - q90),
        0
    )  # (N, H)

    n_horizons = scores.shape[1]

    # Flat alpha across all horizons — every horizon targets the same coverage
    u_alpha_t = np.array([
        np.quantile(scores[:, t], 1.0 - alpha)
        for t in range(n_horizons)
    ])  # (H,)

    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    empirical_coverage = float(
        np.mean((targets >= q10_cal) & (targets <= q90_cal))
    )

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


def save_checkpoint(model, optimizer_point, optimizer_spread, config, epoch, val_loss,
                    train_losses, val_losses, model_save_path):
    torch.save({
        'model_state_dict':           model.state_dict(),
        'optimizer_point_state_dict': optimizer_point.state_dict(),
        'optimizer_spread_state_dict': optimizer_spread.state_dict(),
        'config':                     vars(config),
        'epoch':                      epoch,
        'val_loss':                   val_loss,
        'train_losses':               train_losses.copy(),
        'val_losses':                 val_losses.copy(),
    }, model_save_path)


def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path, model_name):
    """Plot training vs validation loss curves and save to disk."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss",      linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses,   label="Validation Loss", linewidth=1.5, marker='o', markersize=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Quantile Loss", fontsize=12)
    ax.set_title(f"{model_name} — Train vs Validation Loss (Quantile)", fontsize=14)
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


def plot_cosine_and_coverage(
    cosine_sims_epoch,
    q10,
    q50,
    q90,
    targets,
    u_alpha_t,
    save_path,
    model_name,
    alpha=0.1,
):
    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0) * 100
    cal_coverage = np.mean((targets >= q10_cal) & (targets <= q90_cal), axis=0) * 100
    nominal = (1 - alpha) * 100

    mae_q50 = float(np.mean(np.abs(q50 - targets)))
    general_coverage = float(np.mean((targets >= q10_cal) & (targets <= q90_cal)) * 100)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    ax_cos, ax_cov = axes

    # Top: cosine similarity per epoch (dynamic y-range ±0.1)
    epochs = np.arange(1, len(cosine_sims_epoch) + 1)
    ax_cos.plot(
        epochs,
        cosine_sims_epoch,
        color="purple",
        marker="o",
        linewidth=1.4,
        markersize=3,
        label="Avg cosine similarity per epoch",
    )
    if len(cosine_sims_epoch) > 0:
        y_min = min(cosine_sims_epoch) - 0.1
        y_max = max(cosine_sims_epoch) + 0.1
        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1
        ax_cos.set_ylim(max(-1.0, y_min), min(1.0, y_max))
    ax_cos.set_xlabel("Epoch")
    ax_cos.set_ylabel("Cosine Similarity")
    ax_cos.set_title(f"{model_name} — Gradient Interference (Encoder)")
    ax_cos.grid(True, alpha=0.3)
    ax_cos.legend(loc="best")

    # Bottom: coverage per horizon
    horizons = np.arange(1, len(raw_coverage) + 1)
    line_raw, = ax_cov.plot(horizons, raw_coverage, color="steelblue", label="Raw interval coverage")
    line_cal, = ax_cov.plot(horizons, cal_coverage, color="green", label="Calibrated interval coverage")
    line_nom = ax_cov.axhline(
        y=nominal,
        color="orange",
        linestyle="--",
        label=f"Nominal {nominal:.0f}% target",
    )
    ax_cov.set_xlabel("Forecast Horizon (hours)")
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_ylim(80, 100)
    ax_cov.set_title(f"{model_name} — Coverage Per Horizon")
    ax_cov.grid(True, alpha=0.3)

    line_legend = ax_cov.legend(handles=[line_raw, line_cal, line_nom], loc="lower left", fontsize=9)
    ax_cov.add_artist(line_legend)

    metric_handles = [
        Line2D([], [], color="none", label=f"MAE (q50 vs target): {mae_q50:.4f}"),
        Line2D([], [], color="none", label=f"General coverage: {general_coverage:.2f}%"),
    ]
    ax_cov.legend(
        handles=metric_handles,
        loc="lower right",
        frameon=True,
        title="Metrics",
        handlelength=0,
        handletextpad=0,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(
    q50_h,
    targets_h,
    encoder_data,
    train_size,
    val_size,
    demand_mean,
    demand_std,
    save_path,
    model_name,
):
    test_start = train_size + val_size
    test_encoder = encoder_data[test_start:test_start + len(q50_h)]
    last_known = test_encoder[:, -1, 0].detach().cpu().numpy() * demand_std + demand_mean
    persist_pred = np.tile(last_known[:, None], (1, q50_h.shape[1]))

    def _r2(actual, pred):
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10)

    horizons = [(0, "1h"), (23, "24h"), (167, "168h")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (h_idx, h_label) in enumerate(horizons):
        actual = targets_h[:, h_idx]
        pred = q50_h[:, h_idx]
        min_v = min(actual.min(), pred.min())
        max_v = max(actual.max(), pred.max())

        ax = axes[0, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color="steelblue")
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.0)
        ax.set_title(f"{model_name} — LSTM ({h_label})\nR²={_r2(actual, pred):.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    for col, (h_idx, h_label) in enumerate(horizons):
        actual = targets_h[:, h_idx]
        pred = persist_pred[:, h_idx]
        min_v = min(actual.min(), pred.min())
        max_v = max(actual.max(), pred.max())

        ax = axes[1, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color="coral")
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.0)
        ax.set_title(f"{model_name} — Persistence ({h_label})\nR²={_r2(actual, pred):.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Actual vs Predicted (With Persistence Baseline)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_test_predictions(
    q10,
    q50,
    q90,
    targets,
    save_path,
    model_name,
):
    n_samples = q50.shape[0]
    n_panels = min(3, n_samples)
    sample_indices = np.linspace(0, n_samples - 1, n_panels, dtype=int)
    hours = np.arange(1, q50.shape[1] + 1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), squeeze=False)
    axes = axes[:, 0]

    for ax, idx in zip(axes, sample_indices):
        mae = float(np.mean(np.abs(targets[idx] - q50[idx])))
        coverage = float(np.mean((targets[idx] >= q10[idx]) & (targets[idx] <= q90[idx])) * 100.0)
        ax.plot(hours, targets[idx], label="Actual", color="blue", linewidth=1.5)
        ax.plot(hours, q50[idx], label="Median (q50)", color="red", linewidth=1.5)
        ax.fill_between(hours, q10[idx], q90[idx], color="red", alpha=0.18, label="Uncertainty bound")
        ax.set_title(f"{model_name} — Test Window {idx} (MAE={mae:.4f}, Coverage={coverage:.1f}%)")
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("abvaerk")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"{model_name} — Test Set Predictions (3 Example Windows)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(
    config,
    train_loader,
    val_loader,
    cal_loader,
    test_loader,
    train_size,
    val_size,
    cal_size,
    test_size,
    model_save_path,
    dataset_path,
    logger=None,
    patience=20,
    conformal_alpha=0.2,
    model_name="Model_4 Decoupled",
    training_variant="decoupled",
):
    """
    Full training loop with warmup, early stopping, and conformal calibration.

      val set  → early stopping / model selection only
      cal set  → conformal calibration (never seen during training)
      test set → final held-out evaluation (handled outside this function)

    conformal_alpha controls both the interval training loss and the
    conformal calibration step, ensuring they target the same coverage level.
    training_variant is stored for run traceability across side-by-side runs.
    """
    if training_variant not in {"coupled", "decoupled"}:
        raise ValueError(
            f"Unknown training_variant='{training_variant}'. "
            "Expected 'coupled' or 'decoupled'."
        )

    config.training_variant = training_variant
    model = LSTMForecast(config).to(config.device)
    model.training_variant = training_variant

    optimizer_point = torch.optim.AdamW([
        {'params': model.encoderLstm.parameters()},
        {'params': model.decoderLstm.parameters()},
        {'params': model.fc_decoderMedian.parameters()},
    ], lr=config.learning_rate, weight_decay=1e-4)

    optimizer_spread = torch.optim.AdamW([
        {'params': model.encoderLstm.parameters()},
        {'params': model.uncertainty_cell_proj.parameters()},
        {'params': model.uncertaintyDecoderLstm.parameters()},
        {'params': model.fc_uncertaintyDecoderLow.parameters()},
        {'params': model.fc_uncertaintyDecoderHigh.parameters()},
    ], lr=config.learning_rate, weight_decay=1e-4)

    scheduler_point = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_point, mode='min', factor=0.5, patience=10
    )
    scheduler_spread = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_spread, mode='min', factor=0.5, patience=10
    )

    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_cos_sims_epoch = []

    if logger is not None:
        logger.info(
            f"Starting training for {config.epochs} epochs  "
            f"patience={patience}  conformal_alpha={conformal_alpha} "
            f"(target coverage={(1 - conformal_alpha) * 100:.0f}%)  "
            f"variant={training_variant}"
        )

    for epoch in range(1, config.epochs + 1):

        # Linear warmup overrides scheduler for first 3 epochs
        warmup_lr = get_lr(epoch, warmup_epochs=3, base_lr=config.learning_rate)
        for param_group in optimizer_point.param_groups:
            param_group['lr'] = warmup_lr
        for param_group in optimizer_spread.param_groups:
            param_group['lr'] = warmup_lr

        train_loss, train_cos_sim = train_epoch(
            model, train_loader, optimizer_point, optimizer_spread,
            config.device, train_size, conformal_alpha
        )
        train_losses.append(train_loss)
        train_cos_sims_epoch.append(train_cos_sim)

        val_loss = validate_epoch(
            model, val_loader, config.device, val_size, conformal_alpha
        )
        val_losses.append(val_loss)

        if epoch > 3:
            scheduler_point.step(val_loss)
            scheduler_spread.step(val_loss)

        current_lr = optimizer_point.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_epoch        = epoch
            logger.info(
                f"Epoch {epoch}: Train={train_loss:.4f}  Val={val_loss:.4f}  "
                f"CosSim={train_cos_sim:.4f}  LR={current_lr:.2e}  [best — saving checkpoint]"
            )
            save_checkpoint(model, optimizer_point, optimizer_spread, config, epoch, val_loss,
                            train_losses, val_losses, model_save_path)
        else:
            logger.info(
                f"Epoch {epoch}: Train={train_loss:.4f}  Val={val_loss:.4f}  "
                f"CosSim={train_cos_sim:.4f}  LR={current_lr:.2e}  "
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
    checkpoint['train_cos_sims'] = train_cos_sims_epoch
    checkpoint['training_variant'] = training_variant

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

    # ── Plot Data (test + denormalized values) ────────────────────────────────
    q10_test_raw, q50_test_raw, q90_test_raw, tgt_test_raw = collect_predictions(
        model, test_loader, config.device
    )
    q10_test_cal, q90_test_cal = apply_conformal(q10_test_raw, q90_test_raw, u_alpha_t)

    dataset = torch.load(dataset_path, weights_only=False)
    demand_mean = float(dataset.get("demand_mean", 0.0))
    demand_std = float(dataset.get("demand_std", 1.0))
    encoder_data = dataset["encoder"]

    def denorm(arr):
        return arr * demand_std + demand_mean

    q10_cal_h = denorm(q10_test_cal)
    q50_h = denorm(q50_test_raw)
    q90_cal_h = denorm(q90_test_cal)
    tgt_h = denorm(tgt_test_raw)

    # ── Plots ──────────────────────────────────────────────────────────────────
    run_dir = os.path.dirname(model_save_path)
    plot_dir = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    loss_plot_path = os.path.join(plot_dir, "train_val_loss.png")
    cos_cov_plot_path = os.path.join(plot_dir, "cosine_similarity_and_coverage.png")
    scatter_plot_path = os.path.join(plot_dir, "actual_vs_predicted.png")
    test_preds_plot_path = os.path.join(plot_dir, "test_predictions.png")

    plot_train_val_loss(
        train_losses,
        val_losses,
        checkpoint['epoch'],
        loss_plot_path,
        model_name=model_name,
    )
    plot_cosine_and_coverage(
        train_cos_sims_epoch,
        q10_raw,
        q50_raw,
        q90_raw,
        tgt_cal,
        u_alpha_t,
        cos_cov_plot_path,
        model_name=model_name,
        alpha=conformal_alpha,
    )
    plot_actual_vs_predicted(
        q50_h,
        tgt_h,
        encoder_data,
        train_size,
        val_size,
        demand_mean,
        demand_std,
        scatter_plot_path,
        model_name=model_name,
    )
    plot_test_predictions(
        q10_cal_h,
        q50_h,
        q90_cal_h,
        tgt_h,
        test_preds_plot_path,
        model_name=model_name,
    )

    if logger is not None:
        logger.info(f"Loss curve saved to {loss_plot_path}")
        logger.info(f"Cosine + Coverage plot saved to {cos_cov_plot_path}")
        logger.info(f"Actual vs Predicted plot saved to {scatter_plot_path}")
        logger.info(f"Test set predictions plot saved to {test_preds_plot_path}")

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


def interval_score_loss(q_low, q_high, target, alpha):
    H = target.shape[1]
    weights = torch.linspace(1.0, 2.0, H, device=target.device).unsqueeze(0)
    width        = q_high - q_low
    penalty_low  = (2 / alpha) * torch.clamp(q_low - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean((width + penalty_low + penalty_high) * weights)
