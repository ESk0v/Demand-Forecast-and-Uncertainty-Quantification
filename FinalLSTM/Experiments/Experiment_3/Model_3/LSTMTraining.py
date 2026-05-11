import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import matplotlib.pyplot as plt

from LSTMModel import LSTMForecast


def load_and_split_dataset(dataset_path, val_ratio=1 / 12, cal_ratio=1 / 12, test_ratio=1 / 6):
    """
    Chronological 4-way split — no data leakage.

    Effective proportions with defaults:
      train=66.7%  val=16.7%  cal=16.7%
    """
    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset["encoder"]
    decoder_data = dataset["decoder"]
    target_data = dataset["target"]
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size = n_total - valcal_size - test_size

    i0 = 0
    i1 = train_size
    i2 = i1 + valcal_size
    i3 = i2 + test_size  # == n_total

    # Overlapping val/cal split from the same chronological pool.
    # This is intentional for this experiment setup.
    if (val_ratio + cal_ratio) == 0:
        raise ValueError("val_ratio + cal_ratio must be > 0")
    val_size = valcal_size
    cal_size = valcal_size

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset = Subset(full_dataset, range(i1, i2))
    cal_dataset = Subset(full_dataset, range(i1, i2))
    test_dataset = Subset(full_dataset, range(i2, i3))

    print(
        f"Split sizes — train: {train_size}  val: {val_size}  "
        f"cal: {cal_size}  test: {test_size}  (total: {n_total})"
    )

    return (
        train_dataset,
        val_dataset,
        cal_dataset,
        test_dataset,
        train_size,
        val_size,
        cal_size,
        test_size,
    )


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

    g1_norm = g1.norm()
    g2_norm = g2.norm()
    if g1_norm.item() == 0.0 or g2_norm.item() == 0.0:
        return 0.0

    return torch.nn.functional.cosine_similarity(
        g1.unsqueeze(0),
        g2.unsqueeze(0),
    ).item()


def train_epoch(
    model,
    train_loader,
    optimizer_point,
    optimizer_spread,
    device,
    train_size,
    conformal_alpha=0.2,
    median_loss_weight=1.0,
    interval_loss_weight=1.0,
):
    """
    One full pass over the training set.
    """
    model.train()
    epoch_loss = 0.0
    epoch_loss_median = 0.0
    epoch_loss_interval = 0.0
    q10_le_q50 = []
    q50_le_q90 = []
    step_cos_sims = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer_point.zero_grad()
        optimizer_spread.zero_grad()

        q10, q50, q90 = model(enc, dec)

        loss_median = quantile_loss(q50, tgt, q=0.5)
        loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)

        cos_sim = compute_encoder_cosine_similarity(model, loss_median, loss_interval)
        step_cos_sims.append(cos_sim)

        weighted_median = median_loss_weight * loss_median
        weighted_interval = interval_loss_weight * loss_interval

        weighted_median.backward(retain_graph=True)
        weighted_interval.backward()

        torch.nn.utils.clip_grad_norm_(model.encoderLstm.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.decoderLstm.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_decoderMedian.parameters(), max_norm=1.0)
        optimizer_point.step()

        torch.nn.utils.clip_grad_norm_(model.uncertaintyDecoderLstm.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderLow.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderHigh.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.ramp_net.parameters(), max_norm=1.0)
        optimizer_spread.step()

        epoch_loss += (weighted_median + weighted_interval).item() * enc.size(0)
        epoch_loss_median += loss_median.item() * enc.size(0)
        epoch_loss_interval += loss_interval.item() * enc.size(0)
        q10_le_q50.append((q10 <= q50).float().mean().item())
        q50_le_q90.append((q50 <= q90).float().mean().item())

    avg_cos_sim = float(np.mean(step_cos_sims)) if step_cos_sims else 0.0

    print("q10<=q50 avg:", np.mean(q10_le_q50))
    print("q50<=q90 avg:", np.mean(q50_le_q90))
    print(f"Avg Cosine Sim (encoderLstm): {avg_cos_sim:.4f}")

    return (
        epoch_loss / train_size,
        epoch_loss_median / train_size,
        epoch_loss_interval / train_size,
        avg_cos_sim,
        step_cos_sims,
    )


def validate_epoch(model, val_loader, device, val_size, conformal_alpha=0.2):
    """
    Validation loss — same weights as training for consistent early stopping.
    No noise, no dropout.
    """
    model.eval()
    val_median_epoch = 0.0
    val_interval_epoch = 0.0
    val_width_epoch = 0.0
    val_coverage_epoch = 0.0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

            q10, q50, q90 = model(enc, dec)

            loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)
            loss_median = quantile_loss(q50, tgt, q=0.5)
            val_median_epoch += loss_median.item() * enc.size(0)
            val_interval_epoch += loss_interval.item() * enc.size(0)
            val_width_epoch += (q90 - q10).mean().item() * enc.size(0)
            val_coverage_epoch += (
                ((tgt >= q10) & (tgt <= q90)).float().mean().item() * enc.size(0)
            )

    val_median = val_median_epoch / val_size
    val_interval = val_interval_epoch / val_size
    val_width = val_width_epoch / val_size
    val_coverage = val_coverage_epoch / val_size
    val_objective = 0.4 * val_median + 0.6 * val_interval

    return val_objective, val_median, val_interval, val_coverage, val_width


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
    scores = np.maximum(np.maximum(q10 - targets, targets - q90), 0)  # (N, H)

    n_horizons = scores.shape[1]

    # Flat alpha across all horizons — every horizon targets the same coverage
    u_alpha_t = np.array([np.quantile(scores[:, t], 1.0 - alpha) for t in range(n_horizons)])  # (H,)

    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    empirical_coverage = float(np.mean((targets >= q10_cal) & (targets <= q90_cal)))

    per_horizon_coverage = np.mean((targets >= q10_cal) & (targets <= q90_cal), axis=0)
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


def save_checkpoint(
    model,
    optimizer_point,
    optimizer_spread,
    config,
    epoch,
    val_loss,
    train_losses,
    val_losses,
    train_cos_sims_epoch,
    train_cos_sims_step,
    val_median_loss,
    val_interval_loss,
    val_coverage,
    val_width,
    model_save_path,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_point_state_dict": optimizer_point.state_dict(),
            "optimizer_spread_state_dict": optimizer_spread.state_dict(),
            "config": vars(config),
            "epoch": epoch,
            "val_loss": val_loss,
            "train_losses": train_losses.copy(),
            "val_losses": val_losses.copy(),
            "train_cos_sims": train_cos_sims_epoch.copy(),
            "train_cos_sims_step": train_cos_sims_step.copy(),
            "val_median_loss": val_median_loss,
            "val_interval_loss": val_interval_loss,
            "val_coverage": val_coverage,
            "val_width": val_width,
            "early_stop_metric": getattr(config, "early_stop_metric", "val_median"),
            "training_variant": getattr(config, "training_variant", "coupled"),
        },
        model_save_path,
    )


def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path):
    """Plot training vs validation loss curves and save to disk."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.5, marker="o", markersize=2)
    ax.plot(epochs_range, val_losses, label="Validation Loss", linewidth=1.5, marker="o", markersize=2)
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Quantile Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss (Quantile)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if len(all_losses) > 0 and min(all_losses) > 0 and max(all_losses) / min(all_losses) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("Quantile Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cosine_similarity(values, save_path, title, x_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_range = range(1, len(values) + 1)
    ax.plot(
        x_range,
        values,
        label="Gradient Cosine Sim",
        linewidth=1.2,
        marker="o",
        markersize=1.8,
        color="purple",
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_coverage_per_horizon(q10, q90, targets, u_alpha_t, save_path, alpha=0.1):
    """
    Plot raw vs calibrated per-horizon coverage.
    Generated automatically after every training run so you can immediately
    see whether the flat alpha calibration is working as intended.
    """
    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0)
    cal_coverage = np.mean((targets >= q10_cal) & (targets <= q90_cal), axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    horizons = np.arange(1, len(raw_coverage) + 1)
    ax.plot(horizons, raw_coverage * 100, label="Raw interval coverage", color="steelblue")
    ax.plot(horizons, cal_coverage * 100, label="Per-horizon calibrated coverage", color="green")
    ax.axhline(
        y=(1 - alpha) * 100,
        color="orange",
        linestyle="--",
        label=f"Nominal {(1 - alpha) * 100:.0f}% target",
    )
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Prediction Interval Coverage per Horizon\nRaw vs Conformal-Calibrated", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Coverage plot saved to {save_path}")


def train_model(
    config,
    train_loader,
    val_loader,
    cal_loader,
    train_size,
    val_size,
    cal_size,
    model_save_path,
    logger=None,
    patience=20,
    conformal_alpha=0.2,
    training_variant="coupled",
):
    """
    Full training loop with warmup, early stopping, and conformal calibration.

      val set  → early stopping / model selection only
      cal set  → conformal calibration (never seen during training)
      test set → final held-out evaluation (handled outside this function)

    training_variant:
      - "coupled":   both losses update encoder
      - "decoupled": uncertainty loss is detached from encoder
    """
    if training_variant not in {"coupled", "decoupled"}:
        raise ValueError(
            f"Unknown training_variant='{training_variant}'. "
            "Expected 'coupled' or 'decoupled'."
        )

    config.training_variant = training_variant
    config.early_stop_metric = "val_median"
    model = LSTMForecast(config).to(config.device)
    model.training_variant = training_variant

    optimizer_point = torch.optim.AdamW(
        [
            {"params": model.encoderLstm.parameters()},
            {"params": model.decoderLstm.parameters()},
            {"params": model.fc_decoderMedian.parameters()},
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    optimizer_spread = torch.optim.AdamW(
        [
            {"params": model.uncertaintyDecoderLstm.parameters()},
            {"params": model.fc_uncertaintyDecoderLow.parameters()},
            {"params": model.fc_uncertaintyDecoderHigh.parameters()},
            {"params": model.ramp_net.parameters()},
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler_point = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_point, mode="min", factor=0.5, patience=10
    )
    scheduler_spread = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_spread, mode="min", factor=0.5, patience=10
    )

    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_total_losses = []
    train_interval_losses = []
    val_interval_losses = []
    val_objective_losses = []
    val_coverages = []
    val_widths = []
    train_cos_sims_epoch = []
    train_cos_sims_step = []

    if logger is not None:
        logger.info(
            f"Starting training for {config.epochs} epochs  "
            f"patience={patience}  conformal_alpha={conformal_alpha}  "
            f"variant={training_variant}  early_stop={config.early_stop_metric}  "
            f"w_median={config.median_loss_weight:.2f}  "
            f"w_interval={config.interval_loss_weight:.2f}  "
            f"(target coverage={(1 - conformal_alpha) * 100:.0f}%)"
        )

    for epoch in range(1, config.epochs + 1):
        # Linear warmup overrides scheduler for first 3 epochs
        warmup_lr = get_lr(epoch, warmup_epochs=3, base_lr=config.learning_rate)
        for param_group in optimizer_point.param_groups:
            param_group["lr"] = warmup_lr
        for param_group in optimizer_spread.param_groups:
            param_group["lr"] = warmup_lr

        train_loss, train_loss_median, train_loss_interval, train_cos_sim_epoch, train_cos_sim_step_epoch = train_epoch(
            model,
            train_loader,
            optimizer_point,
            optimizer_spread,
            config.device,
            train_size,
            conformal_alpha,
            median_loss_weight=config.median_loss_weight,
            interval_loss_weight=config.interval_loss_weight,
        )
        train_losses.append(train_loss_median)
        train_total_losses.append(train_loss)
        train_interval_losses.append(train_loss_interval)
        train_cos_sims_epoch.append(train_cos_sim_epoch)
        train_cos_sims_step.extend(train_cos_sim_step_epoch)

        (
            val_objective,
            val_median,
            val_interval,
            val_coverage,
            val_width,
        ) = validate_epoch(model, val_loader, config.device, val_size, conformal_alpha)
        val_losses.append(val_median)
        val_objective_losses.append(val_objective)
        val_interval_losses.append(val_interval)
        val_coverages.append(val_coverage)
        val_widths.append(val_width)

        if epoch > 3:
            scheduler_point.step(val_median)
            scheduler_spread.step(val_median)

        current_lr = optimizer_point.param_groups[0]["lr"]

        if val_median < best_val_loss:
            best_val_loss = val_median
            epochs_no_improve = 0
            best_epoch = epoch
            if logger is not None:
                logger.info(
                    f"Epoch {epoch}: "
                    f"TrainMedian={train_loss_median:.4f}  "
                    f"TrainInterval={train_loss_interval:.4f}  "
                    f"TrainWeighted={train_loss:.4f}  "
                    f"ValMedian={val_median:.4f}  "
                    f"ValInterval={val_interval:.4f}  "
                    f"ValObj={val_objective:.4f}  "
                    f"ValCov={val_coverage:.3f}  "
                    f"ValWidth={val_width:.3f}  "
                    f"CosSim={train_cos_sim_epoch:.4f}  LR={current_lr:.2e}  "
                    f"[best — saving checkpoint]"
                )
            save_checkpoint(
                model,
                optimizer_point,
                optimizer_spread,
                config,
                epoch,
                val_median,
                train_losses,
                val_losses,
                train_cos_sims_epoch,
                train_cos_sims_step,
                val_median,
                val_interval,
                val_coverage,
                val_width,
                model_save_path,
            )
        else:
            if logger is not None:
                logger.info(
                    f"Epoch {epoch}: "
                    f"TrainMedian={train_loss_median:.4f}  "
                    f"TrainInterval={train_loss_interval:.4f}  "
                    f"TrainWeighted={train_loss:.4f}  "
                    f"ValMedian={val_median:.4f}  "
                    f"ValInterval={val_interval:.4f}  "
                    f"ValObj={val_objective:.4f}  "
                    f"ValCov={val_coverage:.3f}  "
                    f"ValWidth={val_width:.3f}  "
                    f"CosSim={train_cos_sim_epoch:.4f}  LR={current_lr:.2e}  "
                    f"(best epoch: {best_epoch})"
                )
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger is not None:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

    # Persist final curves into checkpoint
    checkpoint = torch.load(model_save_path, weights_only=False)
    checkpoint["train_losses"] = train_losses
    checkpoint["val_losses"] = val_losses
    checkpoint["train_total_losses"] = train_total_losses
    checkpoint["train_interval_losses"] = train_interval_losses
    checkpoint["val_interval_losses"] = val_interval_losses
    checkpoint["val_objective_losses"] = val_objective_losses
    checkpoint["val_coverages"] = val_coverages
    checkpoint["val_widths"] = val_widths
    checkpoint["train_cos_sims"] = train_cos_sims_epoch
    checkpoint["train_cos_sims_step"] = train_cos_sims_step
    checkpoint["early_stop_metric"] = config.early_stop_metric
    checkpoint["training_variant"] = training_variant

    # Conformal calibration
    model.load_state_dict(checkpoint["model_state_dict"])

    if logger is not None:
        logger.info(f"Running conformal calibration on cal set ({cal_size} samples) ...")

    q10_raw, q50_raw, q90_raw, tgt_cal = collect_predictions(model, cal_loader, config.device)
    _, _, u_alpha_t = conformal_calibration(
        q10_raw, q50_raw, q90_raw, tgt_cal, alpha=conformal_alpha
    )
    checkpoint["conformal_u_alpha"] = u_alpha_t
    checkpoint["conformal_alpha"] = conformal_alpha
    torch.save(checkpoint, model_save_path)

    if logger is not None:
        logger.success(
            f"Training complete. Best checkpoint: epoch {checkpoint['epoch']}  "
            f"val_median={checkpoint['val_loss']:.4f}  "
            f"conformal_u_alpha shape={u_alpha_t.shape}"
        )

    # Plots (variant-specific folder to avoid overwrite across checkpoints)
    run_dir = os.path.dirname(model_save_path)
    model_tag = os.path.splitext(os.path.basename(model_save_path))[0]
    plot_dir = os.path.join(run_dir, "Plots", model_tag)
    os.makedirs(plot_dir, exist_ok=True)

    loss_plot_path = os.path.join(plot_dir, "train_val_loss.png")
    coverage_plot_path = os.path.join(plot_dir, "coverage_per_horizon.png")
    cossim_epoch_plot_path = os.path.join(plot_dir, "gradient_cosine_similarity_epoch.png")
    cossim_step_plot_path = os.path.join(plot_dir, "gradient_cosine_similarity_step.png")

    plot_train_val_loss(train_losses, val_losses, checkpoint["epoch"], loss_plot_path)
    plot_coverage_per_horizon(
        q10_raw,
        q90_raw,
        tgt_cal,
        u_alpha_t,
        coverage_plot_path,
        alpha=conformal_alpha,
    )
    plot_cosine_similarity(
        train_cos_sims_epoch,
        cossim_epoch_plot_path,
        title=f"Encoder Gradient Cosine Similarity per Epoch ({training_variant})",
        x_label="Epoch",
    )
    plot_cosine_similarity(
        train_cos_sims_step,
        cossim_step_plot_path,
        title=f"Encoder Gradient Cosine Similarity per Step ({training_variant})",
        x_label="Training Step",
    )

    if logger is not None:
        logger.info(f"Loss curve saved to {loss_plot_path}")
        logger.info(f"Coverage plot saved to {coverage_plot_path}")
        logger.info(f"Cosine Similarity (epoch) plot saved to {cossim_epoch_plot_path}")
        logger.info(f"Cosine Similarity (step) plot saved to {cossim_step_plot_path}")

    return best_val_loss, train_losses, val_losses


def apply_conformal(q10, q90, u_alpha):
    """
    Apply saved conformal calibration to new predictions.

    Supports:
      - scalar u_alpha → same expansion for all horizons
      - vector u_alpha (per-horizon) → one expansion per forecast step
    """
    is_tensor = False
    device = None
    if isinstance(q10, torch.Tensor):
        is_tensor = True
        device = q10.device
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


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width = q_high - q_low
    penalty_low = (2 / alpha) * torch.clamp(q_low - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)
