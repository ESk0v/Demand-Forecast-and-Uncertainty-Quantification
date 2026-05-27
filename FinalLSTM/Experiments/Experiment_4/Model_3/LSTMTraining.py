import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from LSTMModel import LSTMForecast


def load_and_split_dataset(dataset_path, val_ratio=1/12, cal_ratio=1/12, test_ratio=1/6):
    """
    Chronological 4-way split — no data leakage.

    val and cal intentionally use the same valcal pool (overlap).
    Neither set updates model weights.

    Effective proportions with defaults:
      train=66.7%  val=16.7%  cal=16.7%(same rows as val)  test=16.7%
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

        torch.nn.utils.clip_grad_norm_(model.uncertaintyDecoderLstm.parameters(),      max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderLow.parameters(),    max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.fc_uncertaintyDecoderHigh.parameters(),   max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.ramp_layer.parameters(),                  max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.uncertainty_cell_proj.parameters(),       max_norm=1.0)
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
        {'params': model.uncertaintyDecoderLstm.parameters()},
        {'params': model.fc_uncertaintyDecoderLow.parameters()},
        {'params': model.fc_uncertaintyDecoderHigh.parameters()},
        {'params': model.ramp_layer.parameters()},
        {'params': model.uncertainty_cell_proj.parameters()},
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
    checkpoint['conformal_alpha']   = conformal_alpha

    return best_val_loss, train_losses, val_losses


def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width        = q_high - q_low
    penalty_low  = (2 / alpha) * torch.clamp(q_low - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)