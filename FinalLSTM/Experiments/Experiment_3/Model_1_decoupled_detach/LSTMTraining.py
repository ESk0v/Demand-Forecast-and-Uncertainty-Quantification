import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from LSTMModel import LSTMForecast


# ============================================================
# Dataset
# ============================================================

def load_and_split_dataset(
    dataset_path,
    val_ratio=1/6,
    test_ratio=1/6,
):
    """
    Chronological split.

    train = 66.7%
    val   = 16.7%
    test  = 16.7%
    """

    dataset = torch.load(
        dataset_path,
        weights_only=False
    )

    full_dataset = TensorDataset(
        dataset['encoder'],
        dataset['decoder'],
        dataset['target'],
    )

    n_total = len(full_dataset)

    test_size = int(
        n_total * test_ratio
    )

    val_size = int(
        n_total * val_ratio
    )

    train_size = (
        n_total
        - val_size
        - test_size
    )

    i0 = 0
    i1 = train_size
    i2 = i1 + val_size
    i3 = i2 + test_size

    train_dataset = Subset(
        full_dataset,
        range(i0, i1)
    )

    val_dataset = Subset(
        full_dataset,
        range(i1, i2)
    )

    test_dataset = Subset(
        full_dataset,
        range(i2, i3)
    )

    print(
        f"Split sizes — "
        f"train: {train_size}  "
        f"val: {val_size}  "
        f"test: {test_size}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_size,
        val_size,
        test_size,
    )


# ============================================================
# LR Warmup
# ============================================================

def get_lr(
    epoch,
    warmup_epochs=3,
    base_lr=1e-4
):

    if epoch <= warmup_epochs:
        return (
            base_lr
            * (epoch / warmup_epochs)
        )

    return base_lr


# ============================================================
# Losses
# ============================================================

def quantile_loss(
    pred,
    target,
    q
):

    error = target - pred

    return torch.mean(
        torch.max(
            q * error,
            (q - 1) * error
        )
    )


def interval_score_loss(
    q_low,
    q_high,
    target,
    alpha=0.2
):

    width = q_high - q_low

    penalty_low = (
        2 / alpha
    ) * torch.clamp(
        q_low - target,
        min=0
    )

    penalty_high = (
        2 / alpha
    ) * torch.clamp(
        target - q_high,
        min=0
    )

    return torch.mean(
        width
        + penalty_low
        + penalty_high
    )


# ============================================================
# Gradient interference
# ============================================================

def _flatten_autograd_grads(
    grads
):

    vectors = []

    for grad in grads:

        if grad is None:
            continue

        vectors.append(
            grad.reshape(-1)
        )

    if not vectors:
        return None

    return torch.cat(
        vectors
    )


def compute_encoder_cosine_similarity(
    model,
    loss_median,
    loss_interval
):

    encoder_params = [
        p
        for p in model.encoderLstm.parameters()
        if p.requires_grad
    ]

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

    g1 = _flatten_autograd_grads(
        g1_raw
    )

    g2 = _flatten_autograd_grads(
        g2_raw
    )

    if g1 is None or g2 is None:
        return 0.0

    if g1.norm() == 0 or g2.norm() == 0:
        return 0.0

    return torch.nn.functional.cosine_similarity(
        g1.unsqueeze(0),
        g2.unsqueeze(0)
    ).item()


# ============================================================
# Train / Validate
# ============================================================

def train_epoch(
    model,
    train_loader,
    optimizer_point,
    optimizer_spread,
    device,
    train_size,
    alpha=0.2,
):

    model.train()

    epoch_loss = 0

    cosine_sims = []

    for enc, dec, tgt in train_loader:

        enc = enc.to(device)
        dec = dec.to(device)
        tgt = tgt.to(device)

        optimizer_point.zero_grad()
        optimizer_spread.zero_grad()

        q10, q50, q90 = model(
            enc,
            dec
        )

        loss_median = quantile_loss(
            q50,
            tgt,
            q=0.5
        )

        loss_interval = interval_score_loss(
            q10,
            q90,
            tgt,
            alpha=alpha
        )

        cos_sim = compute_encoder_cosine_similarity(
            model,
            loss_median,
            loss_interval
        )

        cosine_sims.append(
            cos_sim
        )

        q50_cross = (
            q50.detach()
            if model.training_variant == "decoupled"
            else q50
        )

        crossing_penalty = (
            torch.mean(
                torch.relu(
                    q10 - q50_cross
                )
            )
            +
            torch.mean(
                torch.relu(
                    q50_cross - q90
                )
            )
        ) * 0.1

        loss_spread = (
            loss_interval
            + crossing_penalty
        )

        loss_median.backward(
            retain_graph=True
        )

        loss_spread.backward()

        torch.nn.utils.clip_grad_norm_(
            model.encoderLstm.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.decoderLstm.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.fc_decoderMedian.parameters(),
            1.0
        )

        optimizer_point.step()

        torch.nn.utils.clip_grad_norm_(
            model.uncertaintyDecoderLstm.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.uncertainty_hidden_proj.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.uncertainty_cell_proj.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.fc_uncertaintyDecoderLow.parameters(),
            1.0
        )

        torch.nn.utils.clip_grad_norm_(
            model.fc_uncertaintyDecoderHigh.parameters(),
            1.0
        )

        optimizer_spread.step()

        epoch_loss += (
            loss_median
            + loss_interval
        ).item() * enc.size(0)

    avg_cos = float(
        np.mean(cosine_sims)
    )

    return (
        epoch_loss / train_size,
        avg_cos
    )


def validate_epoch(
    model,
    val_loader,
    device,
    val_size,
    alpha=0.2,
):

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for enc, dec, tgt in val_loader:

            enc = enc.to(device)
            dec = dec.to(device)
            tgt = tgt.to(device)

            q10, q50, q90 = model(
                enc,
                dec
            )

            loss_median = quantile_loss(
                q50,
                tgt,
                0.5
            )

            loss_interval = interval_score_loss(
                q10,
                q90,
                tgt,
                alpha=alpha
            )

            loss = (
                0.4 * loss_median
                +
                0.6 * loss_interval
            )

            val_loss += (
                loss.item()
                * enc.size(0)
            )

    return val_loss / val_size


# ============================================================
# Inference
# ============================================================

def collect_predictions(
    model,
    loader,
    device
):

    model.eval()

    all_q10 = []
    all_q50 = []
    all_q90 = []
    all_tgt = []

    with torch.no_grad():

        for enc, dec, tgt in loader:

            enc = enc.to(device)
            dec = dec.to(device)

            q10, q50, q90 = model(
                enc,
                dec
            )

            all_q10.append(
                q10.cpu().numpy()
            )

            all_q50.append(
                q50.cpu().numpy()
            )

            all_q90.append(
                q90.cpu().numpy()
            )

            all_tgt.append(
                tgt.numpy()
            )

    return (
        np.concatenate(all_q10),
        np.concatenate(all_q50),
        np.concatenate(all_q90),
        np.concatenate(all_tgt),
    )


# ============================================================
# Training
# ============================================================

def train_model(
    config,
    train_loader,
    val_loader,
    test_loader,
    train_size,
    val_size,
    test_size,
    model_save_path,
    logger=None,
    patience=20,
):

    model = LSTMForecast(
        config
    ).to(
        config.device
    )

    optimizer_point = torch.optim.AdamW(
        [
            {
                'params':
                model.encoderLstm.parameters()
            },
            {
                'params':
                model.decoderLstm.parameters()
            },
            {
                'params':
                model.fc_decoderMedian.parameters()
            },
        ],
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    optimizer_spread = torch.optim.AdamW(
        [
            {
                'params':
                model.uncertaintyDecoderLstm.parameters()
            },
            {
                'params':
                model.uncertainty_hidden_proj.parameters()
            },
            {
                'params':
                model.uncertainty_cell_proj.parameters()
            },
            {
                'params':
                model.fc_uncertaintyDecoderLow.parameters()
            },
            {
                'params':
                model.fc_uncertaintyDecoderHigh.parameters()
            },
        ],
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    best_val = np.inf
    best_epoch = 0
    no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(
        1,
        config.epochs + 1
    ):

        lr = get_lr(
            epoch,
            base_lr=config.learning_rate
        )

        for pg in optimizer_point.param_groups:
            pg['lr'] = lr

        for pg in optimizer_spread.param_groups:
            pg['lr'] = lr

        train_loss, cos_sim = train_epoch(
            model,
            train_loader,
            optimizer_point,
            optimizer_spread,
            config.device,
            train_size,
        )

        val_loss = validate_epoch(
            model,
            val_loader,
            config.device,
            val_size,
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        if logger:
            logger.info(
                f"Epoch {epoch} "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f} "
                f"Cos={cos_sim:.4f}"
            )

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(
                model.state_dict(),
                model_save_path
            )

        else:

            no_improve += 1

            if no_improve >= patience:
                break

    model.load_state_dict(
        torch.load(
            model_save_path
        )
    )

    return (
        best_val,
        train_losses,
        val_losses
    )