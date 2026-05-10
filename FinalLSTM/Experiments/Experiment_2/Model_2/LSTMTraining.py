import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


# -----------------------------
# DATA SPLIT
# -----------------------------
def load_and_split_dataset(dataset_path, val_ratio=1/12, cal_ratio=1/12, test_ratio=1/6):

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
    i3 = i2 + test_size

    val_dataset  = Subset(full_dataset, range(i1, i2))
    cal_dataset  = Subset(full_dataset, range(i1, i2))
    test_dataset = Subset(full_dataset, range(i2, i3))
    train_dataset = Subset(full_dataset, range(i0, i1))

    print(
        f"Split sizes — train: {train_size}  val: {valcal_size}  "
        f"cal: {valcal_size}  test: {test_size}"
    )

    return train_dataset, val_dataset, cal_dataset, test_dataset, train_size, valcal_size, valcal_size, test_size


# -----------------------------
# LOSS FUNCTIONS
# -----------------------------
def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width = q_high - q_low
    penalty_low = (2 / alpha) * torch.clamp(q_low - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)


# -----------------------------
# TRAINING
# -----------------------------
def train_epoch(model, train_loader, optimizer, device, train_size, alpha=0.2):

    model.train()
    epoch_loss = 0

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer.zero_grad()

        q10, q50, q90 = model(enc, dec)

        loss = (
            quantile_loss(q10, tgt, 0.1) +
            quantile_loss(q50, tgt, 0.5) +
            quantile_loss(q90, tgt, 0.9)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item() * enc.size(0)

    return epoch_loss / train_size


def validate_epoch(model, val_loader, device, val_size):

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

            q10, q50, q90 = model(enc, dec)

            loss = (
                quantile_loss(q10, tgt, 0.1) +
                quantile_loss(q50, tgt, 0.5) +
                quantile_loss(q90, tgt, 0.9)
            )

            val_loss += loss.item() * enc.size(0)

    return val_loss / val_size


# -----------------------------
# PREDICTIONS
# -----------------------------
def collect_predictions(model, loader, device):

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


# -----------------------------
# TRAIN LOOP
# -----------------------------
def train_model(config, train_loader, val_loader, cal_loader,
                train_size, val_size, cal_size,
                model_save_path, logger=None, patience=20, conformal_alpha=0.2):

    model = LSTMForecast(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    if logger:
        logger.info("Starting training (single-head quantile model)")

    for epoch in range(1, config.epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer,
            config.device, train_size, conformal_alpha
        )
        val_loss = validate_epoch(
            model, val_loader, config.device, val_size
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logger:
            logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses
            }, model_save_path)

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger:
                    logger.info("Early stopping triggered")
                break

    # -----------------------------
    # CALIBRATION STEP (unchanged structure, still works)
    # -----------------------------
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    q10, q50, q90, tgt = collect_predictions(model, cal_loader, config.device)

    # simple calibration (optional improvement later)
    u = np.quantile(np.maximum(q10 - tgt, tgt - q90), 0.9)

    checkpoint["conformal_u"] = u
    torch.save(checkpoint, model_save_path)

    return best_val, train_losses, val_losses