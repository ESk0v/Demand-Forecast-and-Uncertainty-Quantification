import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
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

    val_size = valcal_size
    cal_size = valcal_size

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset   = Subset(full_dataset, range(i1, i2))
    cal_dataset   = Subset(full_dataset, range(i1, i2))
    test_dataset  = Subset(full_dataset, range(i2, i3))

    print(f"Split sizes — train: {train_size} val: {val_size} cal: {cal_size} test: {test_size}")

    return (train_dataset, val_dataset, cal_dataset, test_dataset,
            train_size, val_size, cal_size, test_size)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width        = q_high - q_low
    penalty_low  = (2 / alpha) * torch.clamp(q_low - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + VALIDATION (UNCHANGED LOGIC)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer_point, optimizer_spread, device, train_size, conformal_alpha=0.2):

    model.train()
    epoch_loss = 0

    q10_le_q50 = []
    q50_le_q90 = []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer_point.zero_grad()
        optimizer_spread.zero_grad()

        q10, q50, q90 = model(enc, dec)

        loss_median   = quantile_loss(q50, tgt, q=0.5)
        loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)

        crossing_penalty = (
            torch.mean(torch.relu(q10 - q50)) +
            torch.mean(torch.relu(q50 - q90))
        ) * 0.1

        loss_median.backward(retain_graph=True)
        (loss_interval + crossing_penalty).backward()

        optimizer_point.step()
        optimizer_spread.step()

        epoch_loss += (loss_median + loss_interval).item() * enc.size(0)

        q10_le_q50.append((q10 <= q50).float().mean().item())
        q50_le_q90.append((q50 <= q90).float().mean().item())

    print("q10<=q50 avg:", np.mean(q10_le_q50))
    print("q50<=q90 avg:", np.mean(q50_le_q90))

    return epoch_loss / train_size


def validate_epoch(model, val_loader, device, val_size, conformal_alpha=0.2):

    model.eval()
    val_loss_epoch = 0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

            q10, q50, q90 = model(enc, dec)

            loss_interval = interval_score_loss(q10, q90, tgt, alpha=conformal_alpha)
            loss_median   = quantile_loss(q50, tgt, q=0.5)

            val_loss_epoch += (0.6 * loss_interval + 0.4 * loss_median).item() * enc.size(0)

    return val_loss_epoch / val_size


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SAVE (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer_point, optimizer_spread,
                    config, epoch, val_loss,
                    train_losses, val_losses,
                    model_save_path):

    # ✅ FIX: ensure folder exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_point_state_dict': optimizer_point.state_dict(),
        'optimizer_spread_state_dict': optimizer_spread.state_dict(),
        'config': vars(config),
        'epoch': epoch,
        'val_loss': val_loss,

        # IMPORTANT FOR YOUR PLOT SCRIPT
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    torch.save(checkpoint, model_save_path)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (UNCHANGED BEHAVIOUR)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(config, train_loader, val_loader, cal_loader,
                train_size, val_size, cal_size,
                model_save_path, logger=None,
                patience=20, conformal_alpha=0.2):

    model = LSTMForecast(config).to(config.device)

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
    ], lr=config.learning_rate, weight_decay=1e-4)

    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(1, config.epochs + 1):

        train_loss = train_epoch(
            model, train_loader,
            optimizer_point, optimizer_spread,
            config.device, train_size,
            conformal_alpha
        )
        val_loss = validate_epoch(
            model, val_loader,
            config.device, val_size,
            conformal_alpha
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            save_checkpoint(
                model, optimizer_point, optimizer_spread,
                config, epoch, val_loss,
                train_losses, val_losses,
                model_save_path
            )

            print(f"[INFO] Epoch {epoch} Train={train_loss:.5f} Val={val_loss:.5f} [BEST]")
        else:
            print(f"[INFO] Epoch {epoch} Train={train_loss:.5f} Val={val_loss:.5f}")
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered")
                break

    print("[SUCCESS] Training completed")

    return best_val_loss, train_losses, val_losses