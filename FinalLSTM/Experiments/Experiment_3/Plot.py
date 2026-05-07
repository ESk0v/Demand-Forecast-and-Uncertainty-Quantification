"""
InterferencePlot.py
───────────────────
Runs two model variants per epoch and tracks gradient interference:
  - Model WITH detach  (your current architecture)
  - Model WITHOUT detach (baseline — no stop-gradient)

Produces a 4-panel plot showing per-epoch mean and max gradient
interference in encoderLstm and decoderLstm.

Usage:
    python InterferencePlot.py

Expects LSTMModel.py and LSTMTraining.py in the same directory,
and a dataset.pt file at DATASET_PATH.
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH    = "data/dataset.pt"
OUTPUT_PATH     = "gradient_interference.png"
EPOCHS          = 20
PATIENCE        = 5
SEED            = 42

# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Modified training loop that records interference ─────────────────────────
def train_epoch_with_interference(model, train_loader, optimizer_point,
                                   optimizer_spread, device, use_detach=True):
    """
    One training epoch. Returns dict of interference stats:
      enc_mean, enc_max, dec_mean, dec_max
    These are averaged across all batches in the epoch.
    """
    model.train()

    enc_means, enc_maxes = [], []
    dec_means, dec_maxes = [], []

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer_point.zero_grad()
        optimizer_spread.zero_grad()

        if use_detach:
            q10, q50, q90 = model(enc, dec)
        else:
            q10, q50, q90 = model.forward_no_detach(enc, dec)

        # Point loss backward
        loss_median = quantile_loss(q50, tgt, q=0.5)
        loss_median.backward(retain_graph=True)

        # Snapshot encoder/decoder gradients BEFORE spread backward
        enc_grad_before = None
        dec_grad_before = None
        if model.encoderLstm.weight_ih_l0.grad is not None:
            enc_grad_before = model.encoderLstm.weight_ih_l0.grad.clone()
        if model.decoderLstm.weight_ih_l0.grad is not None:
            dec_grad_before = model.decoderLstm.weight_ih_l0.grad.clone()

        # Spread loss backward
        loss_interval = interval_score_loss(q10, q90, tgt, alpha=0.2)
        loss_interval.backward()

        # Measure interference
        if enc_grad_before is not None and model.encoderLstm.weight_ih_l0.grad is not None:
            diff = (model.encoderLstm.weight_ih_l0.grad - enc_grad_before).abs()
            enc_means.append(diff.mean().item())
            enc_maxes.append(diff.max().item())
        else:
            enc_means.append(0.0)
            enc_maxes.append(0.0)

        if dec_grad_before is not None and model.decoderLstm.weight_ih_l0.grad is not None:
            diff = (model.decoderLstm.weight_ih_l0.grad - dec_grad_before).abs()
            dec_means.append(diff.mean().item())
            dec_maxes.append(diff.max().item())
        else:
            dec_means.append(0.0)
            dec_maxes.append(0.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_point.step()
        optimizer_spread.step()

    return dict(
        enc_mean = float(np.mean(enc_means)),
        enc_max  = float(np.mean(enc_maxes)),   # mean of per-batch maxes
        dec_mean = float(np.mean(dec_means)),
        dec_max  = float(np.mean(dec_maxes)),
    )


def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


def interval_score_loss(q_low, q_high, target, alpha=0.2):
    width        = q_high - q_low
    penalty_low  = (2 / alpha) * torch.clamp(q_low  - target, min=0)
    penalty_high = (2 / alpha) * torch.clamp(target - q_high, min=0)
    return torch.mean(width + penalty_low + penalty_high)


# ─── Run experiment ───────────────────────────────────────────────────────────
def run_experiment(use_detach: bool, model_class, config_class,
                   train_loader, device, epochs):
    """Train a model for N epochs and collect interference stats per epoch."""
    set_seed(SEED)
    model = model_class(config_class()).to(device)

    # Patch forward_no_detach if needed
    if not use_detach:
        _patch_no_detach(model)

    optimizer_point = torch.optim.AdamW([
        {'params': model.encoderLstm.parameters()},
        {'params': model.decoderLstm.parameters()},
        {'params': model.fc_decoderMedian.parameters()},
    ], lr=1e-4, weight_decay=1e-4)

    optimizer_spread = torch.optim.AdamW([
        {'params': model.uncertaintyDecoderLstm.parameters()},
        {'params': model.fc_uncertaintyDecoderLow.parameters()},
        {'params': model.fc_uncertaintyDecoderHigh.parameters()},
    ], lr=1e-4, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        stats = train_epoch_with_interference(
            model, train_loader, optimizer_point, optimizer_spread,
            device, use_detach=use_detach
        )
        history.append(stats)
        print(f"  Epoch {epoch:3d}  enc_mean={stats['enc_mean']:.2e}  "
              f"enc_max={stats['enc_max']:.2e}  "
              f"dec_mean={stats['dec_mean']:.2e}  "
              f"dec_max={stats['dec_max']:.2e}")

    return history


def _patch_no_detach(model):
    """
    Add a forward_no_detach method that removes all .detach() calls,
    allowing gradients to flow freely from spread loss into the encoder.
    """
    def forward_no_detach(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoderLstm(encoder_input)
        hidden = self.context_dropout(hidden)
        cell   = self.context_dropout(cell)

        # No detach on decoder init
        decoder_h0 = torch.tanh(self.decoder_h(hidden[-1])).unsqueeze(0)
        decoder_c0 = torch.tanh(self.decoder_c(cell[-1])).unsqueeze(0)
        decoder_output, _ = self.decoderLstm(decoder_input, (decoder_h0, decoder_c0))
        decoder_output    = self.dropout(decoder_output)
        q50               = self.fc_decoderMedian(decoder_output).squeeze(-1)

        # No detach on spread init either
        spread_h0 = torch.tanh(self.uncertaintyDecoderLstm_h(hidden[-1])).unsqueeze(0)
        spread_c0 = torch.tanh(self.uncertaintyDecoderLstm_c(cell[-1])).unsqueeze(0)
        spread_output, _ = self.uncertaintyDecoderLstm(decoder_input, (spread_h0, spread_c0))
        spread_output     = self.dropout(spread_output)

        spread_lo = nn.functional.softplus(
            self.fc_uncertaintyDecoderLow(spread_output).squeeze(-1))
        spread_hi = nn.functional.softplus(
            self.fc_uncertaintyDecoderHigh(spread_output).squeeze(-1))

        horizon_pos = decoder_input[:, :, -1:]
        ramp = torch.sigmoid(self.ramp_layer(horizon_pos)).squeeze(-1)

        q10 = q50 - spread_lo * ramp
        q90 = q50 + spread_hi * ramp
        return q10, q50, q90

    import types
    model.forward_no_detach = types.MethodType(forward_no_detach, model)


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_interference(history_detach, history_no_detach, output_path):
    """
    4-panel plot:
      Top-left    : encoderLstm mean interference
      Top-right   : encoderLstm max  interference
      Bottom-left : decoderLstm mean interference
      Bottom-right: decoderLstm max  interference
    """
    epochs = np.arange(1, len(history_detach) + 1)

    keys   = ["enc_mean", "enc_max", "dec_mean", "dec_max"]
    titles = [
        "Encoder LSTM — Mean gradient interference",
        "Encoder LSTM — Max gradient interference",
        "Decoder LSTM — Mean gradient interference",
        "Decoder LSTM — Max gradient interference",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
    axes = axes.flatten()

    COLOR_DETACH    = "#1f77b4"   # blue  — with detach
    COLOR_NO_DETACH = "#d62728"   # red   — without detach

    for ax, key, title in zip(axes, keys, titles):
        y_det  = [s[key] for s in history_detach]
        y_nodet= [s[key] for s in history_no_detach]

        ax.plot(epochs, y_det,   color=COLOR_DETACH,    linewidth=2.0,
                marker='o', markersize=3, label="With detach (your model)")
        ax.plot(epochs, y_nodet, color=COLOR_NO_DETACH, linewidth=2.0,
                marker='s', markersize=3, label="Without detach (baseline)")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Gradient difference (abs)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(color="#e8e8e8", linewidth=0.7)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        # Shade near-zero region to visually anchor "no interference"
        ax.axhline(y=0, color="#aaaaaa", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Gradient Interference Analysis\n"
        "Effect of stop-gradient (detach) on encoder and decoder gradient updates",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SUCCESS] Interference plot saved → {os.path.abspath(output_path)}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # ── Imports ───────────────────────────────────────────────────────────────
    main_dir = os.path.abspath(os.path.dirname(__file__))
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from LSTMModel    import LSTMForecast, Config
    from LSTMTraining import load_and_split_dataset

    dataset_path = os.path.join(main_dir, DATASET_PATH)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    (train_dataset, val_dataset, cal_dataset, test_dataset,
     train_size, val_size, cal_size, test_size) = load_and_split_dataset(dataset_path)

    config  = Config()
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        pin_memory  = (device == "cuda"),
        num_workers = min(4, os.cpu_count() or 1),
    )

    print("\n══ Running: WITH detach ══")
    history_detach = run_experiment(
        use_detach=True,
        model_class=LSTMForecast,
        config_class=Config,
        train_loader=train_loader,
        device=device,
        epochs=EPOCHS,
    )

    print("\n══ Running: WITHOUT detach ══")
    history_no_detach = run_experiment(
        use_detach=False,
        model_class=LSTMForecast,
        config_class=Config,
        train_loader=train_loader,
        device=device,
        epochs=EPOCHS,
    )

    output_path = os.path.join(main_dir, OUTPUT_PATH)
    plot_interference(history_detach, history_no_detach, output_path)


if __name__ == "__main__":
    main()