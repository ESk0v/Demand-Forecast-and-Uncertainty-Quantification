import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# Plotting Style & Constants
# ──────────────────────────────────────────────────────────────────────────────
STYLE = dict(
    model_color    = "#2563EB",   # blue
    baseline_color = "#DC2626",   # red
    band_color     = "#93C5FD",   # light blue
    nominal_color  = "#F59E0B",   # amber
    cal_color      = "#16A34A",   # green
    raw_color      = "#6B7280",   # grey
    grid_alpha     = 0.25,
    lw             = 1.8,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _savefig(fig, path, dpi=150):
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_splits(dataset_path, val_ratio=None, cal_ratio=None, test_ratio=None):
    ds   = torch.load(dataset_path, weights_only=False)
    full = TensorDataset(ds["encoder"], ds["decoder"], ds["target"])
    
    # Force exact alignment with plot.py
    n_train  = 11472
    n_valcal = 2867
    n_test   = 2867

    i1 = n_train
    i2 = i1 + n_valcal
    i3 = i2 + n_test

    return (
        Subset(full, range(0,  i1)),   # train
        Subset(full, range(i1, i2)),   # val
        Subset(full, range(i2, i3)),   # test
        n_train, n_valcal, n_test,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str):
    # Dynamically find the model directory from the checkpoint path
    # Assuming path is like: Model_5/checkpoints/model.pt
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    module_dir = os.path.dirname(ckpt_dir)
    model_file = os.path.join(module_dir, "LSTMModel.py")
    
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Expected to find {model_file} but it does not exist.")
        
    mod  = _load_module(f"_model_{os.getpid()}", model_file)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg  = mod.Config()
    for k, v in (ckpt.get("config", {}) or {}).items():
        setattr(cfg, k, v)
    cfg.device = device
    
    if "training_variant" in ckpt:
        setattr(cfg, "training_variant", ckpt["training_variant"])
        
    model = mod.LSTMForecast(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt

@torch.no_grad()
def collect_preds(model, loader, device):
    q10s, q50s, q90s, tgts = [], [], [], []
    for enc, dec, tgt in loader:
        enc, dec = enc.to(device), dec.to(device)
        q10, q50, q90 = model(enc, dec)
        q10s.append(_to_np(q10)); q50s.append(_to_np(q50))
        q90s.append(_to_np(q90)); tgts.append(_to_np(tgt))
    return (np.concatenate(q10s), np.concatenate(q50s),
            np.concatenate(q90s), np.concatenate(tgts))

# ──────────────────────────────────────────────────────────────────────────────
# Target Plot Function
# ──────────────────────────────────────────────────────────────────────────────
def plot_coverage_per_horizon(q10_test, q90_test, tgt_test, save_path, alpha=0.1):
    raw_cov = np.mean((tgt_test >= q10_test) & (tgt_test <= q90_test), axis=0) * 100
    nominal = (1 - alpha) * 100
    gen_cov = float(np.mean((tgt_test >= q10_test) & (tgt_test <= q90_test)) * 100)

    H          = tgt_test.shape[1]
    hours      = np.arange(1, H + 1)
    day_ticks  = np.arange(24, H + 1, 24)
    day_labels = [f"day {d // 24}" for d in day_ticks]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hours, raw_cov, color="steelblue", lw=STYLE["lw"], label="Empirical coverage")
    ax.axhline(y=nominal, color=STYLE["nominal_color"],
               linestyle="--", lw=STYLE["lw"], label=f"Nominal {nominal:.0f}%")
    ax.axhline(y=gen_cov, color="steelblue", linestyle=":",
               lw=1.2, label=f"General coverage: {gen_cov:.2f}%")
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels, rotation=45, ha="right")
    ax.set_ylim(75, 100)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Average Coverage per Horizon")
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.legend(loc="lower left", fontsize=14)
    _savefig(fig, save_path)

# ──────────────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to model.pt (e.g., Model_5/checkpoints/model.pt)")
    p.add_argument("--dataset", default="FinalLSTM/Experiments/Experiment_1/Model_5/data/dataset.pt", help="Path to dataset.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--alpha", type=float, default=None, help="Override conformal alpha")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")

    # Load model
    model, config, ckpt = load_model(args.checkpoint, device)
    
    # Load data
    _, _, test_ds, _, _, _ = load_splits(args.dataset)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=min(4, os.cpu_count() or 1),
    )

    # Inference
    print("Running inference on test set...")
    q10_raw, _, q90_raw, tgt_raw = collect_preds(model, test_loader, device)

    # Handle Alpha
    alpha = args.alpha if args.alpha is not None else float(ckpt.get("conformal_alpha", 0.1))

    # Plot & Save to Current Directory
    save_path = os.path.join(os.getcwd(), "coverage_per_horizon.png")
    print("Generating coverage plot...")
    plot_coverage_per_horizon(q10_raw, q90_raw, tgt_raw, save_path, alpha=alpha)

if __name__ == "__main__":
    main()