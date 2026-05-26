import argparse
import importlib.util
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODULE_DIR  = "Model_2_decoupled_no_detach"
MODEL_NAME  = "Model_2_decoupled_no_detach"

CHECKPOINT_CANDIDATES = (
    "model_decoupled.pt",
    "model_coupled.pt",
    "model.pt",
)

EVAL_HORIZONS = {1: "1 h", 24: "24 h", 168: "168 h"}   # 1-indexed step → label


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_checkpoint(module_dir: str, run_subdir: str, explicit: Optional[str]) -> str:
    if explicit:
        p = _resolve(explicit)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return p
    run_dir = os.path.join(BASE_DIR, module_dir, run_subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    for name in CHECKPOINT_CANDIDATES:
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            return cand
    pts = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    if len(pts) == 1:
        return os.path.join(run_dir, pts[0])
    raise RuntimeError(f"Cannot auto-discover checkpoint in {run_dir}: {pts}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_splits(dataset_path, val_ratio=1/10, cal_ratio=1/10, test_ratio=1/10):
    """
    Chronological split:
      default: train = 70%  |  val+cal = 20%  |  test = 10%

    cal_ratio is included so the test boundary is computed identically
    to how it was during training, even though no cal_loader is created here.
    """
    ds   = torch.load(dataset_path, weights_only=False)
    full = TensorDataset(ds["encoder"], ds["decoder"], ds["target"])
    n    = len(full)

    n_test   = int(n * test_ratio)
    n_valcal = int(n * (val_ratio + cal_ratio))
    n_train  = n - n_valcal - n_test

    i1 = n_train
    i2 = i1 + n_valcal
    i3 = i2 + n_test

    print(
        f"Split sizes — train: {n_train}  val+cal: {n_valcal}  "
        f"test: {n_test}  (total: {n})"
    )

    return (
        Subset(full, range(0,  i1)),   # train
        Subset(full, range(i1, i2)),   # val/cal
        Subset(full, range(i2, i3)),   # test
        n_train, n_valcal, n_test,
    )


def split_ratios_for_dataset(dataset_path: str):
    """
    Split policy:
      - dataset.pt            -> val/cal/test = 0.1 / 0.1 / 0.1
      - any other dataset .pt -> val/cal/test = 1/12 / 1/12 / 1/6
    """
    if os.path.basename(dataset_path) == "dataset.pt":
        return 0.1, 0.1, 0.1
    return (1 / 12), (1 / 12), (1 / 6)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────
def load_model(module_dir: str, checkpoint_path: str, device: str):
    model_file = os.path.join(BASE_DIR, module_dir, "LSTMModel.py")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(model_file)
    mod  = _load_module(f"_m2_{os.getpid()}", model_file)
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
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def r2(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10))


def winkler_score(q10, q90, targets, alpha=0.1):
    """
    Per-sample, per-horizon Winkler score, then averaged.
    W = (U-L) + (2/α)·max(L-y, 0) + (2/α)·max(y-U, 0)
    """
    width    = q90 - q10
    pen_low  = np.maximum(q10 - targets, 0.0)
    pen_high = np.maximum(targets - q90, 0.0)
    return width + (2.0 / alpha) * (pen_low + pen_high)   # (n_samples, n_horizons)


def interval_stats(q10, q90, targets):
    """Average interval width and normalized interval width."""
    width        = q90 - q10
    avg_width    = float(np.mean(width))
    avg_pred_abs = float(np.mean(np.abs(targets)))
    norm_width   = avg_width / avg_pred_abs if avg_pred_abs > 0 else np.nan
    return avg_width, norm_width


# ──────────────────────────────────────────────────────────────────────────────
# Persistence baseline
# ──────────────────────────────────────────────────────────────────────────────
def build_persistence(encoder_data, test_start, n_test, demand_mean, demand_std, n_horizons):
    enc_np   = _to_np(encoder_data)
    test_enc = enc_np[test_start:test_start + n_test]
    last     = test_enc[:, -1, 0] * demand_std + demand_mean
    return np.tile(last[:, None], (1, n_horizons))   # (n_test, H)


# ──────────────────────────────────────────────────────────────────────────────
# ─── PLOTS ───────────────────────────────────────────────────────────────────
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


def _savefig(fig, path, dpi=150):
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 1. Error over forecast horizon ─────────────────────────────────────────────
def plot_error_over_horizon(q50_h, tgt_h, persist_h, save_path):
    H     = q50_h.shape[1]
    hours = np.arange(1, H + 1)

    rmse_model   = np.array([rmse(tgt_h[:, t], q50_h[:, t])    for t in range(H)])
    mae_model    = np.array([mae (tgt_h[:, t], q50_h[:, t])     for t in range(H)])
    rmse_persist = np.array([rmse(tgt_h[:, t], persist_h[:, t]) for t in range(H)])
    mae_persist  = np.array([mae (tgt_h[:, t], persist_h[:, t]) for t in range(H)])

    for metric_m, metric_b, name in [
        (rmse_model, rmse_persist, "RMSE"),
        (mae_model,  mae_persist,  "MAE"),
    ]:
        better_hours = hours[metric_m < metric_b]
        if len(better_hours):
            lowest_hour = int(hours[np.argmin(metric_m)])
            print(f"  [{name}] Model beats persistence at hours: {better_hours.tolist()}")
            print(f"  [{name}] Lowest model error at hour: {lowest_hour}")
        else:
            print(f"  [{name}] Model never beats persistence baseline.")

    day_ticks       = np.arange(24, H + 1, 24)
    day_tick_labels = [f"day {d // 24}" for d in day_ticks]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric_m, metric_b, ylabel, title in [
        (axes[0], rmse_model, rmse_persist, "RMSE", "RMSE over Forecast Horizon"),
        (axes[1], mae_model,  mae_persist,  "MAE",  "MAE over Forecast Horizon"),
    ]:
        ax.plot(hours, metric_m, color=STYLE["model_color"],
                lw=STYLE["lw"], label="Proposed model")
        ax.plot(hours, metric_b, color=STYLE["baseline_color"],
                lw=STYLE["lw"], linestyle="--", label="Persistence baseline")
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_tick_labels, rotation=45, ha="right")
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=STYLE["grid_alpha"])

    fig.suptitle("Error over Forecast Horizon", fontsize=14)
    _savefig(fig, save_path)


# 2. Reliability diagram ─────────────────────────────────────────────────────
def plot_reliability_diagram(q10, q90, targets, alpha, save_path, model_name):
    H        = targets.shape[1]
    horizons = np.arange(1, H + 1)
    coverage = np.mean((targets >= q10) & (targets <= q90), axis=0) * 100
    nominal  = (1 - alpha) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(horizons, coverage, color=STYLE["cal_color"],
            lw=STYLE["lw"], label="Empirical coverage")
    ax.axhline(y=nominal, color=STYLE["nominal_color"],
               linestyle="--", lw=STYLE["lw"], label=f"Nominal {nominal:.0f}%")
    ax.fill_between(horizons, nominal, coverage,
                    where=(coverage < nominal), alpha=0.15,
                    color=STYLE["baseline_color"], label="Under-coverage")
    ax.fill_between(horizons, nominal, coverage,
                    where=(coverage >= nominal), alpha=0.12,
                    color=STYLE["model_color"], label="Over-coverage")
    ax.set_ylim(50, 100)
    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"{model_name} – Reliability Diagram (Per-Horizon Coverage)")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=STYLE["grid_alpha"])
    _savefig(fig, save_path)

    return float(np.min(coverage)), float(np.max(coverage))


# 3. Train / val loss ─────────────────────────────────────────────────────────
def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, label="Train loss",      lw=STYLE["lw"], marker="o", markersize=2)
    ax.plot(ep, val_losses,   label="Validation loss", lw=STYLE["lw"], marker="o", markersize=2)
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} – Train vs Validation Loss")
    ax.legend(fontsize=14); ax.grid(True, alpha=STYLE["grid_alpha"])
    all_l = list(train_losses) + list(val_losses)
    if all_l and min(all_l) > 0 and max(all_l) / min(all_l) > 10:
        ax.set_yscale("log"); ax.set_ylabel("Loss (log scale)")
    _savefig(fig, save_path)


# 4. Cosine similarity ────────────────────────────────────────────────────────
def plot_cosine_similarity(cos_sims, save_path, model_name):
    fig, ax = plt.subplots(figsize=(14, 5))
    epochs = np.arange(1, len(cos_sims) + 1)
    ax.plot(epochs, cos_sims, color="purple", marker="o",
            lw=1.4, markersize=3, label="Avg cosine similarity / epoch")
    if len(cos_sims) > 0:
        y_lo = max(-1.0, min(cos_sims) - 0.1)
        y_hi = min(1.0,  max(cos_sims) + 0.1)
        if y_lo == y_hi: y_lo -= 0.1; y_hi += 0.1
        ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{model_name} – Gradient Interference (Encoder)")
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.legend()
    _savefig(fig, save_path)


# 5. Coverage per horizon ─────────────────────────────────────────────────────
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


# 6. Actual vs predicted scatter ─────────────────────────────────────────────
def plot_actual_vs_predicted(
    q50_h, tgt_h, encoder_data, train_size, val_size,
    demand_mean, demand_std, persist_h, save_path, model_name,
):
    horizons = [(h - 1, lbl) for h, lbl in EVAL_HORIZONS.items()]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (h_idx, h_lbl) in enumerate(horizons):
        actual = tgt_h[:, h_idx]; pred = q50_h[:, h_idx]
        lo = min(actual.min(), pred.min()); hi = max(actual.max(), pred.max())
        ax = axes[0, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color=STYLE["model_color"])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_title(f"Proposed Model ({h_lbl})\nR²={r2(actual, pred):.4f}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.grid(True, alpha=STYLE["grid_alpha"])

    for col, (h_idx, h_lbl) in enumerate(horizons):
        actual = tgt_h[:, h_idx]; pred = persist_h[:, h_idx]
        lo = min(actual.min(), pred.min()); hi = max(actual.max(), pred.max())
        ax = axes[1, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color="coral")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_title(f"Persistence Baseline ({h_lbl})\nR²={r2(actual, pred):.4f}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.grid(True, alpha=STYLE["grid_alpha"])

    fig.suptitle("Residual Distribution", fontsize=14)
    _savefig(fig, save_path)


# 7. Test prediction windows ──────────────────────────────────────────────────
def plot_test_predictions(q10, q50, q90, targets, save_path, model_name):
    n     = q50.shape[0]
    idxs  = np.linspace(0, n - 1, min(3, n), dtype=int)
    hours = np.arange(1, q50.shape[1] + 1)
    fig, axes = plt.subplots(len(idxs), 1, figsize=(14, 4 * len(idxs)), squeeze=False)
    for ax, idx in zip(axes[:, 0], idxs):
        m   = float(np.mean(np.abs(targets[idx] - q50[idx])))
        cov = float(np.mean((targets[idx] >= q10[idx]) & (targets[idx] <= q90[idx])) * 100)
        ax.plot(hours, targets[idx], label="Actual",       color="blue", lw=1.5)
        ax.plot(hours, q50[idx],     label="Median (q50)", color="red",  lw=1.5)
        ax.fill_between(hours, q10[idx], q90[idx],
                        color="red", alpha=0.18, label="Uncertainty band")
        ax.set_title(f"{model_name} – Test Window {idx}  (MAE={m:.4f}, Coverage={cov:.1f}%)")
        ax.set_xlabel("Forecast Horizon (hours)"); ax.set_ylabel("Value")
        ax.grid(True, alpha=STYLE["grid_alpha"]); ax.legend(fontsize=9)
    fig.suptitle(f"{model_name} – Test Set Predictions (3 Example Windows)", fontsize=14)
    _savefig(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Console report
# ──────────────────────────────────────────────────────────────────────────────
def print_metrics(q50_h, tgt_h, persist_h, q10_h, q90_h, alpha,
                  cov_min, cov_max, model_name):
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  NUMERIC RESULTS  –  {model_name}")
    print(sep)

    print("\n[1] RMSE / MAE / R² at key horizons")
    print(f"{'Horizon':>10}  {'RMSE model':>12}  {'MAE model':>12}  {'R² model':>10}  "
          f"{'RMSE pers':>12}  {'MAE pers':>12}  {'R² pers':>10}")
    for h_step, h_lbl in EVAL_HORIZONS.items():
        hi = h_step - 1
        a  = tgt_h[:, hi]; m = q50_h[:, hi]; b = persist_h[:, hi]
        print(f"{h_lbl:>10}  "
              f"{rmse(a, m):>12.4f}  {mae(a, m):>12.4f}  {r2(a, m):>10.4f}  "
              f"{rmse(a, b):>12.4f}  {mae(a, b):>12.4f}  {r2(a, b):>10.4f}")

    avg_w, norm_w = interval_stats(q10_h, q90_h, tgt_h)
    avg_pred      = float(np.mean(q50_h))
    print(f"\n[2] Interval statistics (raw, full test set)")
    print(f"    Average interval width (q90-q10) : {avg_w:.4f}")
    print(f"    Average prediction value (q50)   : {avg_pred:.4f}")
    print(f"    Normalized interval width         : {norm_w:.4f}  (width / |avg target|)")

    W       = winkler_score(q10_h, q90_h, tgt_h, alpha=alpha)
    avg_W   = float(np.mean(W))
    per_h_W = np.mean(W, axis=0)
    print(f"\n[3] Winkler score  (α={alpha},  raw intervals)")
    print(f"    Overall mean Winkler score        : {avg_W:.4f}")
    print(f"    Per key horizon:")
    for h_step, h_lbl in EVAL_HORIZONS.items():
        print(f"      {h_lbl:>6} : {per_h_W[h_step - 1]:.4f}")

    print(f"\n[4] Reliability diagram (per-horizon coverage)")
    print(f"    Min empirical coverage : {cov_min:.2f}%")
    print(f"    Max empirical coverage : {cov_max:.2f}%")
    print(f"    Nominal coverage       : {(1-alpha)*100:.1f}%")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model and generate all required outputs.")
    p.add_argument("--dataset",           default="data/dataset.pt")
    p.add_argument("--run-subdir",        default="original")
    p.add_argument("--checkpoint",        default=None)
    p.add_argument("--output-dir",        default=None)
    p.add_argument("--conformal-alpha",   type=float, default=None)
    p.add_argument("--eval-batch-size",   type=int,   default=512)
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path    = _resolve(args.dataset)
    checkpoint_path = _find_checkpoint(MODULE_DIR, args.run_subdir, args.checkpoint)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(dataset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, config, ckpt = load_model(MODULE_DIR, checkpoint_path, device)

    # ── Split policy by dataset type ──────────────────────────────────────────
    val_ratio, cal_ratio, test_ratio = split_ratios_for_dataset(dataset_path)
    print(
        f"Split ratios: val={val_ratio:.6f}  cal={cal_ratio:.6f}  "
        f"test={test_ratio:.6f}"
    )

    _train_ds, _val_ds, test_ds, train_size, val_size, test_size = load_splits(
        dataset_path,
        val_ratio  = val_ratio,
        cal_ratio  = cal_ratio,
        test_ratio = test_ratio,
    )

    def _loader(ds):
        return DataLoader(
            ds, batch_size=args.eval_batch_size, shuffle=False,
            pin_memory=(device == "cuda"),
            num_workers=min(4, os.cpu_count() or 1),
        )

    test_loader = _loader(test_ds)

    alpha = args.conformal_alpha or float(ckpt.get("conformal_alpha", 0.1))

    # ── Predictions ───────────────────────────────────────────────────────────
    q10_raw, q50_raw, q90_raw, tgt_raw = collect_preds(model, test_loader, device)

    # ── De-normalise ──────────────────────────────────────────────────────────
    blob     = torch.load(dataset_path, weights_only=False)
    d_mean   = float(blob.get("demand_mean", 0.0))
    d_std    = float(blob.get("demand_std",  1.0))
    enc_data = blob["encoder"]

    def dn(x): return x * d_std + d_mean

    q10_h = dn(q10_raw);  q50_h = dn(q50_raw)
    q90_h = dn(q90_raw);  tgt_h = dn(tgt_raw)

    # ── Persistence baseline ──────────────────────────────────────────────────
    persist_h = build_persistence(
        enc_data, train_size + val_size, test_size, d_mean, d_std, tgt_h.shape[1]
    )

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir  = os.path.dirname(checkpoint_path)
    plot_dir = _resolve(args.output_dir) if args.output_dir else os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {plot_dir}")

    plot_error_over_horizon(
        q50_h, tgt_h, persist_h,
        os.path.join(plot_dir, "error_over_horizon.png"),
    )

    cov_min, cov_max = plot_reliability_diagram(
        q10_h, q90_h, tgt_h, alpha,
        os.path.join(plot_dir, "reliability_diagram.png"), MODEL_NAME,
    )

    train_losses = list(ckpt.get("train_losses", []))
    val_losses   = list(ckpt.get("val_losses",   []))
    best_epoch   = int(ckpt.get("epoch", 1))
    if train_losses and val_losses:
        plot_train_val_loss(
            train_losses, val_losses, best_epoch,
            os.path.join(plot_dir, "train_val_loss.png"), MODEL_NAME,
        )
    else:
        print("[WARN] train/val losses missing – skipping train_val_loss.png")

    cos_sims = list(_to_np(ckpt.get("train_cos_sims", [])))
    if not cos_sims:
        cos_sims = [0.0] * len(train_losses)
    plot_cosine_similarity(
        cos_sims,
        os.path.join(plot_dir, "cosine_similarity.png"),
        MODEL_NAME,
    )

    plot_coverage_per_horizon(
        q10_raw, q90_raw, tgt_raw,
        os.path.join(plot_dir, "coverage_per_horizon.png"),
        alpha=alpha,
    )

    plot_actual_vs_predicted(
        q50_h, tgt_h, enc_data, train_size, val_size,
        d_mean, d_std, persist_h,
        os.path.join(plot_dir, "actual_vs_predicted.png"), MODEL_NAME,
    )

    plot_test_predictions(
        q10_h, q50_h, q90_h, tgt_h,
        os.path.join(plot_dir, "test_predictions.png"), MODEL_NAME,
    )

    # ── Numeric report ────────────────────────────────────────────────────────
    print_metrics(
        q50_h, tgt_h, persist_h,
        q10_h, q90_h, alpha,
        cov_min, cov_max, MODEL_NAME,
    )


if __name__ == "__main__":
    main()
