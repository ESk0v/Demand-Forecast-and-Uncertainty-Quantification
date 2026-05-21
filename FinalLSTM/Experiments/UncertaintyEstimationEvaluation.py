"""
Plot.py
───────
Loads pre-trained model checkpoints (NO re-training) and generates a
coverage box-plot chart.

Chart layout:
    x-axis  → model name
    y-axis  → empirical coverage
    horizontal dashed line → nominal target coverage
    box plots → distribution of per-week average coverage across test weeks
                  - center line  : mean coverage across all weeks
                  - box edges    : 25th / 75th percentile of weekly averages
                  - whiskers     : min / max weekly average (best & worst week)

Usage:
    python Plot.py
"""

import os
import importlib.util
from typing import Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader, TensorDataset


# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    dict(
        name            = "Model 1\nDecoupled Detach",
        module_dir      = "Experiment_3/Model_1_decoupled_detach",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_1_decoupled_detach/checkpoints/model_decoupled.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 2\nDecoupled No Detach",
        module_dir      = "Experiment_3/Model_2_decoupled_no_detach",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_2_decoupled_no_detach/checkpoints/model_decoupled.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 3\nCoupled 2 Loss",
        module_dir      = "Experiment_3/Model_3_coupled_2loss",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_3_coupled_2loss/checkpoints/model_coupled.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 4\nCoupled 2 Sub-loss",
        module_dir      = "Experiment_3/Model_4_coupled_2subloss",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_4_coupled_2subloss/checkpoints/model_coupled.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 6",
        module_dir      = "Experiment_3/Model_6",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_6/checkpoints/model.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 7",
        module_dir      = "Experiment_3/Model_7",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_7/checkpoints/model.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 8",
        module_dir      = "Experiment_3/Model_8",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_8/checkpoints/model.pt",
        conformal_alpha = 0.10,
    ),
    dict(
        name            = "Model 9",
        module_dir      = "Experiment_3/Model_9",
        dataset_path    = "Experiment_3/data/dataset.pt",
        model_save_path = "Experiment_3/Model_9/checkpoints/model.pt",
        conformal_alpha = 0.10,
    ),
]

SAMPLES_PER_WEEK = None
OUTPUT_PLOT_PATH = "Plot_reliability.png"

VAL_RATIO  = 1.0 / 12.0
CAL_RATIO  = 1.0 / 12.0
TEST_RATIO = 1.0 / 6.0

PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _abs(path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path))


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_test_dataset(dataset_path: str):
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    enc = data["encoder"]
    dec = data["decoder"]
    tgt = data["target"]

    n_total     = len(tgt)
    test_size   = int(n_total * TEST_RATIO)
    valcal_size = int(n_total * (VAL_RATIO + CAL_RATIO))
    train_size  = n_total - valcal_size - test_size

    i2 = train_size + valcal_size
    i3 = i2 + test_size

    test_ds = TensorDataset(enc[i2:i3], dec[i2:i3], tgt[i2:i3])
    return data, test_ds


def load_model_predictions(cfg: dict) -> dict:
    print(f"[INFO] Loading checkpoint for '{cfg['name']}' ...")

    dataset_path    = _abs(cfg["dataset_path"])
    model_save_path = _abs(cfg["model_save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.isfile(model_save_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_save_path}\n"
            f"Run main.py first to train the model."
        )

    safe_name   = cfg["name"].replace(" ", "_").replace("\n", "_")
    module_path = _abs(os.path.join(cfg["module_dir"], "LSTMModel.py"))
    lstm_model  = _load_module(f"lstm_model_{safe_name}", module_path)

    checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)

    config = lstm_model.Config()
    ckpt_config = checkpoint.get("config")
    if isinstance(ckpt_config, dict):
        for k, v in ckpt_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
    config.device = "cpu"

    model = lstm_model.LSTMForecast(config).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_ds = _split_test_dataset(dataset_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=max(256, config.batch_size * 8),
        shuffle=False,
    )

    q10_all, q90_all, tgt_all = [], [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            q10, _, q90 = model(enc, dec)
            q10_all.append(q10.cpu().numpy())
            q90_all.append(q90.cpu().numpy())
            tgt_all.append(tgt.cpu().numpy())

    q10_raw = np.concatenate(q10_all)
    q90_raw = np.concatenate(q90_all)
    targets = np.concatenate(tgt_all)

    conformal_alpha = checkpoint.get("conformal_alpha", cfg["conformal_alpha"])

    print(f"[OK]   '{cfg['name']}' — test shape: {targets.shape}")

    return dict(
        name            = cfg["name"],
        q10_raw         = q10_raw,
        q90_raw         = q90_raw,
        targets         = targets,
        conformal_alpha = conformal_alpha,
    )


# ─── Coverage computation ─────────────────────────────────────────────────────

def coverage_boxplot_stats(
    q10_raw: np.ndarray,
    q90_raw: np.ndarray,
    targets: np.ndarray,
    nominal_level: float,
    samples_per_week: Optional[int] = None,
) -> dict:
    """
    Compute weekly coverage statistics for a single nominal level.
    Returns mean, q25, q75, min, max across weeks (all in %).
    """
    N, H = targets.shape

    scores = np.maximum(
        np.maximum(q10_raw - targets, targets - q90_raw), 0.0
    )  # (N, H)

    spw      = samples_per_week or H or 1
    n_weeks  = max(1, N // spw)
    week_idx = np.array_split(np.arange(N), n_weeks)

    ta = 1.0 - nominal_level

    if ta <= 0.0:
        weekly = np.ones(n_weeks)
    elif ta >= 1.0:
        weekly = np.zeros(n_weeks)
    else:
        thresh  = np.quantile(scores, 1.0 - ta, axis=0)   # (H,)
        covered = scores <= thresh[np.newaxis, :]          # (N, H)
        weekly  = np.array([covered[idx, :].mean() for idx in week_idx])

    return dict(
        mean = float(weekly.mean())             * 100,
        q25  = float(np.percentile(weekly, 25)) * 100,
        q75  = float(np.percentile(weekly, 75)) * 100,
        wmin = float(weekly.min())              * 100,
        wmax = float(weekly.max())              * 100,
    )


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_coverage(results: list, output_path: str):
    """
    One box per model spread along the x-axis.
    A horizontal dashed line marks the shared nominal target.
    """
    n_models  = len(results)
    xs        = np.arange(n_models)
    box_width = 0.45

    # Collect coverage stats for every model
    all_stats = []
    for res in results:
        nominal = 1.0 - res["conformal_alpha"]
        stats   = coverage_boxplot_stats(
            res["q10_raw"], res["q90_raw"], res["targets"],
            nominal, SAMPLES_PER_WEEK,
        )
        all_stats.append(stats)

    # Unique target lines
    target_pcts = sorted(set(
        round((1.0 - res["conformal_alpha"]) * 100, 4) for res in results
    ))

    # y-axis range with breathing room
    all_vals = [v for s in all_stats for v in (s["wmin"], s["wmax"])]
    all_vals.extend(target_pcts)
    y_lo = max(0,   min(all_vals) - 3)
    y_hi = min(100, max(all_vals) + 4)   # +4 to leave room for mean labels

    fig, ax = plt.subplots(
        figsize=(max(10, n_models * 1.5), 7),
        facecolor="white",
    )
    ax.set_facecolor("white")

    # Horizontal target line(s)
    for tp in target_pcts:
        ax.axhline(
            tp,
            color="#888888", linestyle="--", linewidth=1.5,
            label=f"Target {tp:.4g}%",
            zorder=2,
        )

    # Subtle background bands
    primary = target_pcts[0]
    ax.axhspan(primary, 101,   color="#2ca02c", alpha=0.04, zorder=0)
    ax.axhspan(-1,      primary, color="#d62728", alpha=0.04, zorder=0)
    ax.text(n_models - 0.55, primary - 2.0, "Under-covered",
            color="#d62728", fontsize=8, alpha=0.7, style="italic", ha="right")
    ax.text(n_models - 0.55, primary + 0.6, "Over-covered",
            color="#2ca02c", fontsize=8, alpha=0.7, style="italic", ha="right")

    # Draw one box per model
    for i, (res, stats) in enumerate(zip(results, all_stats)):
        color = PALETTE[i % len(PALETTE)]
        xc    = xs[i]

        mean = stats["mean"]
        q25  = stats["q25"]
        q75  = stats["q75"]
        wmin = stats["wmin"]
        wmax = stats["wmax"]

        # Whisker line
        ax.plot([xc, xc], [wmin, wmax],
                color=color, linewidth=1.5, zorder=3)

        # Whisker caps
        cap_w = box_width * 0.40
        for cap_y in (wmin, wmax):
            ax.plot([xc - cap_w, xc + cap_w], [cap_y, cap_y],
                    color=color, linewidth=1.5, zorder=3)

        # IQR box
        ax.add_patch(plt.Rectangle(
            (xc - box_width / 2, q25),
            box_width, q75 - q25,
            facecolor=color, alpha=0.30,
            edgecolor=color, linewidth=1.5, zorder=4,
        ))

        # Mean line
        ax.plot(
            [xc - box_width / 2, xc + box_width / 2],
            [mean, mean],
            color=color, linewidth=2.2, zorder=5,
        )

        # Mean value label above the whisker
        ax.text(xc, wmax + 0.35, f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                color=color, fontweight="bold")

    # Axes
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"] for r in results], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlim(-0.65, n_models - 0.35)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Model", fontsize=12, labelpad=8)
    ax.set_ylabel("Empirical Coverage", fontsize=12, labelpad=8)
    ax.set_title(
        "Coverage Distribution per Model — Conformal Prediction Intervals",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(colors="#444444", labelsize=10)

    ax.legend(fontsize=10, loc="lower right", framealpha=0.9,
              edgecolor="#cccccc")

    # Box-plot key
    ax.text(
        0.015, 0.03,
        "Box: IQR of weekly avg coverage\n"
        "Whiskers: min / max weekly avg\n"
        "─  Mean coverage",
        transform=ax.transAxes, fontsize=7.5,
        ha="left", va="bottom", color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SUCCESS] Plot saved → {os.path.abspath(output_path)}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    results = []
    for cfg in MODEL_CONFIGS:
        print(f"\n══ Model: {cfg['name']} ══")
        result = load_model_predictions(cfg)
        results.append(result)

    print("\n[INFO] All checkpoints loaded. Generating coverage plot ...")
    plot_coverage(results, _abs(OUTPUT_PLOT_PATH))
    print(f"[DONE] {_abs(OUTPUT_PLOT_PATH)}")


if __name__ == "__main__":
    main()