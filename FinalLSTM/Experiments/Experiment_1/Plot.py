"""
Plot.py
───────
Loads pre-trained model synthetic_data_1 (NO re-training) and generates a
clean reliability diagram with box plots.

Reliability diagram:
    x-axis  → nominal coverage level  (what we want to hit)
    y-axis  → empirical coverage       (what we actually hit)
    diagonal → perfect calibration line
    box plots → distribution of per-week average coverage across test weeks
                  - center line  : mean coverage across all weeks
                  - box edges    : 25th / 75th percentile of weekly averages
                  - whiskers     : min / max weekly average (highest & lowest week)

Usage:
    python Plot.py

Configure MODEL_CONFIGS below to point at your synthetic_data_1.
"""

import os
import sys
import importlib
from typing import Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader


# ─── Configuration ────────────────────────────────────────────────────────────
#
#  Each entry needs:
#    name            – label in the legend
#    module_dir      – folder containing LSTMModel.py and LSTMTraining.py
#    dataset_path    – .pt dataset file  (same split as used during training)
#    model_save_path – path to the saved checkpoint  (.pt)
#    conformal_alpha – 1 – target_coverage stored in the checkpoint
#
MODEL_CONFIGS = [
        dict(
        name            = "10% target",
        module_dir      = "Model_1",
        dataset_path    = "Model_1/data/dataset.pt",
        model_save_path = "Model_1/checkpoints/model.pt",
        conformal_alpha = 0.90,   # 1 - 0.50
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "30% target",
        module_dir      = "Model_2",
        dataset_path    = "Model_2/data/dataset.pt",
        model_save_path = "Model_2/checkpoints/model.pt",
        conformal_alpha = 0.70,   # 1 - 0.60
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "50% target",
        module_dir      = "Model_3",
        dataset_path    = "Model_3/data/dataset.pt",
        model_save_path = "Model_3/checkpoints/model.pt",
        conformal_alpha = 0.50,   # 1 - 0.70
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "70% target",
        module_dir      = "Model_4",
        dataset_path    = "Model_4/data/dataset.pt",
        model_save_path = "Model_4/checkpoints/model.pt",
        conformal_alpha = 0.30,   # 1 - 0.80
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "90% target",
        module_dir      = "Model_5",
        dataset_path    = "Model_5/data/dataset.pt",
        model_save_path = "Model_5/checkpoints/model.pt",
        conformal_alpha = 0.10,   # 1 - 0.90
        epochs          = 50,
        patience        = 5,
    ),
]

# Number of nominal coverage levels to evaluate (spread 0 → 1)
N_LEVELS = 21          # e.g. 0 %, 5 %, 10 %, … 100 %

# How many test samples per "week" for the box-plot breakdown.
# If your test set has T time steps and each week = 7 steps, set this to 7.
# Set to None to let the script guess (uses horizon length).
SAMPLES_PER_WEEK = None

OUTPUT_PLOT_PATH = "Plot_reliability.png"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _import_module(module_dir: str, module_name: str):
    """Import module_name from module_dir, bypassing any cached version."""
    main_dir       = os.path.abspath(os.path.dirname(__file__))
    abs_module_dir = os.path.abspath(os.path.join(main_dir, module_dir))

    if not os.path.isdir(abs_module_dir):
        raise FileNotFoundError(f"Module folder not found: {abs_module_dir}")

    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    sys.path.insert(0, abs_module_dir)

    if module_name in sys.modules:
        del sys.modules[module_name]

    return importlib.import_module(module_name)


def load_model_predictions(cfg: dict) -> dict:
    """
    Load a checkpoint and run it on the held-out test set.
    Returns raw q10, q90 predictions and targets.
    """
    print(f"[INFO] Loading checkpoint for '{cfg['name']}' …")

    lstm_training = _import_module(cfg["module_dir"], "LSTMTraining")
    lstm_model    = _import_module(cfg["module_dir"], "LSTMModel")

    load_and_split_dataset = lstm_training.load_and_split_dataset
    collect_predictions    = lstm_training.collect_predictions
    Config                 = lstm_model.Config
    LSTMForecast           = lstm_model.LSTMForecast

    main_dir        = os.path.abspath(os.path.dirname(__file__))
    dataset_path    = os.path.join(main_dir, cfg["dataset_path"])
    model_save_path = os.path.join(main_dir, cfg["model_save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.isfile(model_save_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_save_path}\n"
            f"Run main.py first to train the model."
        )

    # Load dataset — we only need the test split
    (_, _, test_dataset, _, _, _) = load_and_split_dataset(dataset_path)

    config     = Config()
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = min(4, os.cpu_count() or 1)

    test_loader = DataLoader(
        test_dataset,
        batch_size    = config.batch_size * 8,
        shuffle       = False,
        pin_memory    = (device == "cuda"),
        num_workers   = num_workers,
        persistent_workers = False,
    )

    checkpoint = torch.load(model_save_path, weights_only=False)
    model      = LSTMForecast(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    conformal_alpha = checkpoint.get("conformal_alpha", cfg["conformal_alpha"])

    q10_raw, q50_raw, q90_raw, targets = collect_predictions(
        model, test_loader, device
    )

    # Clean up so next model gets fresh imports
    abs_module_dir = os.path.abspath(
        os.path.join(main_dir, cfg["module_dir"])
    )
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    for name in ("LSTMTraining", "LSTMModel"):
        if name in sys.modules:
            del sys.modules[name]

    print(f"[OK]   '{cfg['name']}' — test shape: {targets.shape}")

    return dict(
        name            = cfg["name"],
        q10_raw         = q10_raw,    # (N, H)
        q90_raw         = q90_raw,    # (N, H)
        targets         = targets,    # (N, H)
        conformal_alpha = conformal_alpha,
    )


# ─── Coverage computation ─────────────────────────────────────────────────────
def coverage_boxplot_stats(
    q10_raw: np.ndarray,
    q90_raw: np.ndarray,
    targets: np.ndarray,
    nominal_levels: np.ndarray,
    samples_per_week: Optional[int] = None,
) -> dict:
    """
    For each nominal level p:
      1. Derive the empirical threshold at level p from non-conformity scores.
      2. Compute per-sample coverage  (bool).
      3. Group samples into "weeks" and compute weekly average coverage.
      4. Return statistics across weeks: mean, q25, q75, min, max.

    Returns a dict of arrays, each of length len(nominal_levels).
    """
    N, H = targets.shape

    # Non-conformity score per sample, averaged over forecast horizon
    scores = np.maximum(
        np.maximum(q10_raw - targets, targets - q90_raw), 0.0
    )  # (N, H)

    # Group samples into weeks
    spw = samples_per_week or H or 1
    n_weeks   = max(1, N // spw)
    week_ends = np.array_split(np.arange(N), n_weeks)

    stats = {k: np.empty(len(nominal_levels)) for k in
             ("mean", "q25", "q75", "wmin", "wmax")}

    for i, p in enumerate(nominal_levels):
        ta = 1.0 - p

        if ta <= 0.0:
            # All covered by definition
            weekly = np.ones(n_weeks)
        elif ta >= 1.0:
            weekly = np.zeros(n_weeks)
        else:
            # Threshold derived from ALL test samples (approximates cal set)
            thresh  = np.quantile(scores, 1.0 - ta, axis=0)  # (H,)
            covered = (scores <= thresh[np.newaxis, :])       # (N, H)
            # Weekly average: mean over samples in the week, then over horizons
            weekly  = np.array([
                covered[idx, :].mean()
                for idx in week_ends
            ])  # (n_weeks,)

        stats["mean"][i] = float(weekly.mean())
        stats["q25"][i]  = float(np.percentile(weekly, 25))
        stats["q75"][i]  = float(np.percentile(weekly, 75))
        stats["wmin"][i] = float(weekly.min())
        stats["wmax"][i] = float(weekly.max())

    return stats


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_reliability(results: list, output_path: str):
    """
    Clean white-background reliability diagram.

    Draws:
      - Dashed diagonal  → perfect calibration
      - One box plot per model, placed at the model's nominal target on the x-axis:
            whisker top    → highest weekly average coverage (best week)
            box top        → 75th percentile of weekly averages
            center line    → mean coverage across all weeks
            box bottom     → 25th percentile of weekly averages
            whisker bottom → lowest weekly average coverage (worst week)
    """

    # ── Color palette (colorblind-friendly, works on white) ───────────────────
    PALETTE = [
        "#1f77b4",   # muted blue
        "#d62728",   # brick red
        "#2ca02c",   # green
        "#ff7f0e",   # orange
        "#9467bd",   # purple
    ]

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    ax.set_facecolor("white")

    # ── Perfect-calibration diagonal ─────────────────────────────────────────
    diag = np.linspace(0, 100, 200)
    ax.plot(
        diag, diag,
        color="#888888", linestyle="--", linewidth=1.5,
        label="Perfect calibration", zorder=2
    )

    # ── Light under/over-coverage zones ──────────────────────────────────────
    ax.fill_between(diag, diag, 100, color="#2ca02c", alpha=0.04, zorder=0)
    ax.fill_between(diag, 0,    diag, color="#d62728", alpha=0.04, zorder=0)
    ax.text(68, 14, "Under-covered", color="#d62728",
            fontsize=8, alpha=0.7, style="italic")
    ax.text(4,  88, "Over-covered",  color="#2ca02c",
            fontsize=8, alpha=0.7, style="italic")

    # ── Per-model single box plot ─────────────────────────────────────────────
    box_width = 3.0   # width in % units on the x-axis

    print("\n" + "─" * 72)
    print(f"  {'Model':<20} | {'Target':>8} | {'Min':>8} | {'Mean':>8} | {'Max':>8}")
    print("─" * 72)

    for model_idx, res in enumerate(results):
        color          = PALETTE[model_idx % len(PALETTE)]
        nominal_target = (1.0 - res["conformal_alpha"]) * 100.0   # x position

        # Compute weekly coverage stats at ONLY this model's nominal target
        target_level   = np.array([1.0 - res["conformal_alpha"]])
        stats = coverage_boxplot_stats(
            res["q10_raw"], res["q90_raw"], res["targets"],
            target_level, SAMPLES_PER_WEEK,
        )

        mean = float(stats["mean"][0]) * 100
        q25  = float(stats["q25"][0])  * 100
        q75  = float(stats["q75"][0])  * 100
        wmin = float(stats["wmin"][0]) * 100
        wmax = float(stats["wmax"][0]) * 100
        xc   = nominal_target

        print(f"  {res['name']:<20} | Target: {nominal_target:5.1f}% | Min: {wmin:5.1f}% | Mean: {mean:5.1f}% | Max: {wmax:5.1f}%")

        label = (f"{res['name']}"
                 f", mean {mean:.1f}%")

        # Whisker: min → max
        ax.plot([xc, xc], [wmin, wmax],
                color=color, linewidth=1.5, zorder=3)
        # Whisker caps
        cap_w = box_width * 0.45
        for cap_y in (wmin, wmax):
            ax.plot([xc - cap_w, xc + cap_w], [cap_y, cap_y],
                    color=color, linewidth=1.5, zorder=3)
        # IQR box
        ax.add_patch(plt.Rectangle(
            (xc - box_width / 2, q25),
            box_width, q75 - q25,
            facecolor=color, alpha=0.30,
            edgecolor=color, linewidth=1.5, zorder=4
        ))
        # Mean line
        ax.plot([xc - box_width / 2, xc + box_width / 2],
                [mean, mean],
                color=color, linewidth=2.2, zorder=5)

        # Legend swatch
        ax.plot([], [], color=color, linewidth=2.2, label=label)

    print("─" * 72)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Nominal Coverage Level", fontsize=12, labelpad=8)
    ax.set_ylabel("Empirical Coverage", fontsize=12, labelpad=8)
    ax.set_title(
        "Reliability Diagram — Conformal Prediction Intervals",
        fontsize=13, fontweight="bold", pad=14
    )

    ax.grid(color="#e0e0e0", linewidth=0.7, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(colors="#444444", labelsize=10)

    # Legend
    ax.legend(fontsize=14, loc="upper left", framealpha=0.9,
              edgecolor="#cccccc")

    # Box-plot key (small annotation)
    ax.text(
        0.985, 0.03,
        "Box: IQR of weekly avg coverage\n"
        "Whiskers: min / max weekly avg\n"
        "─  Mean coverage",
        transform=ax.transAxes, fontsize=14,
        ha="right", va="bottom", color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SUCCESS] Reliability plot saved → {os.path.abspath(output_path)}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    results = []
    for cfg in MODEL_CONFIGS:
        print(f"\n══ Model: {cfg['name']} ══")
        result = load_model_predictions(cfg)
        results.append(result)

    print("\n[INFO] All synthetic_data_1 loaded. Generating reliability diagram …")
    plot_reliability(results, OUTPUT_PLOT_PATH)
    print(f"[DONE] {os.path.abspath(OUTPUT_PLOT_PATH)}")


if __name__ == "__main__":
    main()