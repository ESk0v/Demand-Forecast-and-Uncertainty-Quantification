"""
CoveragePlot/main.py
────────────────────
Trains N models (each with its own conformal alpha / coverage target),
runs predictions on the same held-out test set, and produces a single
reliability diagram:

    x-axis  → nominal coverage level  (0 % … 100 %)
    y-axis  → empirical coverage       (0 % … 100 %)
    diagonal → perfect calibration line

One curve per model, averaged over all forecast horizons.
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader

# ─── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Per-model configuration ──────────────────────────────────────────────────
#
#  Add / remove entries here to run more or fewer models.
#  Each entry needs:
#    name            – human-readable label shown in the plot legend
#    module_dir      – path to the folder containing LSTMModel.py,
#                      LSTMTraining.py and LSTMMain.py
#    dataset_path    – .pt dataset file used for training & evaluation
#    model_save_path – where the trained checkpoint (.pt) is written
#    conformal_alpha – 1 – target_coverage  (0.40 → 60 %, 0.20 → 80 %)
#    epochs          – training epochs
#    patience        – early-stopping patience

MODEL_CONFIGS = [
    dict(
        name            = "60 % target",
        module_dir      = "Model_1",
        dataset_path    = "Model_1/data/dataset.pt",
        model_save_path = "Model_1/checkpoints/model.pt",
        conformal_alpha = 0.40,   # 1 - 0.60
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "80 % target",
        module_dir      = "Model_2",
        dataset_path    = "Model_2/data/dataset.pt",
        model_save_path = "Model_2/checkpoints/model.pt",
        conformal_alpha = 0.20,   # 1 - 0.80
        epochs          = 50,
        patience        = 5,
    ),
]

# Where to save the final reliability plot
OUTPUT_PLOT_PATH = "CoveragePlot_reliability.png"


# ─── Minimal logger (drop-in if you don't have loguru) ────────────────────────
class _SimpleLogger:
    def info(self,    msg): print(f"[INFO]    {msg}")
    def success(self, msg): print(f"[SUCCESS] {msg}")
    def warning(self, msg): print(f"[WARN]    {msg}")
    def error(self,   msg): print(f"[ERROR]   {msg}")

logger = _SimpleLogger()


# ─── Training helper ──────────────────────────────────────────────────────────
def train_one_model(cfg: dict) -> dict:
    """
    Dynamically imports the correct LSTMModel / LSTMTraining from the
    model-specific subfolder, trains the model, and returns a dict with
    everything needed for the reliability diagram.
    """
    module_dir = cfg["module_dir"]

    # Resolve module_dir relative to main.py's own location so the script
    # works regardless of which directory you run it from.
    main_dir       = os.path.abspath(os.path.dirname(__file__))
    abs_module_dir = os.path.abspath(os.path.join(main_dir, module_dir))

    if not os.path.isdir(abs_module_dir):
        raise FileNotFoundError(
            f"Model folder not found: {abs_module_dir}\n"
            f"Expected 'module_dir' to be a subfolder next to main.py."
        )

    # Put the model's folder first so its local LSTMModel / LSTMTraining
    # are found before anything else on sys.path.
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    sys.path.insert(0, abs_module_dir)

    # Also ensure main.py's own directory is on the path.
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    import importlib

    # Remove any cached versions from a previous model iteration so we
    # always get a fresh load from the current model's folder.
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    lstm_training = importlib.import_module("LSTMTraining")
    lstm_model    = importlib.import_module("LSTMModel")

    load_and_split_dataset = lstm_training.load_and_split_dataset
    train_model            = lstm_training.train_model
    collect_predictions    = lstm_training.collect_predictions
    Config                 = lstm_model.Config
    LSTMForecast           = lstm_model.LSTMForecast

    set_seed(GLOBAL_SEED)

    # Resolve all paths relative to main.py's own directory
    dataset_path    = os.path.join(main_dir, cfg["dataset_path"])
    model_save_path = os.path.join(main_dir, cfg["model_save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Place your dataset.pt inside {os.path.join(abs_module_dir, 'data')}"
        )

    # Dataset split
    (train_dataset, val_dataset, cal_dataset, test_dataset,
     train_size, val_size, cal_size, test_size) = load_and_split_dataset(
        dataset_path
    )

    config         = Config()
    config.epochs  = cfg["epochs"]
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers    = min(4, os.cpu_count() or 1)

    def make_loader(ds, shuffle, bs):
        return DataLoader(
            ds,
            batch_size    = bs,
            shuffle       = shuffle,
            pin_memory    = (device == "cuda"),
            num_workers   = num_workers,
            persistent_workers = False,
        )

    train_loader = make_loader(train_dataset, True,  config.batch_size)
    val_loader   = make_loader(val_dataset,   False, config.batch_size * 8)
    cal_loader   = make_loader(cal_dataset,   False, config.batch_size * 8)
    test_loader  = make_loader(test_dataset,  False, config.batch_size * 8)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Training ──────────────────────────────────────────────────────────────
    logger.info(f"[{cfg['name']}] Starting training …")

    train_model(
        config, train_loader, val_loader, cal_loader,
        train_size, val_size, cal_size,
        model_save_path,
        logger       = logger,
        patience     = cfg["patience"],
        conformal_alpha = cfg["conformal_alpha"],
    )

    logger.success(f"[{cfg['name']}] Training done.")

    # ── Test-set predictions ───────────────────────────────────────────────────
    checkpoint = torch.load(model_save_path, weights_only=False)
    model = LSTMForecast(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    u_alpha_t       = checkpoint["conformal_u_alpha"]   # (H,)
    conformal_alpha = checkpoint.get("conformal_alpha", cfg["conformal_alpha"])

    q10_raw, q50_raw, q90_raw, targets = collect_predictions(model, test_loader, device)

    # Remove stale imports so the next model gets its own fresh copy
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    return dict(
        name            = cfg["name"],
        q10_raw         = q10_raw,
        q90_raw         = q90_raw,
        targets         = targets,
        u_alpha_t       = u_alpha_t,
        conformal_alpha = conformal_alpha,
    )


# ─── Reliability diagram ──────────────────────────────────────────────────────
def compute_empirical_coverage(q10_raw, q90_raw, targets, u_alpha_t,
                                nominal_levels: np.ndarray) -> np.ndarray:
    """
    For each nominal level p, derive symmetric conformal corrections that
    target that level and compute the fraction of test samples actually
    covered — averaged over all forecast horizons.

    u_alpha_t was computed for a specific conformal_alpha.  To evaluate at
    other nominal levels we scale u_alpha_t proportionally: the expansion
    needed for 90 % coverage is roughly (0.10/original_alpha) × u_alpha_t.
    """
    base_u = u_alpha_t  # shape (H,)
    empirical = np.empty(len(nominal_levels))

    for i, p in enumerate(nominal_levels):
        target_alpha = 1.0 - p
        # Scale the per-horizon expansion so it targets this nominal level.
        # Simple linear rescaling is standard for exchangeable conformal PI.
        scale    = target_alpha / (np.mean(base_u) + 1e-9) if target_alpha > 0 else 0.0
        u_scaled = base_u * (target_alpha / (1.0 - (1.0 - np.mean(base_u))))

        # Recompute scaling via the empirical quantile directly:
        # For each nominal level, find u such that exactly p of cal scores ≤ u.
        # Here we derive it from the already-stored u_alpha_t by linear interp.
        q10_cal = q10_raw - u_alpha_t[np.newaxis, :] * (1.0 - p) / max(1e-9, 1.0 - (1.0 - np.mean(base_u) / np.mean(base_u)))
        q90_cal = q90_raw + u_alpha_t[np.newaxis, :] * (1.0 - p) / max(1e-9, 1.0 - (1.0 - np.mean(base_u) / np.mean(base_u)))

        # Simpler, numerically robust approach:
        # scale u linearly so the expected coverage hits p
        alpha_orig = target_alpha   # what we want
        u_for_p = base_u * (alpha_orig / max(np.mean(base_u), 1e-6))
        # That's circular — use the direct score-quantile approach instead:
        # Reconstruct per-sample non-conformity scores from q10/q90
        scores = np.maximum(
            np.maximum(q10_raw - targets, targets - q90_raw), 0.0
        )  # (N, H)

        # For coverage p we need the (1-target_alpha) quantile of scores
        if target_alpha <= 0:
            q_thresh = np.zeros(scores.shape[1])
        elif target_alpha >= 1:
            q_thresh = np.full(scores.shape[1], np.inf)
        else:
            q_thresh = np.quantile(scores, 1.0 - target_alpha, axis=0)  # (H,)

        covered    = (scores <= q_thresh[np.newaxis, :])   # (N, H)
        empirical[i] = float(covered.mean())

    return empirical


def plot_reliability(results: list, output_path: str):
    """
    Reliability diagram.

      x-axis  → nominal coverage level  (0–100 %)
      y-axis  → empirical coverage       (0–100 %)
      dashed  → perfect-calibration diagonal

    One curve per model; shaded band shows ± std across horizons at each level.
    """

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    BG        = "#0f0f14"
    GRID      = "#1e1e28"
    DIAG      = "#e8e8e8"
    PALETTE   = ["#38bdf8", "#fb923c", "#a3e635", "#c084fc", "#f472b6"]
    FONT_MAIN = "monospace"

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
    ax.set_facecolor(BG)

    nominal_levels = np.linspace(0.0, 1.0, 101)

    # Perfect-calibration diagonal
    ax.plot(
        nominal_levels * 100, nominal_levels * 100,
        color=DIAG, linestyle="--", linewidth=1.4, alpha=0.55,
        label="Perfect calibration", zorder=2
    )

    # Fill between diagonal and axes to mark under/over-coverage zones
    ax.fill_between(
        nominal_levels * 100, nominal_levels * 100, 100,
        color="#22c55e", alpha=0.04
    )
    ax.fill_between(
        nominal_levels * 100, 0, nominal_levels * 100,
        color="#ef4444", alpha=0.04
    )

    for idx, res in enumerate(results):
        color = PALETTE[idx % len(PALETTE)]

        # Per-horizon empirical coverage at each nominal level
        scores   = np.maximum(
            np.maximum(res["q10_raw"] - res["targets"],
                       res["targets"] - res["q90_raw"]), 0.0
        )  # (N, H)
        n_horiz  = scores.shape[1]

        all_emp  = np.empty((len(nominal_levels), n_horiz))  # (levels, H)

        for h in range(n_horiz):
            for i, p in enumerate(nominal_levels):
                ta = 1.0 - p
                if ta <= 0:
                    all_emp[i, h] = 1.0
                elif ta >= 1:
                    all_emp[i, h] = 0.0
                else:
                    thresh         = np.quantile(scores[:, h], 1.0 - ta)
                    all_emp[i, h]  = float((scores[:, h] <= thresh).mean())

        mean_emp = all_emp.mean(axis=1)      # (levels,)
        std_emp  = all_emp.std(axis=1)       # (levels,)

        nominal_target = 1.0 - res["conformal_alpha"]
        ax.plot(
            nominal_levels * 100, mean_emp * 100,
            color=color, linewidth=2.2, zorder=4,
            label=f"{res['name']}  (target {nominal_target*100:.0f} %)"
        )
        ax.fill_between(
            nominal_levels * 100,
            (mean_emp - std_emp) * 100,
            (mean_emp + std_emp) * 100,
            color=color, alpha=0.13, zorder=3
        )

        # Mark the nominal target on the curve
        idx_target = np.argmin(np.abs(nominal_levels - nominal_target))
        ax.scatter(
            nominal_levels[idx_target] * 100,
            mean_emp[idx_target] * 100,
            s=70, color=color, zorder=6, edgecolors="white", linewidths=0.8
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    ax.tick_params(colors="#9ca3af", labelsize=10)
    ax.xaxis.label.set_color("#d1d5db")
    ax.yaxis.label.set_color("#d1d5db")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Nominal Coverage Level", fontsize=12, labelpad=10,
                  fontfamily=FONT_MAIN)
    ax.set_ylabel("Empirical Coverage (avg over horizons)", fontsize=12,
                  labelpad=10, fontfamily=FONT_MAIN)
    ax.set_title(
        "Reliability Diagram — Conformal Prediction Intervals",
        fontsize=14, color="#f9fafb", pad=16, fontfamily=FONT_MAIN, fontweight="bold"
    )

    ax.grid(color=GRID, linewidth=0.8, zorder=1)

    # Annotation boxes
    ax.text(
        72, 18, "Under-covered", color="#ef4444", fontsize=9,
        alpha=0.7, fontfamily=FONT_MAIN, style="italic"
    )
    ax.text(
        8, 82, "Over-covered", color="#22c55e", fontsize=9,
        alpha=0.7, fontfamily=FONT_MAIN, style="italic"
    )

    legend = ax.legend(
        fontsize=10, facecolor="#1a1a24", edgecolor=GRID,
        labelcolor="#e5e7eb", loc="upper left",
        prop={"family": FONT_MAIN}
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[SUCCESS] Reliability plot saved → {output_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    set_seed(GLOBAL_SEED)

    results = []
    for cfg in MODEL_CONFIGS:
        logger.info(f"══ Model: {cfg['name']} ══")
        result = train_one_model(cfg)
        results.append(result)

    logger.info("All models trained. Generating reliability diagram …")
    plot_reliability(results, OUTPUT_PLOT_PATH)
    logger.success(f"Done. Plot saved to: {os.path.abspath(OUTPUT_PLOT_PATH)}")


if __name__ == "__main__":
    main()