import os
import sys
import importlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset


# ──────────────────────────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    dict(
        name="Baseline (UncertaintyDecoder)",
        module_dir="Model_1",
        dataset_path="Model_1/data/dataset.pt",
        model_save_path="Model_1/checkpoints/model.pt",
        color_median="#2563EB",
        scatter_color="#2563EB",
    ),
    dict(
        name="Ablated (Direct Output Head)",
        module_dir="Model_2",
        dataset_path="Model_2/data/dataset.pt",
        model_save_path="Model_2/checkpoints/model.pt",
        color_median="#DC2626",
        scatter_color="#DC2626",
    ),
]

RESIDUAL_HORIZONS = [0, 23, 167]
RESIDUAL_LABELS   = ["1h Ahead (horizon 0)",
                      "24h Ahead (horizon 23)",
                      "168h Ahead (horizon 167)"]

MAIN_DIR         = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR       = os.path.join(MAIN_DIR, "plots")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NOMINAL_COVERAGE = 0.90


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
class _L:
    def info(self, m):    print(f"[INFO]    {m}")
    def success(self, m): print(f"[SUCCESS] {m}")
    def warning(self, m): print(f"[WARN]    {m}")
    def error(self, m):   print(f"[ERROR]   {m}")

log = _L()


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic imports
# ──────────────────────────────────────────────────────────────────────────────
def _load_modules(module_dir: str):
    abs_dir = os.path.abspath(os.path.join(MAIN_DIR, module_dir))
    if abs_dir in sys.path:
        sys.path.remove(abs_dir)
    sys.path.insert(0, abs_dir)
    if MAIN_DIR not in sys.path:
        sys.path.insert(0, MAIN_DIR)
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    m_model    = importlib.import_module("LSTMModel")
    m_training = importlib.import_module("LSTMTraining")
    return m_model, m_training


def _unload_modules(module_dir: str):
    abs_dir = os.path.abspath(os.path.join(MAIN_DIR, module_dir))
    if abs_dir in sys.path:
        sys.path.remove(abs_dir)
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ──────────────────────────────────────────────────────────────────────────────
# Test subset — mirrors LSTMTraining.load_and_split_dataset exactly
# ──────────────────────────────────────────────────────────────────────────────
def get_test_subset(dataset_path, val_ratio=1/12, cal_ratio=1/12, test_ratio=1/6):
    raw         = torch.load(dataset_path, weights_only=False)
    full        = TensorDataset(raw["encoder"], raw["decoder"], raw["target"])
    n_total     = len(full)
    test_size   = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size  = n_total - valcal_size - test_size
    test_start  = train_size + valcal_size
    test_end    = test_start + test_size
    return (
        Subset(full, range(test_start, test_end)),
        float(raw["demand_mean"]),
        float(raw["demand_std"]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset_subset, batch_size=64):
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
    q10s, q50s, q90s, tgts = [], [], [], []
    model.eval()
    for enc, dec, tgt in loader:
        enc, dec = enc.to(DEVICE), dec.to(DEVICE)
        q10, q50, q90 = model(enc, dec)
        q10s.append(q10.cpu().numpy())
        q50s.append(q50.cpu().numpy())
        q90s.append(q90.cpu().numpy())
        tgts.append(tgt.numpy())
    return (
        np.concatenate(q10s),
        np.concatenate(q50s),
        np.concatenate(q90s),
        np.concatenate(tgts),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rescale
# ──────────────────────────────────────────────────────────────────────────────
def rescale(arr, mean, std):
    return arr * std + mean


# ──────────────────────────────────────────────────────────────────────────────
# Coverage (normalised space)
# ──────────────────────────────────────────────────────────────────────────────
def compute_coverage(q10, q90, true):
    inside = (true >= q10) & (true <= q90)
    return float(inside.mean()), inside.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Scatter subplot
# ──────────────────────────────────────────────────────────────────────────────
def _draw_scatter_ax(ax, actual, predicted, color, title):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    ax.scatter(actual, predicted, s=4, color=color, alpha=0.35, linewidths=0)

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x (perfect)")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Actual abvaerk (MWh)", fontsize=9)
    ax.set_ylabel("Predicted abvaerk (MWh)", fontsize=9)
    ax.text(0.05, 0.95, f"R² = {r2:.4f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Combined figure
# ──────────────────────────────────────────────────────────────────────────────
def build_combined_figure(results, save_path):
    n_horiz  = len(RESIDUAL_HORIZONS)
    n_models = len(results)

    fig, axes = plt.subplots(
        nrows=n_models + 1,
        ncols=n_horiz,
        figsize=(15, 5 * (n_models + 1)),
    )

    model_names = " vs ".join(r["name"] for r in results)
    fig.suptitle(
        f"Actual vs Predicted at Three Forecast Horizons (Test Set)\n{model_names}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Scatter rows ──────────────────────────────────────────────────────────
    for row, res in enumerate(results):
        for col, (h_idx, h_label) in enumerate(
            zip(RESIDUAL_HORIZONS, RESIDUAL_LABELS)
        ):
            ax = axes[row, col]
            actual    = res["true_mwh"][:, h_idx]
            predicted = res["q50_mwh"][:, h_idx]
            _draw_scatter_ax(
                ax, actual, predicted,
                color=res["scatter_color"],
                title=f"{res['name']} — {h_label}",
            )

    # ── Coverage row (full width) ─────────────────────────────────────────────
    for col in range(n_horiz):
        axes[n_models, col].set_visible(False)

    ax_cov = fig.add_subplot(
        axes[n_models, 0].get_gridspec()[n_models, :]
    )

    forecast_len = results[0]["horizon_coverage"].shape[0]
    horizons     = np.arange(1, forecast_len + 1)

    for res in results:
        cov_pct = res["horizon_coverage"] * 100
        ax_cov.plot(
            horizons, cov_pct,
            color=res["color_median"],
            linewidth=1.8,
            label=f"{res['name']}  (avg {cov_pct.mean():.1f}%)",
        )

    ax_cov.axhline(
        NOMINAL_COVERAGE * 100,
        color="black", linestyle="--", linewidth=1.2,
        label=f"Nominal {NOMINAL_COVERAGE * 100:.0f}%",
    )

    ax_cov.set_title("Prediction Interval Coverage per Forecast Horizon",
                     fontsize=11, fontweight="bold")
    ax_cov.set_xlabel("Forecast Horizon (hours)", fontsize=10)
    ax_cov.set_ylabel("Coverage (%)", fontsize=10)
    ax_cov.set_xlim(1, forecast_len)
    ax_cov.set_ylim(0, 105)
    ax_cov.grid(True, alpha=0.25)
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)
    ax_cov.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.success(f"Saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for cfg in MODEL_CONFIGS:
        log.info(f"== Loading: {cfg['name']} ==")

        m_model, _ = _load_modules(cfg["module_dir"])
        Config       = m_model.Config
        LSTMForecast = m_model.LSTMForecast

        dataset_path = os.path.join(MAIN_DIR, cfg["dataset_path"])
        test_dataset, demand_mean, demand_std = get_test_subset(dataset_path)
        log.info(f"Scaler — mean={demand_mean:.4f}  std={demand_std:.4f}  "
                 f"test samples={len(test_dataset)}")

        checkpoint_path = os.path.join(MAIN_DIR, cfg["model_save_path"])
        config = Config()
        model  = LSTMForecast(config).to(DEVICE)
        state  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

        log.info("Running inference on test set ...")
        q10_n, q50_n, q90_n, tgt_n = run_inference(model, test_dataset, config.batch_size)

        # Coverage in normalised space
        _, horizon_coverage = compute_coverage(q10_n, q90_n, tgt_n)

        # Rescale to MWh
        q50_mwh  = rescale(q50_n, demand_mean, demand_std)
        true_mwh = rescale(tgt_n, demand_mean, demand_std)

        results.append(dict(
            name             = cfg["name"],
            color_median     = cfg["color_median"],
            scatter_color    = cfg["scatter_color"],
            q50_mwh          = q50_mwh,
            true_mwh         = true_mwh,
            horizon_coverage = horizon_coverage,
        ))

        _unload_modules(cfg["module_dir"])

    build_combined_figure(
        results,
        save_path=os.path.join(OUTPUT_DIR, "fig_combined.png"),
    )

    log.success(f"All done — outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()