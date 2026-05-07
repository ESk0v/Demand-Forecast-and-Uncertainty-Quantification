import os
import sys
import importlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    dict(
        name="Baseline (UncertaintyDecoder)",
        module_dir="Model_1",
        dataset_path="Model_1/data/dataset.pt",
        model_save_path="Model_1/checkpoints/model.pt",
        color_median="#2563EB",   # blue
        color_band="#93C5FD",
        color_actual="#1E293B",
    ),
    dict(
        name="Ablated (Direct Output Head)",
        module_dir="Model_2",
        dataset_path="Model_2/data/dataset.pt",
        model_save_path="Model_2/checkpoints/model.pt",
        color_median="#DC2626",   # red
        color_band="#FCA5A5",
        color_actual="#1E293B",
    ),
]


MAIN_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(MAIN_DIR, "plots")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCHES_FOR_INFERENCE = 4
NOMINAL_COVERAGE = 0.90


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
class _L:
    def info(self, m): print(f"[INFO]    {m}")
    def success(self, m): print(f"[SUCCESS] {m}")
    def warning(self, m): print(f"[WARN]    {m}")
    def error(self, m): print(f"[ERROR]   {m}")


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

    m_model = importlib.import_module("LSTMModel")
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
# Inference
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, max_batches=None):

    model.eval()

    q10_list = []
    q50_list = []
    q90_list = []
    true_list = []

    for batch_idx, batch in enumerate(loader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        enc_in, dec_in, target = batch

        enc_in = enc_in.to(DEVICE)
        dec_in = dec_in.to(DEVICE)
        target = target.to(DEVICE)

        q10, q50, q90 = model(enc_in, dec_in)

        q10_list.append(q10.cpu().numpy())
        q50_list.append(q50.cpu().numpy())
        q90_list.append(q90.cpu().numpy())
        true_list.append(target.cpu().numpy())

    return (
        np.concatenate(q10_list, axis=0),
        np.concatenate(q50_list, axis=0),
        np.concatenate(q90_list, axis=0),
        np.concatenate(true_list, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────────────────────────
def compute_coverage(q10, q90, true):

    inside = (true >= q10) & (true <= q90)

    overall_coverage = float(inside.mean())

    # Coverage per horizon step
    horizon_coverage = inside.mean(axis=0)

    return overall_coverage, horizon_coverage


# ──────────────────────────────────────────────────────────────────────────────
# Common styling
# ──────────────────────────────────────────────────────────────────────────────
def _style_axes(ax, title, xlabel, ylabel):

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=9)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Coverage vs Horizon
# ──────────────────────────────────────────────────────────────────────────────
def plot_coverage(coverage_results, save_path):

    forecast_len = len(coverage_results[0]["horizon_coverage"])
    horizons = np.arange(1, forecast_len + 1)

    fig, ax = plt.subplots(figsize=(14, 5))

    for result in coverage_results:

        coverage_pct = result["horizon_coverage"] * 100

        ax.plot(
            horizons,
            coverage_pct,
            color=result["color_median"],
            linewidth=1.8,
            label=result["name"],
        )

        print(
            f"{result['name']}: "
            f"{coverage_pct.mean():.2f}% average coverage"
        )

    # Nominal line
    ax.axhline(
        NOMINAL_COVERAGE * 100,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Nominal {NOMINAL_COVERAGE*100:.0f}%"
    )

    _style_axes(
        ax,
        title="Prediction Interval Coverage per Horizon",
        xlabel="Forecast Horizon (hours)",
        ylabel="Coverage (%)"
    )

    ax.set_xlim(1, forecast_len)
    ax.set_ylim(0, 105)

    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    log.success(f"Coverage plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2/3 — Inference plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_inference_week(
    q10,
    q50,
    q90,
    true,
    model_name,
    color_median,
    color_band,
    color_actual,
    save_path,
):

    sample_idx = 0

    forecast_len = q50.shape[1]
    steps = np.arange(forecast_len)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Interval
    ax.fill_between(
        steps,
        q10[sample_idx],
        q90[sample_idx],
        color=color_band,
        alpha=0.55,
        label="q10–q90 interval",
    )

    # Median
    ax.plot(
        steps,
        q50[sample_idx],
        color=color_median,
        linewidth=1.8,
        label="q50 forecast",
    )

    # Actual
    ax.plot(
        steps,
        true[sample_idx],
        color=color_actual,
        linewidth=1.2,
        linestyle="--",
        label="Actual",
    )

    # Single sample coverage
    sample_cov, _ = compute_coverage(
        q10[sample_idx][None, :],
        q90[sample_idx][None, :],
        true[sample_idx][None, :]
    )

    ax.text(
        0.98,
        0.96,
        f"Sample coverage: {sample_cov*100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
        ),
    )

    _style_axes(
        ax,
        title=f"Inference — {model_name}",
        xlabel="Forecast Horizon (hours)",
        ylabel="Target value"
    )

    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    log.success(f"Inference plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coverage_results = []
    inference_data = []

    for cfg in MODEL_CONFIGS:

        log.info(f"══ Loading: {cfg['name']} ══")

        m_model, m_training = _load_modules(cfg["module_dir"])

        Config = m_model.Config
        LSTMForecast = m_model.LSTMForecast
        load_and_split_dataset = m_training.load_and_split_dataset

        dataset_path = os.path.join(MAIN_DIR, cfg["dataset_path"])

        (_, _, _, test_dataset,
         _, _, _, _) = load_and_split_dataset(dataset_path)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        checkpoint_path = os.path.join(
            MAIN_DIR,
            cfg["model_save_path"]
        )

        config = Config()

        model = LSTMForecast(config).to(DEVICE)

        state = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)

        log.info("Checkpoint loaded. Running inference...")

        # Full inference for Figure 1
        q10_all, q50_all, q90_all, true_all = run_inference(
            model,
            test_loader
        )

        coverage, horizon_coverage = compute_coverage(
            q10_all,
            q90_all,
            true_all
        )

        log.info(
            f"Empirical coverage: {coverage*100:.2f}%"
        )

        coverage_results.append({
            "name": cfg["name"],
            "horizon_coverage": horizon_coverage,
            "color_median": cfg["color_median"],
        })

        # Partial inference for Figures 2/3
        q10_inf, q50_inf, q90_inf, true_inf = run_inference(
            model,
            test_loader,
            max_batches=MAX_BATCHES_FOR_INFERENCE
        )

        inference_data.append(
            (q10_inf, q50_inf, q90_inf, true_inf, cfg)
        )

        _unload_modules(cfg["module_dir"])

    # Figure 1
    plot_coverage(
        coverage_results,
        os.path.join(
            OUTPUT_DIR,
            "fig1_coverage_comparison.png"
        )
    )

    # Figure 2/3
    for idx, (q10, q50, q90, true, cfg) in enumerate(
        inference_data,
        start=2
    ):

        save_path = os.path.join(
            OUTPUT_DIR,
            f"fig{idx}.png"
        )

        plot_inference_week(
            q10=q10,
            q50=q50,
            q90=q90,
            true=true,
            model_name=cfg["name"],
            color_median=cfg["color_median"],
            color_band=cfg["color_band"],
            color_actual=cfg["color_actual"],
            save_path=save_path,
        )

    log.success(
        f"All plots written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()