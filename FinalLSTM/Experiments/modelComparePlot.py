import os
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from torch.utils.data import DataLoader, TensorDataset


MODEL_CONFIGS = [
    dict(
        name="Model 1 Decoupled Detach",
        module_dir="Experiment_3/Model_1_decoupled_detach",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_1_decoupled_detach/synthetic_data_1/model_decoupled.pt",
        conformal_alpha=0.10,
        base_color="#1f77b4",
    ),
    dict(
        name="Model 2 Decoupled No Detach",
        module_dir="Experiment_3/Model_2_decoupled_no_detach",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_2_decoupled_no_detach/synthetic_data_1/model_decoupled.pt",
        conformal_alpha=0.10,
        base_color="#1f77b4",
    ),
    dict(
        name="Model 3 Coupled 2 Loss",
        module_dir="Experiment_3/Model_3_coupled_2loss",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_3_coupled_2loss/synthetic_data_1/model_coupled.pt",
        conformal_alpha=0.10,
        base_color="#2ca02c",
    ),
    dict(
        name="Model 4 Coupled 2 Sub-loss",
        module_dir="Experiment_3/Model_4_coupled_2subloss",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_4_coupled_2subloss/synthetic_data_1/model_coupled.pt",
        conformal_alpha=0.10,
        base_color="#ff7f0e",
    ),
    dict(
        name="Model 6 Test",
        module_dir="Experiment_3/Model_6",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_6/checkpoints/model.pt",
        conformal_alpha=0.10,
        base_color="#ff0eb7",
    ),
    dict(
        name="Model 7 Test 2",
        module_dir="Experiment_3/Model_7",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_7/synthetic_data_1/model.pt",
        conformal_alpha=0.10,
        base_color="#ff0eb7",
    ),
    dict(
        name="Model 8 Test 3",
        module_dir="Experiment_3/Model_8",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_8/synthetic_data_1/model.pt",
        conformal_alpha=0.10,
        base_color="#ff0eb7",
    ),
    dict(
        name="Model 9 Test 3",
        module_dir="Experiment_3/Model_9",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_9/synthetic_data_1/model.pt",
        conformal_alpha=0.10,
        base_color="#ff0eb7",
    ),
]


VAL_RATIO = 1.0 / 12.0
CAL_RATIO = 1.0 / 12.0
TEST_RATIO = 1.0 / 6.0

METRIC_SPECS = [
    ("mae", "MAE", "MAE"),
    ("coverage_over_target", "Coverage over target", "CoverageΔ"),
    ("cosine_over_target", "Cosine over target", "CosineΔ"),
]


def _abs(path_from_this_file: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path_from_this_file))


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

    n_total = len(tgt)
    test_size = int(n_total * TEST_RATIO)
    valcal_size = int(n_total * (VAL_RATIO + CAL_RATIO))
    train_size = n_total - valcal_size - test_size

    i2 = train_size + valcal_size
    i3 = i2 + test_size

    test_ds = TensorDataset(enc[i2:i3], dec[i2:i3], tgt[i2:i3])
    return data, test_ds


def _load_model(cfg: dict, checkpoint: dict):
    module_path = _abs(os.path.join(cfg["module_dir"], "LSTMModel.py"))
    module = _load_module(f"lstm_model_{cfg['name'].replace(' ', '_')}", module_path)

    config = module.Config()
    ckpt_config = checkpoint.get("config")
    if isinstance(ckpt_config, dict):
        for key, value in ckpt_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.device = "cpu"

    model = module.LSTMForecast(config).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(getattr(config, "batch_size", 64))


def _predict_test(model, test_loader):
    q10_all, q50_all, q90_all, tgt_all = [], [], [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            q10, q50, q90 = model(enc, dec)
            q10_all.append(q10.cpu().numpy())
            q50_all.append(q50.cpu().numpy())
            q90_all.append(q90.cpu().numpy())
            tgt_all.append(tgt.cpu().numpy())
    return (
        np.concatenate(q10_all),
        np.concatenate(q50_all),
        np.concatenate(q90_all),
        np.concatenate(tgt_all),
    )


def _compute_metrics(cfg: dict):
    ckpt_path = _abs(cfg["model_save_path"])
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint for '{cfg['name']}', skipping: {ckpt_path}")
        return None

    dataset_path = _abs(cfg["dataset_path"])
    if not os.path.exists(dataset_path):
        print(f"[WARN] Missing dataset for '{cfg['name']}', skipping: {dataset_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, batch_size = _load_model(cfg, checkpoint)
    eval_batch_size = max(256, batch_size * 8)

    dataset_raw, test_ds = _split_test_dataset(dataset_path)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    q10, q50, q90, tgt = _predict_test(model, test_loader)

    demand_mean = float(dataset_raw.get("demand_mean", 0.0))
    demand_std = float(dataset_raw.get("demand_std", 1.0))

    q50_h = q50 * demand_std + demand_mean
    tgt_h = tgt * demand_std + demand_mean
    mae = float(np.mean(np.abs(q50_h - tgt_h)))

    observed_coverage = float(np.mean((tgt >= q10) & (tgt <= q90)) * 100.0)
    target_coverage = float((1.0 - cfg["conformal_alpha"]) * 100.0)
    coverage_over_target = observed_coverage - target_coverage

    cos_vals = checkpoint.get("train_cos_sims", [])
    if cos_vals is None or len(cos_vals) == 0:
        observed_cosine = 0.0
    else:
        cos_vals = np.asarray(cos_vals, dtype=np.float64)
        observed_cosine = float(np.mean(cos_vals))
    target_cosine = 0.0
    cosine_over_target = observed_cosine - target_cosine

    return dict(
        name=cfg["name"],
        mae=mae,
        coverage_over_target=coverage_over_target,
        cosine_over_target=cosine_over_target,
        observed_cosine=observed_cosine,
        target_cosine=target_cosine,
    )


def _shade_color(base_hex: str, shade_idx: int, n_shades: int) -> tuple[float, float, float]:
    """
    Create a lighter/darker nuance from a base color.
    shade_idx: 0..n_shades-1
    """
    rgb = np.array(to_rgb(base_hex))
    # 0 -> closest to base, last -> lightest nuance
    if n_shades <= 1:
        blend = 0.0
    else:
        blend = 0.12 + (0.45 * (shade_idx / (n_shades - 1)))
    shaded = rgb * (1.0 - blend) + np.ones(3) * blend
    return tuple(shaded.tolist())


def plot_model_comparison(results: list[dict], save_path: str):
    names = [r["name"] for r in results]
    n_models = len(names)
    n_metrics = len(METRIC_SPECS)
    x = np.arange(n_models)
    width = 0.18
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * width

    all_values = []
    for key, _, _ in METRIC_SPECS:
        all_values.extend([r[key] for r in results])

    y_min = min(min(all_values) - 1.0, 0.0)
    y_max = max(all_values) + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("#fbfbfd")
    ax.set_facecolor("#fbfbfd")

    # Build quick lookup for model base colors from configs
    base_color_by_name = {cfg["name"]: cfg["base_color"] for cfg in MODEL_CONFIGS}
    label_offset = 0.02 * (y_max - y_min)

    for i, (key, _, short_label) in enumerate(METRIC_SPECS):
        values = [r[key] for r in results]
        bar_colors = [
            _shade_color(base_color_by_name[r["name"]], i, n_metrics)
            for r in results
        ]
        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=bar_colors,
            edgecolor="#2f2f2f",
            linewidth=0.6,
            alpha=0.95,
        )

        # Metric name + actual value on every column
        for bar in bars:
            v = bar.get_height()
            tx = bar.get_x() + bar.get_width() / 2.0
            if abs(v) > 1e-12:
                ty = v / 2.0
                va = "center"
            else:
                # Zero-height bars have no interior area; place label just above baseline.
                ty = 0.0 + label_offset
                va = "bottom"
            ax.text(
                tx,
                ty,
                f"{short_label}\n{v:.2f}",
                ha="center",
                va=va,
                fontsize=8.5,
                fontweight="bold",
                color="#2b2b2b",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.82, edgecolor="none"),
            )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f6f6f")
    ax.spines["bottom"].set_color("#6f6f6f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Experiment 3 Model Comparison")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend: models and their base colors
    legend_handles = [
        Patch(color=base_color_by_name[name], label=name)
        for name in names
    ]
    legend_models = ax.legend(
        handles=legend_handles,
        title="Models (base colors)",
        loc="upper left",
        frameon=True,
    )
    legend_models.get_frame().set_alpha(0.92)
    ax.add_artist(legend_models)

    # Secondary legend: what each column label means
    metric_text = ",  ".join([f"{short}={label}" for _, label, short in METRIC_SPECS])
    ax.text(
        0.99,
        0.99,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#3b3b3b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#d0d0d0"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved plot: {save_path}")


def main():
    results = []
    for cfg in MODEL_CONFIGS:
        print(f"[INFO] Evaluating: {cfg['name']}")
        metrics = _compute_metrics(cfg)
        if metrics is not None:
            results.append(metrics)

    if not results:
        print("[ERROR] No valid models found to compare.")
        return

    save_path = _abs("modelComparePlot.png")
    plot_model_comparison(results, save_path)

    print("[INFO] Included models:")
    for r in results:
        print(
            f"  - {r['name']}: "
            f"MAE={r['mae']:.4f}, "
            f"CoverageΔ={r['coverage_over_target']:.4f}, "
            f"CosineΔ={r['cosine_over_target']:.4f} "
            f"(obs={r['observed_cosine']:.4f}, target={r['target_cosine']:.4f})"
        )


if __name__ == "__main__":
    main()
