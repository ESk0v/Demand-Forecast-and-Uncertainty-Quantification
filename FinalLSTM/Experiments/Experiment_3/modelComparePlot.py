import argparse
import os
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, TensorDataset


MODEL_CONFIGS = [
    dict(
        name="Model 1 Decoupled - Detach",
        module_dir="Model_1_decoupled_detach",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model_decoupled.pt",
            "syn2": "model_decoupled.pt",
            "syn3": "model_decoupled.pt",
        },
        conformal_alpha=0.10,
        color="#1f77b4",
    ),
    dict(
        name="Model 2 Decoupled - No Detach",
        module_dir="Model_2_decoupled_no_detach",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model_decoupled.pt",
            "syn2": "model_decoupled.pt",
            "syn3": "model_decoupled.pt",
        },
        conformal_alpha=0.10,
        color="#ff7f0e",
    ),
    dict(
        name="Model 3 Coupled - 2 Loss",
        module_dir="Model_3_coupled_2loss",
        checkpoint_files={
            "original": "model_coupled.pt",
            "syn1": "model_coupled.pt",
            "syn2": "model_coupled.pt",
            "syn3": "model_coupled.pt",
        },
        conformal_alpha=0.10,
        color="#2ca02c",
    ),
    dict(
        name="Model 4 Coupled - 2 Sub-loss",
        module_dir="Model_4_coupled_2subloss",
        checkpoint_files={
            "original": "model_coupled.pt",
            "syn1": "model_coupled.pt",
            "syn2": "model_coupled.pt",
            "syn3": "model_coupled.pt",
        },
        conformal_alpha=0.10,
        color="#d62728",
    ),
    dict(
        name="Model 5 Coupled - 1 Pinball Loss",
        module_dir="Model_5_coupled_1loss",
        checkpoint_files={
            "original": "model_coupled.pt",
            "syn1": "model_coupled.pt",
            "syn2": "model_coupled.pt",
            "syn3": "model_coupled.pt",
        },
        conformal_alpha=0.10,
        color="#9467bd",
    ),
    dict(
        name="Model 6 Sequence - Sigmoid",
        module_dir="Model_6",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model.pt",
            "syn2": "model.pt",
            "syn3": "model.pt",
        },
        conformal_alpha=0.10,
        color="#8c564b",
    ),
    dict(
        name="Model 7 Sequence - No Sigmoid",
        module_dir="Model_7",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model.pt",
            "syn2": "model.pt",
            "syn3": "model.pt",
        },
        conformal_alpha=0.10,
        color="#e377c2",
    ),
    dict(
        name="Model 8 Sequence - Asymmetric Penalty (1.5 - 0.5)",
        module_dir="Model_8",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model.pt",
            "syn2": "model.pt",
            "syn3": "model.pt",
        },
        conformal_alpha=0.10,
        color="#7f7f7f",
    ),
    dict(
        name="Model 9 Sequence - Asymmetric Penalty (2.0 - 0.5)",
        module_dir="Model_9",
        checkpoint_files={
            "original": "model.pt",
            "syn1": "model.pt",
            "syn2": "model.pt",
            "syn3": "model.pt",
        },
        conformal_alpha=0.10,
        color="#bcbd22",
    ),
]

DATASET_SPECS = {
    "original": dict(
        file="data/dataset.pt",
        run_dir="original",
        val_ratio=0.10,
        cal_ratio=0.10,
        test_ratio=0.10,
        overlap_val_cal=True,
        title="dataset.pt",
    ),
    "syn1": dict(
        file="data/dataset_syn1.pt",
        run_dir="synthetic_data_1",
        val_ratio=1.0 / 12.0,
        cal_ratio=1.0 / 12.0,
        test_ratio=1.0 / 6.0,
        overlap_val_cal=True,
        title="dataset_syn1.pt",
    ),
    "syn2": dict(
        file="data/dataset_syn2.pt",
        run_dir="synthetic_data_2",
        val_ratio=1.0 / 12.0,
        cal_ratio=1.0 / 12.0,
        test_ratio=1.0 / 6.0,
        overlap_val_cal=True,
        title="dataset_syn2.pt",
    ),
    "syn3": dict(
        file="data/dataset_syn3.pt",
        run_dir="synthetic_data_3",
        val_ratio=1.0 / 12.0,
        cal_ratio=1.0 / 12.0,
        test_ratio=1.0 / 6.0,
        overlap_val_cal=True,
        title="dataset_syn3.pt",
    ),
}

METRIC_SPECS = [
    ("mae", "MAE", "MAE"),
    ("coverage_abs_dev_mean", "Coverage abs dev mean", "CoverageMAD"),
    ("cosine_over_target", "Cosine over target", "CosineΔ"),
]

METRIC_HATCHES = ["", "//", "xx"]


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


def _split_test_dataset(
    dataset_path: str,
    val_ratio: float,
    cal_ratio: float,
    test_ratio: float,
):
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    enc = data["encoder"]
    dec = data["decoder"]
    tgt = data["target"]

    n_total = len(tgt)
    test_size = int(n_total * test_ratio)

    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size = n_total - valcal_size - test_size
    i2 = train_size + valcal_size

    i3 = i2 + test_size
    test_ds = TensorDataset(enc[i2:i3], dec[i2:i3], tgt[i2:i3])
    return data, test_ds


def _checkpoint_path_for(cfg: dict, dataset_key: str):
    run_dir = DATASET_SPECS[dataset_key]["run_dir"]
    filename = cfg["checkpoint_files"][dataset_key]
    return _abs(os.path.join(cfg["module_dir"], run_dir, filename))


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
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # Backward compatibility: some runs saved raw state_dict directly.
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format for '{cfg['name']}': {type(checkpoint)}")

    model.load_state_dict(state_dict)
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


def _compute_metrics(cfg: dict, dataset_key: str):
    ckpt_path = _checkpoint_path_for(cfg, dataset_key)
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint for '{cfg['name']}', skipping: {ckpt_path}")
        return None

    dataset_path = _abs(DATASET_SPECS[dataset_key]["file"])
    if not os.path.exists(dataset_path):
        print(f"[WARN] Missing dataset '{dataset_key}', skipping '{cfg['name']}': {dataset_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, batch_size = _load_model(cfg, checkpoint)
    eval_batch_size = max(256, batch_size * 8)

    dataset_raw, test_ds = _split_test_dataset(
        dataset_path,
        val_ratio=DATASET_SPECS[dataset_key]["val_ratio"],
        cal_ratio=DATASET_SPECS[dataset_key]["cal_ratio"],
        test_ratio=DATASET_SPECS[dataset_key]["test_ratio"],
    )
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    q10, q50, q90, tgt = _predict_test(model, test_loader)

    demand_mean = float(dataset_raw.get("demand_mean", 0.0))
    demand_std = float(dataset_raw.get("demand_std", 1.0))

    q50_h = q50 * demand_std + demand_mean
    tgt_h = tgt * demand_std + demand_mean
    mae = float(np.mean(np.abs(q50_h - tgt_h)))

    target_coverage = float((1.0 - cfg["conformal_alpha"]) * 100.0)
    coverage_per_h = np.mean((tgt >= q10) & (tgt <= q90), axis=0) * 100.0
    coverage_abs_dev_mean = float(np.mean(np.abs(coverage_per_h - target_coverage)))

    cos_vals = checkpoint.get("train_cos_sims", [])
    if cos_vals is None or len(cos_vals) == 0:
        observed_cosine = 0.0
    else:
        observed_cosine = float(np.mean(np.asarray(cos_vals, dtype=np.float64)))
    target_cosine = 0.0
    cosine_over_target = observed_cosine - target_cosine

    return dict(
        name=cfg["name"],
        color=cfg["color"],
        mae=mae,
        coverage_abs_dev_mean=coverage_abs_dev_mean,
        cosine_over_target=cosine_over_target,
        observed_cosine=observed_cosine,
        target_cosine=target_cosine,
    )


def plot_model_comparison(results: list[dict], save_path: str, title_suffix: str):
    names = [r["name"] for r in results]
    n_models = len(names)
    n_metrics = len(METRIC_SPECS)
    x = np.arange(n_models)
    width = 0.22
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * width

    all_values = []
    for key, _, _ in METRIC_SPECS:
        all_values.extend([r[key] for r in results])

    raw_min = min(all_values) if all_values else 0.0
    raw_max = max(all_values) if all_values else 1.0
    span = max(raw_max - raw_min, 1.0)
    if raw_min >= 0:
        y_min = 0.0
    else:
        y_min = raw_min - 0.15 * span
    y_max = raw_max + 0.15 * span

    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("#fbfbfd")
    ax.set_facecolor("#fbfbfd")

    label_base_offset = 0.015 * (y_max - y_min)
    label_stagger = 0.025 * (y_max - y_min)

    for i, (key, _, short_label) in enumerate(METRIC_SPECS):
        values = [r[key] for r in results]
        bar_colors = [r["color"] for r in results]
        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=bar_colors,
            edgecolor="#2f2f2f",
            linewidth=0.6,
            alpha=0.95,
            hatch=METRIC_HATCHES[i],
        )

        # Solution 1: staggered external labels (always visible, including zeros).
        for bar in bars:
            v = bar.get_height()
            tx = bar.get_x() + bar.get_width() / 2.0
            if v >= 0:
                ty = v + label_base_offset + i * label_stagger
                va = "bottom"
            else:
                ty = v - label_base_offset - i * label_stagger
                va = "top"
            ax.text(
                tx,
                ty,
                f"{v:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
                color="#2b2b2b",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    alpha=0.88,
                    edgecolor="none",
                ),
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f6f6f")
    ax.spines["bottom"].set_color("#6f6f6f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Experiment 3 Model Comparison - {title_suffix}")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Model legend inside plot (top-left), as requested.
    model_handles = [Patch(color=r["color"], label=r["name"]) for r in results]
    legend_models = ax.legend(
        handles=model_handles,
        title="Models",
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    legend_models.get_frame().set_alpha(0.92)
    ax.add_artist(legend_models)

    metric_handles = [
        Patch(facecolor="white", edgecolor="#2f2f2f", hatch=METRIC_HATCHES[i], label=label)
        for i, (_, label, _) in enumerate(METRIC_SPECS)
    ]
    legend_metrics = ax.legend(
        handles=metric_handles,
        title="Metrics",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    legend_metrics.get_frame().set_alpha(0.92)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved plot: {save_path}")


def _dataset_keys_from_arg(dataset_arg: str):
    if dataset_arg == "all":
        return ["original", "syn1", "syn2", "syn3"]
    return [dataset_arg]


def main():
    parser = argparse.ArgumentParser(description="Compare Experiment 3 models across datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "original", "syn1", "syn2", "syn3"],
        default="all",
        help="Generate plot(s) for one dataset or all datasets.",
    )
    args = parser.parse_args()

    for dataset_key in _dataset_keys_from_arg(args.dataset):
        dataset_title = DATASET_SPECS[dataset_key]["title"]
        print(f"[INFO] Running comparison for {dataset_title}")

        results = []
        for cfg in MODEL_CONFIGS:
            print(f"[INFO] Evaluating: {cfg['name']}")
            metrics = _compute_metrics(cfg, dataset_key)
            if metrics is not None:
                results.append(metrics)

        if not results:
            print(f"[ERROR] No valid models found to compare for {dataset_title}.")
            continue

        save_name = f"compare_plots/modelComparePlot_{dataset_key}.png"
        save_path = _abs(save_name)
        plot_model_comparison(results, save_path, dataset_title)

        print("[INFO] Included models:")
        for r in results:
            print(
                f"  - {r['name']}: "
                f"MAE={r['mae']:.4f}, "
                f"CoverageMAD={r['coverage_abs_dev_mean']:.4f}, "
                f"CosineΔ={r['cosine_over_target']:.4f} "
                f"(obs={r['observed_cosine']:.4f}, target={r['target_cosine']:.4f})"
            )


if __name__ == "__main__":
    main()
