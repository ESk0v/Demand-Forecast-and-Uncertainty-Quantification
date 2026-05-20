import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset, TensorDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_SPECS = [
    {"module_dir": "Model_1_decoupled_detach", "model_name": "Model_1 Decoupled - Detach"},
    {"module_dir": "Model_2_decoupled_no_detach", "model_name": "Model_2 Decoupled - No Detach"},
    {"module_dir": "Model_3_coupled_2loss", "model_name": "Model_3 Coupled - 2 loss functions"},
    {"module_dir": "Model_4_coupled_2subloss", "model_name": "Model_4 Coupled - 2 sub-loss function"},
    {"module_dir": "Model_5_coupled_1loss", "model_name": "Model_5 Coupled - 1 pinball loss"},
    # {"module_dir": "Model_6", "model_name": "Model_6 Sequence - Sigmoid"},
    {"module_dir": "Model_7", "model_name": "Model_7 Sequence - No Sigmoid"},
    {"module_dir": "Model_8", "model_name": "Model_8 Sequence - Asymmetric Penalty (1.5 - 0.5)"},
    {"module_dir": "Model_9", "model_name": "Model_9 Sequence - Asymmetric Penalty (2.0 - 0.5)"},
]

CHECKPOINT_CANDIDATES = (
    "model_decoupled.pt",
    "model_coupled.pt",
    "model.pt",
)


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_checkpoint_path(module_dir: str, run_subdir: str, explicit_checkpoint: str | None) -> str:
    if explicit_checkpoint:
        cp = _resolve_path(explicit_checkpoint)
        if not os.path.isfile(cp):
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        return cp

    run_dir = os.path.join(BASE_DIR, module_dir, run_subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    for name in CHECKPOINT_CANDIDATES:
        candidate = os.path.join(run_dir, name)
        if os.path.isfile(candidate):
            return candidate

    pt_files = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    if len(pt_files) == 1:
        return os.path.join(run_dir, pt_files[0])

    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found in: {run_dir}")

    raise RuntimeError(
        f"Multiple checkpoint files found in {run_dir}: {pt_files}. "
        "Pass --checkpoint explicitly."
    )


def load_and_split_dataset(dataset_path, val_ratio=1 / 12, cal_ratio=1 / 12, test_ratio=1 / 6):
    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset["encoder"]
    decoder_data = dataset["decoder"]
    target_data = dataset["target"]
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size = n_total - valcal_size - test_size

    i0 = 0
    i1 = train_size
    i2 = i1 + valcal_size
    i3 = i2 + test_size

    val_size = valcal_size
    cal_size = valcal_size

    train_dataset = Subset(full_dataset, range(i0, i1))
    val_dataset = Subset(full_dataset, range(i1, i2))
    cal_dataset = Subset(full_dataset, range(i1, i2))
    test_dataset = Subset(full_dataset, range(i2, i3))

    return (
        train_dataset,
        val_dataset,
        cal_dataset,
        test_dataset,
        train_size,
        val_size,
        cal_size,
        test_size,
    )


def collect_predictions(model, loader, device):
    model.eval()
    all_q10, all_q50, all_q90, all_tgt = [], [], [], []

    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec = enc.to(device), dec.to(device)
            q10, q50, q90 = model(enc, dec)

            all_q10.append(_to_numpy(q10))
            all_q50.append(_to_numpy(q50))
            all_q90.append(_to_numpy(q90))
            all_tgt.append(_to_numpy(tgt))

    return (
        np.concatenate(all_q10),
        np.concatenate(all_q50),
        np.concatenate(all_q90),
        np.concatenate(all_tgt),
    )


def conformal_calibration(q10, q50, q90, targets, alpha=0.1):
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    scores = np.maximum(np.maximum(q10 - targets, targets - q90), 0.0)
    n_horizons = scores.shape[1]
    u_alpha_t = np.array([
        np.quantile(scores[:, t], 1.0 - alpha)
        for t in range(n_horizons)
    ])

    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    empirical_coverage = float(np.mean((targets >= q10_cal) & (targets <= q90_cal)))
    print(
        f"Empirical coverage on cal set: {empirical_coverage:.3f} "
        f"(target ~= {1 - alpha:.2f})"
    )

    return q10_cal, q90_cal, u_alpha_t


def apply_conformal(q10, q90, u_alpha):
    q10 = _to_numpy(q10)
    q90 = _to_numpy(q90)
    u_alpha = _to_numpy(u_alpha)

    if np.isscalar(u_alpha) or np.ndim(u_alpha) == 0:
        q10_cal = q10 - float(u_alpha)
        q90_cal = q90 + float(u_alpha)
    else:
        q10_cal = q10 - u_alpha[np.newaxis, :]
        q90_cal = q90 + u_alpha[np.newaxis, :]

    return q10_cal, q90_cal


def plot_train_val_loss(train_losses, val_losses, best_epoch, save_path, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.5, marker="o", markersize=2)
    ax.plot(epochs_range, val_losses, label="Validation Loss", linewidth=1.5, marker="o", markersize=2)
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} - Train vs Validation Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if len(all_losses) > 0 and min(all_losses) > 0 and max(all_losses) / min(all_losses) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("Loss (log scale)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cosine_and_coverage(
    cosine_sims_epoch,
    q10,
    q50,
    q90,
    targets,
    u_alpha_t,
    save_path,
    model_name,
    alpha=0.1,
):
    q10_cal = q10 - u_alpha_t[np.newaxis, :]
    q90_cal = q90 + u_alpha_t[np.newaxis, :]

    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0) * 100
    cal_coverage = np.mean((targets >= q10_cal) & (targets <= q90_cal), axis=0) * 100
    nominal = (1 - alpha) * 100

    mae_q50 = float(np.mean(np.abs(q50 - targets)))
    general_coverage = float(np.mean((targets >= q10) & (targets <= q90)) * 100)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    ax_cos, ax_cov = axes

    epochs = np.arange(1, len(cosine_sims_epoch) + 1)
    ax_cos.plot(
        epochs,
        cosine_sims_epoch,
        color="purple",
        marker="o",
        linewidth=1.4,
        markersize=3,
        label="Avg cosine similarity per epoch",
    )
    if len(cosine_sims_epoch) > 0:
        y_min = min(cosine_sims_epoch) - 0.1
        y_max = max(cosine_sims_epoch) + 0.1
        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1
        ax_cos.set_ylim(max(-1.0, y_min), min(1.0, y_max))
    ax_cos.set_xlabel("Epoch")
    ax_cos.set_ylabel("Cosine Similarity")
    ax_cos.set_title(f"{model_name} - Gradient Interference (Encoder)")
    ax_cos.grid(True, alpha=0.3)
    ax_cos.legend(loc="best")

    horizons = np.arange(1, len(raw_coverage) + 1)
    line_raw, = ax_cov.plot(horizons, raw_coverage, color="steelblue", label="Raw interval coverage")
    line_cal, = ax_cov.plot(horizons, cal_coverage, color="green", label="Calibrated interval coverage")
    line_nom = ax_cov.axhline(
        y=nominal,
        color="orange",
        linestyle="--",
        label=f"Nominal {nominal:.0f}% target",
    )
    ax_cov.set_xlabel("Forecast Horizon (hours)")
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_ylim(50, 100)
    ax_cov.set_title(f"{model_name} - Coverage Per Horizon")
    ax_cov.grid(True, alpha=0.3)

    line_legend = ax_cov.legend(handles=[line_raw, line_cal, line_nom], loc="lower left", fontsize=9)
    ax_cov.add_artist(line_legend)

    metric_handles = [
        Line2D([], [], color="none", label=f"MAE (q50 vs target): {mae_q50:.4f}"),
        Line2D([], [], color="none", label=f"General coverage: {general_coverage:.2f}%"),
    ]
    ax_cov.legend(
        handles=metric_handles,
        loc="lower right",
        frameon=True,
        title="Metrics",
        handlelength=0,
        handletextpad=0,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(
    q50_h,
    targets_h,
    encoder_data,
    train_size,
    val_size,
    demand_mean,
    demand_std,
    save_path,
    model_name,
):
    encoder_np = _to_numpy(encoder_data)
    test_start = train_size + val_size
    test_encoder = encoder_np[test_start:test_start + len(q50_h)]
    last_known = test_encoder[:, -1, 0] * demand_std + demand_mean
    persist_pred = np.tile(last_known[:, None], (1, q50_h.shape[1]))

    def _r2(actual, pred):
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10)

    horizons = [(0, "1h"), (23, "24h"), (167, "168h")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (h_idx, h_label) in enumerate(horizons):
        actual = targets_h[:, h_idx]
        pred = q50_h[:, h_idx]
        min_v = min(actual.min(), pred.min())
        max_v = max(actual.max(), pred.max())

        ax = axes[0, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color="steelblue")
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.0)
        ax.set_title(f"{model_name} - LSTM ({h_label})\nR2={_r2(actual, pred):.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    for col, (h_idx, h_label) in enumerate(horizons):
        actual = targets_h[:, h_idx]
        pred = persist_pred[:, h_idx]
        min_v = min(actual.min(), pred.min())
        max_v = max(actual.max(), pred.max())

        ax = axes[1, col]
        ax.scatter(actual, pred, s=4, alpha=0.25, color="coral")
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.0)
        ax.set_title(f"{model_name} - Persistence ({h_label})\nR2={_r2(actual, pred):.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} - Actual vs Predicted (With Persistence Baseline)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_test_predictions(q10, q50, q90, targets, save_path, model_name):
    n_samples = q50.shape[0]
    n_panels = min(3, n_samples)
    sample_indices = np.linspace(0, n_samples - 1, n_panels, dtype=int)
    hours = np.arange(1, q50.shape[1] + 1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), squeeze=False)
    axes = axes[:, 0]

    for ax, idx in zip(axes, sample_indices):
        mae = float(np.mean(np.abs(targets[idx] - q50[idx])))
        coverage = float(np.mean((targets[idx] >= q10[idx]) & (targets[idx] <= q90[idx])) * 100.0)
        ax.plot(hours, targets[idx], label="Actual", color="blue", linewidth=1.5)
        ax.plot(hours, q50[idx], label="Median (q50)", color="red", linewidth=1.5)
        ax.fill_between(hours, q10[idx], q90[idx], color="red", alpha=0.18, label="Uncertainty bound")
        ax.set_title(f"{model_name} - Test Window {idx} (MAE={mae:.4f}, Coverage={coverage:.1f}%)")
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("abvaerk")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"{model_name} - Test Set Predictions (3 Example Windows)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _load_model_and_checkpoint(module_dir: str, checkpoint_path: str, device: str):
    model_file = os.path.join(BASE_DIR, module_dir, "LSTMModel.py")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model definition not found: {model_file}")

    module_name = f"_exp3_{module_dir}_model_{os.getpid()}"
    lstm_model_mod = _load_module_from_file(module_name, model_file)
    Config = lstm_model_mod.Config
    LSTMForecast = lstm_model_mod.LSTMForecast

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = Config()
    cfg_from_checkpoint = checkpoint.get("config", {})
    if isinstance(cfg_from_checkpoint, dict):
        for key, value in cfg_from_checkpoint.items():
            setattr(config, key, value)

    config.device = device
    if "training_variant" in checkpoint:
        setattr(config, "training_variant", checkpoint["training_variant"])

    model = LSTMForecast(config).to(device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' missing in checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if hasattr(model, "training_variant") and "training_variant" in checkpoint:
        model.training_variant = checkpoint["training_variant"]

    return model, config, checkpoint


def generate_plots_for_checkpoint(
    module_dir: str,
    checkpoint_path: str,
    dataset_path: str,
    model_name: str,
    conformal_alpha_override: float | None = None,
    output_dir: str | None = None,
    eval_batch_size: int = 512,
):
    dataset_path = _resolve_path(dataset_path)
    checkpoint_path = _resolve_path(checkpoint_path)
    if output_dir is not None:
        output_dir = _resolve_path(output_dir)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, checkpoint = _load_model_and_checkpoint(module_dir, checkpoint_path, device)

    (
        _train_dataset,
        _val_dataset,
        cal_dataset,
        test_dataset,
        train_size,
        val_size,
        cal_size,
        _test_size,
    ) = load_and_split_dataset(dataset_path)

    cal_loader = DataLoader(
        cal_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=False,
    )

    alpha = conformal_alpha_override
    if alpha is None:
        alpha = float(checkpoint.get("conformal_alpha", 0.1))

    q10_raw, q50_raw, q90_raw, tgt_cal = collect_predictions(model, cal_loader, device)

    if "conformal_u_alpha" in checkpoint:
        u_alpha_t = _to_numpy(checkpoint["conformal_u_alpha"])
    else:
        print(
            f"[WARN] conformal_u_alpha missing in checkpoint for {module_dir}. "
            f"Calibrating now with alpha={alpha:.3f}."
        )
        _, _, u_alpha_t = conformal_calibration(q10_raw, q50_raw, q90_raw, tgt_cal, alpha=alpha)

    q10_test_raw, q50_test_raw, q90_test_raw, tgt_test_raw = collect_predictions(
        model, test_loader, device
    )
    q10_test_cal, q90_test_cal = apply_conformal(q10_test_raw, q90_test_raw, u_alpha_t)

    dataset_blob = torch.load(dataset_path, weights_only=False)
    demand_mean = float(dataset_blob.get("demand_mean", 0.0))
    demand_std = float(dataset_blob.get("demand_std", 1.0))
    encoder_data = dataset_blob["encoder"]

    def denorm(arr):
        return arr * demand_std + demand_mean

    q10_cal_h = denorm(q10_test_cal)
    q50_h = denorm(q50_test_raw)
    q90_cal_h = denorm(q90_test_cal)
    tgt_h = denorm(tgt_test_raw)

    train_losses = list(checkpoint.get("train_losses", []))
    val_losses = list(checkpoint.get("val_losses", []))
    best_epoch = int(checkpoint.get("epoch", 1))

    train_cos_sims_epoch = checkpoint.get("train_cos_sims")
    if train_cos_sims_epoch is None:
        train_cos_sims_epoch = [0.0] * len(train_losses)
    else:
        train_cos_sims_epoch = list(_to_numpy(train_cos_sims_epoch))
        if len(train_cos_sims_epoch) == 0 and len(train_losses) > 0:
            train_cos_sims_epoch = [0.0] * len(train_losses)

    run_dir = os.path.dirname(checkpoint_path)
    plot_dir = output_dir if output_dir else os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    loss_plot_path = os.path.join(plot_dir, "train_val_loss.png")
    cos_cov_plot_path = os.path.join(plot_dir, "cosine_similarity_and_coverage.png")
    scatter_plot_path = os.path.join(plot_dir, "actual_vs_predicted.png")
    test_preds_plot_path = os.path.join(plot_dir, "test_predictions.png")

    if train_losses and val_losses:
        plot_train_val_loss(
            train_losses,
            val_losses,
            best_epoch,
            loss_plot_path,
            model_name=model_name,
        )
    else:
        print(
            f"[WARN] train/val losses missing for {module_dir}; "
            "skipping train_val_loss.png"
        )

    plot_cosine_and_coverage(
        train_cos_sims_epoch,
        q10_raw,
        q50_raw,
        q90_raw,
        tgt_cal,
        _to_numpy(u_alpha_t),
        cos_cov_plot_path,
        model_name=model_name,
        alpha=alpha,
    )
    plot_actual_vs_predicted(
        q50_h,
        tgt_h,
        encoder_data,
        train_size,
        val_size,
        demand_mean,
        demand_std,
        scatter_plot_path,
        model_name=model_name,
    )
    plot_test_predictions(
        q10_cal_h,
        q50_h,
        q90_cal_h,
        tgt_h,
        test_preds_plot_path,
        model_name=model_name,
    )

    print(f"[SUCCESS] {model_name}")
    if train_losses and val_losses:
        print(f"  - {loss_plot_path}")
    print(f"  - {cos_cov_plot_path}")
    print(f"  - {scatter_plot_path}")
    print(f"  - {test_preds_plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the 4 Experiment_3 plots for trained Model_1..Model_5 synthetic_data_1."
    )
    parser.add_argument(
        "--dataset",
        default="data/dataset.pt",
        help="Dataset path (absolute or relative to Experiment_3).",
    )
    parser.add_argument(
        "--run-subdir",
        default="original",
        help="Run subfolder inside each model dir used to locate synthetic_data_1.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate plots for Models 1..5 using --run-subdir auto-discovery.",
    )
    parser.add_argument(
        "--model-dir",
        help="Single model directory (e.g. Model_2_decoupled_no_detach).",
    )
    parser.add_argument(
        "--model-name",
        help="Display name used in plot titles for single-model mode.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Explicit checkpoint path (.pt). If omitted, auto-discovered in model_dir/run_subdir.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional custom output directory for plots in single-model mode.",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=None,
        help="Optional override for conformal alpha.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="Batch size for inference during plot generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_models:
        for spec in DEFAULT_MODEL_SPECS:
            checkpoint_path = _find_checkpoint_path(
                module_dir=spec["module_dir"],
                run_subdir=args.run_subdir,
                explicit_checkpoint=None,
            )
            generate_plots_for_checkpoint(
                module_dir=spec["module_dir"],
                checkpoint_path=checkpoint_path,
                dataset_path=args.dataset,
                model_name=spec["model_name"],
                conformal_alpha_override=args.conformal_alpha,
                output_dir=None,
                eval_batch_size=args.eval_batch_size,
            )
        return

    if not args.model_dir:
        raise ValueError("Provide --model-dir for single-model mode, or use --all-models.")

    checkpoint_path = _find_checkpoint_path(
        module_dir=args.model_dir,
        run_subdir=args.run_subdir,
        explicit_checkpoint=args.checkpoint,
    )
    model_name = args.model_name if args.model_name else args.model_dir
    generate_plots_for_checkpoint(
        module_dir=args.model_dir,
        checkpoint_path=checkpoint_path,
        dataset_path=args.dataset,
        model_name=model_name,
        conformal_alpha_override=args.conformal_alpha,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
