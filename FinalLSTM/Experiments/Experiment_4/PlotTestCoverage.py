import os
import sys
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

MODEL_CONFIGS = [
    {"name": "Model_1", "dir": "Model_1", "ckpt": "Model_1/original/model.pt"},
    {"name": "Model_2", "dir": "Model_2", "ckpt": "Model_2/original/model.pt"},
    {"name": "Model_3", "dir": "Model_3", "ckpt": "Model_3/original/model.pt"},
    {"name": "Model_4", "dir": "Model_4", "ckpt": "Model_4/original/model.pt"},
]


def _load_model_module(model_dir, main_dir):
    abs_model_dir = os.path.abspath(os.path.join(main_dir, model_dir))
    if not os.path.isdir(abs_model_dir):
        raise FileNotFoundError(f"Model folder not found: {abs_model_dir}")

    if abs_model_dir in sys.path:
        sys.path.remove(abs_model_dir)
    sys.path.insert(0, abs_model_dir)

    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    lstm_training = importlib.import_module("LSTMTraining")
    lstm_model = importlib.import_module("LSTMModel")
    return lstm_training, lstm_model


def _load_checkpoint_model(lstm_model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    config = lstm_model.Config()
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        for key, value in checkpoint["config"].items():
            if hasattr(config, key):
                setattr(config, key, value)

    model = lstm_model.LSTMForecast(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _collect_predictions(model, loader, device):
    all_q10, all_q50, all_q90, all_tgt = [], [], [], []
    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            q10, q50, q90 = model(enc, dec)
            all_q10.append(q10.cpu().numpy())
            all_q50.append(q50.cpu().numpy())
            all_q90.append(q90.cpu().numpy())
            all_tgt.append(tgt.cpu().numpy())

    return (
        np.concatenate(all_q10),
        np.concatenate(all_q50),
        np.concatenate(all_q90),
        np.concatenate(all_tgt),
    )


def _plot_raw_coverage(q10, q90, targets, alpha, save_path):
    raw_coverage = np.mean((targets >= q10) & (targets <= q90), axis=0)
    raw_mean = raw_coverage.mean() * 100
    raw_mad = np.mean(np.abs(raw_coverage - (1 - alpha))) * 100

    horizons = np.arange(1, len(raw_coverage) + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        horizons,
        raw_coverage * 100,
        label=f"Raw interval coverage (mean={raw_mean:.2f}%, MAD={raw_mad:.2f}%)",
        color="steelblue",
    )
    ax.axhline(
        y=(1 - alpha) * 100,
        color="orange",
        linestyle="--",
        label=f"Nominal {(1 - alpha) * 100:.0f}% target",
    )
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Prediction Interval Coverage per Horizon\nRaw (Test Set)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    main_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(main_dir, "data", "dataset.pt")

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg in MODEL_CONFIGS:
        lstm_training, lstm_model = _load_model_module(cfg["dir"], main_dir)

        (train_dataset, val_dataset, cal_dataset, test_dataset,
         train_size, val_size, cal_size, test_size) = lstm_training.load_and_split_dataset(
            dataset_path
        )

        checkpoint_path = os.path.join(main_dir, cfg["ckpt"])
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, checkpoint = _load_checkpoint_model(lstm_model, checkpoint_path, device)
        alpha = float(checkpoint.get("conformal_alpha", 0.1))

        batch_size = getattr(lstm_model.Config, "batch_size", 64)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 8,
            shuffle=False,
            pin_memory=(device == "cuda"),
            num_workers=min(4, os.cpu_count() or 1),
            persistent_workers=False,
        )

        q10, q50, q90, tgt = _collect_predictions(model, test_loader, device)

        run_dir = os.path.dirname(os.path.join(main_dir, cfg["ckpt"]))
        plot_dir = os.path.join(run_dir, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        coverage_plot_path = os.path.join(plot_dir, "coverage_per_horizon.png")

        _plot_raw_coverage(q10, q90, tgt, alpha, coverage_plot_path)
        print(f"[{cfg['name']}] Saved test coverage plot -> {coverage_plot_path}")


if __name__ == "__main__":
    main()

