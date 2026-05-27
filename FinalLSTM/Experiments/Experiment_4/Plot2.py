import os
import sys
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader


def _mean_winkler_score(q_low, q_high, target, alpha):
    q_low = np.asarray(q_low, dtype=np.float64)
    q_high = np.asarray(q_high, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Guard against crossed intervals.
    lo = np.minimum(q_low, q_high)
    hi = np.maximum(q_low, q_high)

    width = hi - lo
    penalty_low = (2.0 / alpha) * np.clip(lo - target, a_min=0.0, a_max=None)
    penalty_high = (2.0 / alpha) * np.clip(target - hi, a_min=0.0, a_max=None)
    return float(np.mean(width + penalty_low + penalty_high))


def _coverage_mad(q_low, q_high, target, alpha):
    q_low = np.asarray(q_low, dtype=np.float64)
    q_high = np.asarray(q_high, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    lo = np.minimum(q_low, q_high)
    hi = np.maximum(q_low, q_high)

    # coverage per horizon
    coverage_h = np.mean((target >= lo) & (target <= hi), axis=0)
    nominal = 1.0 - alpha
    return float(np.mean(np.abs(coverage_h - nominal)))


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
        for k, v in checkpoint["config"].items():
            if hasattr(config, k):
                setattr(config, k, v)

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


def _denorm(values, mean, std):
    return values * std + mean


def main():
    main_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(main_dir, "data", "dataset.pt")

    dataset_blob = torch.load(dataset_path, weights_only=False)
    demand_mean = float(dataset_blob.get("demand_mean", 0.0))
    demand_std = float(dataset_blob.get("demand_std", 1.0))

    model_configs = [
        # {"name": "Model_1", "dir": "Model_1", "ckpt": "Model_1/original/model.pt"},
        # {"name": "Model_2", "dir": "Model_2", "ckpt": "Model_2/original/model.pt"},
        {"name": "Model_3", "dir": "Model_3", "ckpt": "Model_3/original/model.pt"},
        # {"name": "Model_4", "dir": "Model_4", "ckpt": "Model_4/original/model.pt"},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nModel Results (raw intervals)\n" + "-" * 60)
    for cfg in model_configs:
        lstm_training, lstm_model = _load_model_module(cfg["dir"], main_dir)

        # split data using each model's training code
        (train_dataset, val_dataset, cal_dataset, test_dataset,
         train_size, val_size, cal_size, test_size) = lstm_training.load_and_split_dataset(
            dataset_path
        )

        checkpoint_path = os.path.join(main_dir, cfg["ckpt"])
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, checkpoint = _load_checkpoint_model(lstm_model, checkpoint_path, device)

        alpha = 0.3 #float(checkpoint.get("conformal_alpha", 0.1))

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

        # De-normalize to original demand units
        q10 = _denorm(q10, demand_mean, demand_std)
        q90 = _denorm(q90, demand_mean, demand_std)
        tgt = _denorm(tgt, demand_mean, demand_std)

        winkler = _mean_winkler_score(q10, q90, tgt, alpha=alpha)
        cov_mad = _coverage_mad(q10, q90, tgt, alpha=alpha)

        print(
            f"{cfg['name']}: "
            f"Winkler={winkler:.4f}, "
            f"CoverageMAD={cov_mad*100:.2f}% "
            f"(alpha={alpha:.2f})"
        )

    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()