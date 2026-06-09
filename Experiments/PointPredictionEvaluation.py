import os
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


MODEL_CONFIGS = [
    dict(
        name="Model 1 Decoupled Detach",
        module_dir="Experiment_3/Model_1_decoupled_detach",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_1_decoupled_detach/checkpoints/model_decoupled.pt",
    ),
    dict(
        name="Model 2 Decoupled No Detach",
        module_dir="Experiment_3/Model_2_decoupled_no_detach",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_2_decoupled_no_detach/checkpoints/model_decoupled.pt",
    ),
    dict(
        name="Model 3 Coupled 2 Loss",
        module_dir="Experiment_3/Model_3_coupled_2loss",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_3_coupled_2loss/checkpoints/model_coupled.pt",
    ),
    dict(
        name="Model 4 Coupled 2 Sub-loss",
        module_dir="Experiment_3/Model_4_coupled_2subloss",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_4_coupled_2subloss/checkpoints/model_coupled.pt",
    ),
    dict(
        name="Model 6 Test",
        module_dir="Experiment_3/Model_6",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_6/checkpoints/model.pt",
    ),
    dict(
        name="Model 7 Test 2",
        module_dir="Experiment_3/Model_7",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_7/checkpoints/model.pt",
    ),
    dict(
        name="Model 8 Test 3",
        module_dir="Experiment_3/Model_8",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_8/checkpoints/model.pt",
    ),
    dict(
        name="Model 9 Test 3",
        module_dir="Experiment_3/Model_9",
        dataset_path="Experiment_3/data/dataset.pt",
        model_save_path="Experiment_3/Model_9/checkpoints/model.pt",
    ),
]


VAL_RATIO = 1.0 / 12.0
CAL_RATIO = 1.0 / 12.0
TEST_RATIO = 1.0 / 6.0


def _abs(path_from_this_file: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path_from_this_file))


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
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


def _load_model(cfg, checkpoint):
    module_path = _abs(os.path.join(cfg["module_dir"], "LSTMModel.py"))
    module = _load_module(f"model_{cfg['name'].replace(' ', '_')}", module_path)

    config = module.Config()
    ckpt_config = checkpoint.get("config", {})

    if isinstance(ckpt_config, dict):
        for k, v in ckpt_config.items():
            if hasattr(config, k):
                setattr(config, k, v)

    config.device = "cpu"

    model = module.LSTMForecast(config).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, int(getattr(config, "batch_size", 64))


def _predict_q50(model, loader):
    q50_all, tgt_all = [], []

    with torch.no_grad():
        for enc, dec, tgt in loader:
            _, q50, _ = model(enc, dec)
            q50_all.append(q50.cpu().numpy())
            tgt_all.append(tgt.cpu().numpy())

    return np.concatenate(q50_all), np.concatenate(tgt_all)


def _compute_abs_errors(cfg):
    ckpt_path = _abs(cfg["model_save_path"])
    dataset_path = _abs(cfg["dataset_path"])

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint: {cfg['name']}")
        return None

    if not os.path.exists(dataset_path):
        print(f"[WARN] Missing dataset: {cfg['name']}")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, batch_size = _load_model(cfg, checkpoint)

    dataset_raw, test_ds = _split_test_dataset(dataset_path)
    loader = DataLoader(test_ds, batch_size=max(256, batch_size * 8), shuffle=False)

    q50, tgt = _predict_q50(model, loader)

    mean = float(dataset_raw.get("demand_mean", 0.0))
    std = float(dataset_raw.get("demand_std", 1.0))

    q50 = q50 * std + mean
    tgt = tgt * std + mean

    # ✅ CRITICAL FIX: flatten to 1D distribution
    abs_errors = np.abs(tgt - q50).reshape(-1)

    return abs_errors


def plot_boxplot(all_errors, labels, save_path):
    plt.figure(figsize=(12, 6))

    # ✅ FIX: matplotlib 3.9+ compatibility
    plt.boxplot(all_errors, tick_labels=labels, showfliers=True)

    plt.title("Point Prediction Error Distribution (|y - q50|)")
    plt.ylabel("Absolute Error (original scale)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[SUCCESS] Saved plot: {save_path}")


def main():
    all_errors = []
    labels = []

    for cfg in MODEL_CONFIGS:
        print(f"[INFO] Processing {cfg['name']}")

        err = _compute_abs_errors(cfg)
        if err is None:
            continue

        all_errors.append(err)
        labels.append(cfg["name"])

    if not all_errors:
        print("[ERROR] No valid models found.")
        return

    save_path = _abs("point_prediction_boxplot.png")
    plot_boxplot(all_errors, labels, save_path)


if __name__ == "__main__":
    main()