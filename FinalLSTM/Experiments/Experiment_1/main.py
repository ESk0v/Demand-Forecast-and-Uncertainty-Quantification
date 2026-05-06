"""
CoveragePlot/main.py
────────────────────
Trains N models (each with its own conformal alpha / coverage target)
and saves checkpoints. Run Plot.py separately to generate the plot.
"""

import os
import sys
import random
import numpy as np
import torch
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

MODEL_CONFIGS = [
    dict(
        name            = "50% target",
        module_dir      = "Model_1",
        dataset_path    = "Model_1/data/dataset.pt",
        model_save_path = "Model_1/checkpoints/model.pt",
        conformal_alpha = 0.50,   # 1 - 0.50
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "60% target",
        module_dir      = "Model_2",
        dataset_path    = "Model_2/data/dataset.pt",
        model_save_path = "Model_2/checkpoints/model.pt",
        conformal_alpha = 0.40,   # 1 - 0.60
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "70% target",
        module_dir      = "Model_3",
        dataset_path    = "Model_3/data/dataset.pt",
        model_save_path = "Model_3/checkpoints/model.pt",
        conformal_alpha = 0.30,   # 1 - 0.70
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "80% target",
        module_dir      = "Model_4",
        dataset_path    = "Model_4/data/dataset.pt",
        model_save_path = "Model_4/checkpoints/model.pt",
        conformal_alpha = 0.20,   # 1 - 0.80
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


# ─── Minimal logger ───────────────────────────────────────────────────────────
class _SimpleLogger:
    def info(self,    msg): print(f"[INFO]    {msg}")
    def success(self, msg): print(f"[SUCCESS] {msg}")
    def warning(self, msg): print(f"[WARN]    {msg}")
    def error(self,   msg): print(f"[ERROR]   {msg}")

logger = _SimpleLogger()


# ─── Training helper ──────────────────────────────────────────────────────────
def train_one_model(cfg: dict):
    """
    Dynamically imports LSTMModel / LSTMTraining from the model-specific
    subfolder and trains the model, saving the checkpoint to disk.
    """
    import importlib

    module_dir     = cfg["module_dir"]
    main_dir       = os.path.abspath(os.path.dirname(__file__))
    abs_module_dir = os.path.abspath(os.path.join(main_dir, module_dir))

    if not os.path.isdir(abs_module_dir):
        raise FileNotFoundError(
            f"Model folder not found: {abs_module_dir}\n"
            f"Expected 'module_dir' to be a subfolder next to main.py."
        )

    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    sys.path.insert(0, abs_module_dir)

    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    lstm_training = importlib.import_module("LSTMTraining")
    lstm_model    = importlib.import_module("LSTMModel")

    load_and_split_dataset = lstm_training.load_and_split_dataset
    train_model            = lstm_training.train_model
    Config                 = lstm_model.Config

    set_seed(GLOBAL_SEED)

    dataset_path    = os.path.join(main_dir, cfg["dataset_path"])
    model_save_path = os.path.join(main_dir, cfg["model_save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Place your dataset.pt inside {os.path.join(abs_module_dir, 'data')}"
        )

    (train_dataset, val_dataset, cal_dataset, test_dataset,
     train_size, val_size, cal_size, test_size) = load_and_split_dataset(
        dataset_path
    )

    config        = Config()
    config.epochs = cfg["epochs"]
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers   = min(4, os.cpu_count() or 1)

    def make_loader(ds, shuffle, bs):
        return DataLoader(
            ds,
            batch_size         = bs,
            shuffle            = shuffle,
            pin_memory         = (device == "cuda"),
            num_workers        = num_workers,
            persistent_workers = False,
        )

    train_loader = make_loader(train_dataset, True,  config.batch_size)
    val_loader   = make_loader(val_dataset,   False, config.batch_size * 8)
    cal_loader   = make_loader(cal_dataset,   False, config.batch_size * 8)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"[{cfg['name']}] Starting training …")

    train_model(
        config, train_loader, val_loader, cal_loader,
        train_size, val_size, cal_size,
        model_save_path,
        logger          = logger,
        patience        = cfg["patience"],
        conformal_alpha = cfg["conformal_alpha"],
    )

    logger.success(f"[{cfg['name']}] Checkpoint saved → {model_save_path}")

    # Clean up imports so the next model gets a fresh load
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    set_seed(GLOBAL_SEED)

    for cfg in MODEL_CONFIGS:
        logger.info(f"══ Model: {cfg['name']} ══")
        train_one_model(cfg)

    logger.success("All models trained. Run Plot.py to generate the reliability diagram.")


if __name__ == "__main__":
    main()