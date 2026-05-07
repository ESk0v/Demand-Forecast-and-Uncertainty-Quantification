"""
GradientInterference/main.py
─────────────────────────────
Trains two model variants to compare gradient interference:
  - Model_1: WITH detach    (your architecture — stop-gradient)
  - Model_2: WITHOUT detach (baseline — gradients flow freely)

Each model folder contains its own LSTMModel.py, LSTMTraining.py and LSTMMain.py.
This file dynamically imports and calls LSTMMain from each folder.

Run InterferencePlot.py separately to generate the visualisation.
"""

import os
import sys
import importlib
import random
import numpy as np
import torch

# ─── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Experiment configs ───────────────────────────────────────────────────────
# module_dir   → subfolder containing LSTMModel.py, LSTMTraining.py, LSTMMain.py
# dataset_path → path to dataset.pt relative to this file
# save_path    → where the trained checkpoint will be saved
# use_detach   → informational only — actual behaviour is defined in LSTMModel.py

EXPERIMENT_CONFIGS = [
    dict(
        name            = "With detach",
        module_dir      = "Model_1",
        dataset_path    = "Model_1/data/dataset.pt",
        save_path       = "Model_1/checkpoints/model.pt",
        use_detach      = True,
        epochs          = 50,
        patience        = 5,
    ),
    dict(
        name            = "Without detach",
        module_dir      = "Model_2",
        dataset_path    = "Model_2/data/dataset.pt",
        save_path       = "Model_2/checkpoints/model.pt",
        use_detach      = False,
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


# ─── Run one experiment ───────────────────────────────────────────────────────
def run_one_experiment(cfg: dict):
    """
    Dynamically imports LSTMMain from the experiment's module_dir
    and calls it with the configured dataset and save paths.
    """
    module_dir     = cfg["module_dir"]
    main_dir       = os.path.abspath(os.path.dirname(__file__))
    abs_module_dir = os.path.abspath(os.path.join(main_dir, module_dir))

    if not os.path.isdir(abs_module_dir):
        raise FileNotFoundError(
            f"Module folder not found: {abs_module_dir}\n"
            f"Expected '{module_dir}' to be a subfolder next to main.py."
        )

    # ── Fresh import — clear any cached modules from a previous experiment ─
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    sys.path.insert(0, abs_module_dir)

    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    for mod_name in ("LSTMModel", "LSTMTraining", "LSTMMain"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    lstm_main_module = importlib.import_module("LSTMMain")
    LSTMMain         = lstm_main_module.LSTMMain

    # ── Resolve paths ──────────────────────────────────────────────────────
    dataset_path = os.path.join(main_dir, cfg["dataset_path"])
    save_path    = os.path.join(main_dir, cfg["save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Place dataset.pt inside {os.path.join(abs_module_dir, 'data')}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    set_seed(GLOBAL_SEED)

    # ── Call LSTMMain ──────────────────────────────────────────────────────
    LSTMMain(
        filePaths = [dataset_path, save_path],
        epochs    = cfg["epochs"],
        patience  = cfg["patience"],
        logger    = logger,
    )

    # ── Clean up imports so next experiment gets a fresh load ──────────────
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    for mod_name in ("LSTMModel", "LSTMTraining", "LSTMMain"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    set_seed(GLOBAL_SEED)

    for cfg in EXPERIMENT_CONFIGS:
        logger.info(f"══ Experiment: {cfg['name']} ══")
        run_one_experiment(cfg)
        logger.success(f"[{cfg['name']}] Complete.")

    logger.success(
        "All experiments complete. "
        "Run InterferencePlot.py to generate the visualisation."
    )


if __name__ == "__main__":
    main()