import os
import sys
import importlib
import random
import numpy as np
import torch

GLOBAL_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


EXPERIMENT_CONFIGS = [
    dict(
        name="With detach",
        module_dir="Model_1",
        dataset_path="Model_1/data/dataset.pt",
        save_path="Model_1/checkpoints/model.pt",
        epochs=2,
        patience=5,
    ),
    dict(
        name="Without detach",
        module_dir="Model_2",
        dataset_path="Model_2/data/dataset.pt",
        save_path="Model_2/checkpoints/model.pt",
        epochs=2,
        patience=5,
    ),
]


class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def success(self, msg):
        print(f"[SUCCESS] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


logger = Logger()


def run_one_experiment(cfg):

    module_dir = cfg["module_dir"]

    base_dir = os.path.abspath(os.path.dirname(__file__))
    module_path = os.path.join(base_dir, module_dir)

    sys.path.insert(0, module_path)

    for mod in ("LSTMModel", "LSTMTraining", "LSTMMain"):
        if mod in sys.modules:
            del sys.modules[mod]

    lstm_main_module = importlib.import_module("LSTMMain")

    dataset_path = os.path.join(base_dir, cfg["dataset_path"])
    save_path = os.path.join(base_dir, cfg["save_path"])

    set_seed(GLOBAL_SEED)

    lstm_main_module.LSTMMain(
        filePaths=[dataset_path, save_path],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        logger=logger
    )

    sys.path.remove(module_path)


def main():

    set_seed(GLOBAL_SEED)

    for cfg in EXPERIMENT_CONFIGS:

        logger.info(f"Running {cfg['name']}")

        run_one_experiment(cfg)

        logger.success(f"{cfg['name']} completed")

    logger.success("All training complete")
    logger.success("Run Plot.py next")


if __name__ == "__main__":
    main()