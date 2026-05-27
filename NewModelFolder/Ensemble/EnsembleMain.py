import sys
from pathlib import Path
import os
from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config

def main(filePaths=None, epochs=1, n_models=3, patience=1, logger=None):

    # -----------------------------
    # Paths
    # -----------------------------
    dataset_path      = filePaths[0]
    run_dir           = filePaths[2]
    plot_dir          = os.path.join(run_dir, "Plots")
    ensemble_save_dir = os.path.join(run_dir, "EnsembleModels")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ensemble_save_dir, exist_ok=True)

    # Configuration
    config = Config()
    batch_size = config.batch_size
    device = config.device

    # Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset, batch_size)

    # Train Ensemble
    _TrainEnsemble(
        n_models=n_models,
        epochs=epochs,
        patience=patience,
        train_loader=trainLoader,
        val_loader=valLoader,
        config=config,
        save_dir=ensemble_save_dir,
        logger=logger
    )
    logger.success("Ensemble training completed successfully!")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ensemble_save_dir, config)

    logger.info("Generating Ensemble plots...")

    _EvaluateModel(testLoader, models, device, demand_mean, demand_std, n_models, plot_dir=plot_dir)
