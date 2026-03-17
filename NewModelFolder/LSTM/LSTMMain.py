from torch.utils.data import DataLoader
from LSTMModel import Config
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTM.LSTMTraining import load_and_split_dataset, train_model
from LSTM.GenerateREADME import generate_training_readme

def LSTMMain(filePaths=None, epochs=1, patience=5, logger=None):
    # Paths
    dataset_path    = filePaths[0]
    model_save_path = filePaths[1]
    run_dir = os.path.dirname(model_save_path)
    os.makedirs(run_dir, exist_ok=True)

    # Load and split dataset
    train_dataset, val_dataset, train_size, val_size = load_and_split_dataset(dataset_path)
    logger.info(f"Dataset loaded: {train_size} training samples, {val_size} validation samples")

    # Compute test_size and n_total for the README
    raw = torch.load(dataset_path, weights_only=False)
    n_total   = len(raw['target'])
    test_size = n_total - train_size - val_size

    # Create config and data loaders
    config = Config()
    config.epochs = epochs

    num_workers = min(4, os.cpu_count())  # use 4 or as many CPUs as available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=(device =='cuda'), 
        num_workers=num_workers, 
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        pin_memory=(device =='cuda'), 
        num_workers=num_workers, 
        persistent_workers=False
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    best_val_loss, train_losses, val_losses = train_model(
        config, train_loader, val_loader, train_size, val_size,
        model_save_path, logger, patience=patience
    )

    # Load checkpoint for metadata
    checkpoint = torch.load(model_save_path, weights_only=False)
    logger.success("LSTM training completed successfully!")
    logger.info("Generating training README...")

    # Generate training README in run_dir
    generate_training_readme(
        plot_dir      = run_dir,
        model_filename = os.path.basename(model_save_path),
        config        = config,
        train_size    = train_size,
        val_size      = val_size,
        test_size     = test_size,
        n_total       = n_total,
        epochs_run    = len(train_losses),
        best_epoch    = checkpoint['epoch'],
        best_val_loss = checkpoint['val_loss'],
        early_stopped = (len(train_losses) < config.epochs),
        patience      = patience
    )

    logger.success("Training README successfully generated")