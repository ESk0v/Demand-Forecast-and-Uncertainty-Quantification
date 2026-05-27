from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

import sys
import os
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import LSTMForecast


def _DataLoader(dataset_path):
    """Load dataset from the given path and return tensors + demand stats."""
    dataset = torch.load(dataset_path, weights_only=False)

    tensor_dataset = TensorDataset(
        dataset['encoder'],
        dataset['decoder'],
        dataset['target']
    )

    if 'demand_mean' in dataset and 'demand_std' in dataset:
        demand_mean = dataset['demand_mean']
        demand_std  = dataset['demand_std']
    else:
        targets = dataset['target'].numpy()
        demand_mean = float(targets.mean())
        demand_std  = float(targets.std())

    return tensor_dataset, demand_mean, demand_std


def _DatasetSplit(dataset, batch_size):
    # Split dataset into train/val/test

    val_ratio, test_ratio = 0.1, 0.1
    n_total = len(dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, n_total))

    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainLoader, valLoader, testLoader


def _EnsemblePredict(models, enc, dec, demand_mean, demand_std):
    mus = []
    vars_ = []

    for model in models:
        model.eval()
        with torch.no_grad():
            mu, log_var = model(enc, dec)
            var = torch.exp(log_var)

            mus.append(mu.cpu())
            vars_.append(var.cpu())

    mus = torch.stack(mus, dim=0)
    vars_ = torch.stack(vars_, dim=0)

    # Ensemble mean (normalized space)
    mean_norm = mus.mean(dim=0)

    # Epistemic variance
    epistemic_norm = mus.var(dim=0, unbiased=False)

    # Aleatoric variance
    aleatoric_norm = vars_.mean(dim=0)

    total_var_norm = epistemic_norm + aleatoric_norm
    total_std_norm = torch.sqrt(total_var_norm)

    # Convert back to real MW units
    mean_real = mean_norm * demand_std + demand_mean
    total_std_real = total_std_norm * demand_std
    epistemic_real = torch.sqrt(epistemic_norm) * demand_std
    aleatoric_real = torch.sqrt(aleatoric_norm) * demand_std

    return mean_real, total_std_real, epistemic_real, aleatoric_real


def _LoadEnsembleModels(model_dir, config):
    device = config.device
    model_dir = Path(model_dir)
    model_paths = sorted(model_dir.glob("lstm_seq2seq_model_*.pth"))

    models = []

    for path in model_paths:
        with torch.serialization.safe_globals([type(config)]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        saved_config = checkpoint["config"]
        model = LSTMForecast(saved_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    return models


def _TrainEnsemble(n_models, epochs, patience, train_loader, val_loader, save_dir, config, base_seed=1000, logger=None):
    """
    Train an ensemble of encoder-decoder LSTM models with different random seeds.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_paths = []

    for i in range(n_models):
        seed = base_seed + i
        logger.info(f"Training ensemble model {i+1}/{n_models} with seed {seed}")

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model and config
        config.epochs=epochs
        device = config.device
        model = LSTMForecast(config).to(device)

        criterion = nn.GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = patience

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0

            for enc, dec, tgt in train_loader:
                enc, dec, tgt = enc.to(device).float(), dec.to(device).float(), tgt.to(device).float()

                optimizer.zero_grad()
                mu, log_var = model(enc, dec)
                loss = criterion(mu, tgt, torch.exp(log_var))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * enc.size(0)

            train_loss = epoch_loss / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for enc, dec, tgt in val_loader:
                    enc, dec, tgt = enc.to(device).float(), dec.to(device).float(), tgt.to(device).float()
                    mu, log_var = model(enc, dec)
                    loss = criterion(mu, tgt, torch.exp(log_var))
                    val_loss_epoch += loss.item() * enc.size(0)

            val_loss = val_loss_epoch / len(val_loader.dataset)
            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Model {i+1} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model_path = save_dir / f"lstm_seq2seq_model_{i+1}.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config
                }, model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        logger.success(f"Finished training model at epoch: {epoch}")
        model_paths.append(model_path)

    return model_paths