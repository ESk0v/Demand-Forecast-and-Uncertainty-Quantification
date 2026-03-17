import copy
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import optuna
from optuna.trial import Trial
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import Config, LSTMForecast


def train_model(config, train_loader, val_loader, train_size, val_size, device, 
                trial=None, max_epochs=None, patience=None, logger=None):

    model = LSTMForecast(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    epochs = max_epochs if max_epochs is not None else config.epochs
    
    # Gradient accumulation
    accumulation_steps = 4
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, (enc, dec, tgt) in enumerate(train_loader):
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            
            # Mixed precision forward + loss
            with torch.amp.autocast(device_type='cuda', enabled=(device == "cuda")):
                output, _ = model(enc, dec)
                loss = criterion(output, tgt)
                loss = loss / accumulation_steps  # scale for gradient accumulation

            # Backward with GradScaler
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * enc.size(0) * accumulation_steps

        # Handle remaining gradients if not divisible
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss = epoch_loss / train_size
        
        # Validation
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                output, _ = model(enc, dec)
                val_loss_epoch += criterion(output, tgt).item() * enc.size(0)
        val_loss = val_loss_epoch / val_size
        
        scheduler.step(val_loss)
        
        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                #raise optuna.TrialPruned()
                logger.warning(f"Trial should be pruned at epoch {epoch} with val loss {val_loss:.6f}")
                break
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())

            logger.info(
                f"New best epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger is not None:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, model

def trialSuggestions(trial: Trial, patience=None, train_dataset=None, val_dataset=None, device=None, 
              local=False, logger=None, epoch=None, workers=None):
    
    # Create config with suggested hyperparameters
    config = Config()
    
    # Suggest hyperparameters - OPTIMIZED RANGES
    config.hidden_size     = trial.suggest_int('hidden_size', 32, 128, step=32)
    config.num_layers      = trial.suggest_int('num_layers', 1, 3)
    config.dropout         = trial.suggest_float('dropout', 0.1, 0.4)
    config.context_dropout = trial.suggest_float('context_dropout', 0.05, 0.3)
    config.batch_size      = trial.suggest_categorical('batch_size', [32, 64, 128])
    config.learning_rate   = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    config.device = device
    
    logger.info(f"Trial {trial.number} starting with hyperparameters:\n"
                f"                                                             \033[1mhidden size     :\033[0m\033[37m {config.hidden_size}\n"
                f"                                                             \033[1mnumber of layers:\033[0m\033[37m {config.num_layers}\n"
                f"                                                             \033[1mdropout         :\033[0m\033[37m {config.dropout:.4f}\n"
                f"                                                             \033[1mcontext dropout :\033[0m\033[37m {config.context_dropout:.4f}\n"
                f"                                                             \033[1mbatch size      :\033[0m\033[37m {config.batch_size}\n"
                f"                                                             \033[1mlearning rate   :\033[0m\033[37m {config.learning_rate:.6g}")
    
    # Create data loaders with OPTIMIZED SETTINGS
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    # pin_memory and num_workers for faster data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=(device == "cuda"),  # Faster GPU transfer
        num_workers=workers,        # Parallel data loading
        persistent_workers=False        # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=workers,
        persistent_workers=False
    )
    
    try:
        # Train model with early stopping
        best_val_loss, _ = train_model(
            config, train_loader, val_loader, train_size, val_size, 
            device, trial=trial, max_epochs=epoch, patience=patience, logger=logger
        )
        
        return best_val_loss
    
    except Exception as e:
        logger.error(
            f"Trial {trial.number} failed.\n"
            f"Hyperparameters: {config.__dict__}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        raise  # Re-raise to let Optuna handle it properly


def load_dataset(local=False, filePaths=None, logger=None):
    
    # Setup paths
    dataset_path = filePaths[0]

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    logger.info("Dataset loaded")

    # Train/Val/Test split
    val_ratio = 0.1
    test_ratio = 0.1
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size
    
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))
    
    logger.info(f"Dataset split into \033[1m(train | val | test)\033[0m\033[37m with sizes: \033[1m({train_size} | {val_size} | {test_size})\033[0m\033[37m")
    
    return train_dataset, val_dataset, test_dataset

