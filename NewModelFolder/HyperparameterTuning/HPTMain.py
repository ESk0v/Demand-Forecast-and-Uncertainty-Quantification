from datetime import datetime
import optuna
import torch
import os
import json
import logging
from HyperparameterTuning.HPTHelpers import (
    load_dataset, trialSuggestions
)

def hptmain(n_trials, epochs, patience, local, filePaths, logger=None):
    
    optuna.logging.disable_default_handler()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hyperparameter tuning is running on device: {device}")
    
    # Enable cudnn benchmarking for faster training if on GPU
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("✓ Enabled cudnn.benchmark for faster GPU training")
    
    # Load dataset
    train_dataset, val_dataset, _ = load_dataset(local, filePaths, logger)
    
    # Create Optuna study
    study_name = f"lstm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    logger.info(f"Running hyperparameter search with {n_trials} trials and {epochs} epochs per trial... ")    

    # Run trials one by one and log after each
    num_workers = min(4, os.cpu_count())  # use 4 or as many CPUs as available
    logger.debug(f"Using num_workers={num_workers} for DataLoader")
    for i in range(n_trials):
        study.optimize(
            lambda trial: trialSuggestions(trial, patience, train_dataset, val_dataset, device, local, logger, epoch=epochs, workers=num_workers),
            n_trials=1
        ) 

        current_trial_loss = study.trials[-1].value
        if current_trial_loss is None:
            logger.error(f"Trial {i} failed.")
        else:
            logger.success(
                f"Trial {i} completed with validation loss: {current_trial_loss:.6f} | "
                f"Current best: Trial {study.best_trial.number} with loss {study.best_value:.6f}"
            )
    # Print results
    logger.info("Hyperparameter tuning evaluation")
    logger.info(f"Best trial: Validation Loss: {study.best_trial.value:.6f}")
    logger.info(f"Best hyperparameters:\n"
            f"                                                             hidden size: {study.best_trial.params['hidden_size']}\n"
            f"                                                             number of layers: {study.best_trial.params['num_layers']}\n"
            f"                                                             dropout: {study.best_trial.params['dropout']:.6f}\n"
            f"                                                             context dropout: {study.best_trial.params['context_dropout']:.6f}\n"
            f"                                                             batch size: {study.best_trial.params['batch_size']}\n"
            f"                                                             learning rate: {study.best_trial.params['learning_rate']:.6f}")

    best_params_file = filePaths[1]
    
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    with open(best_params_file, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    return study