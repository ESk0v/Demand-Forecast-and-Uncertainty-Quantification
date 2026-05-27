from datetime import datetime
import optuna
import torch
import os
import json
from HyperparameterTuning.HPTHelpers import (
    load_dataset, trialSuggestions
)

def hptmain(n_trials, epochs, patience, local, filePaths, logger=None):

    optuna.logging.disable_default_handler()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hyperparameter tuning running on: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    # load_dataset returns 4 splits — tuning only uses train + val
    train_dataset, val_dataset, _, _ = load_dataset(local, filePaths, logger)

    study_name = f"lstm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    logger.info(f"Running {n_trials} trials × {epochs} epochs each...")

    num_workers = min(4, os.cpu_count())
    logger.debug(f"DataLoader num_workers={num_workers}")

    for i in range(n_trials):
        study.optimize(
            lambda trial: trialSuggestions(
                trial, patience, train_dataset, val_dataset,
                device, local, logger, epoch=epochs, workers=num_workers
            ),
            n_trials=1
        )

        current_loss = study.trials[-1].value
        if current_loss is None:
            logger.error(f"Trial {i} failed.")
        else:
            logger.success(
                f"Trial {i} — val_loss: {current_loss:.6f} | "
                f"Best so far: Trial {study.best_trial.number} "
                f"val_loss={study.best_value:.6f}"
            )

    # ── Results ────────────────────────────────────────────────────────────────
    best = study.best_trial
    logger.info(
        f"Tuning complete — best trial: {best.number}  "
        f"val_loss={best.value:.6f}\n"
        f"  hidden_size    : {best.params['hidden_size']}\n"
        f"  num_layers     : {best.params['num_layers']}\n"
        f"  dropout        : {best.params['dropout']:.6f}\n"
        f"  context_dropout: {best.params['context_dropout']:.6f}\n"
        f"  batch_size     : {best.params['batch_size']}\n"
        f"  learning_rate  : {best.params['learning_rate']:.6f}"
    )

    best_params_file = filePaths[1]
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    with open(best_params_file, 'w') as f:
        json.dump(best.params, f, indent=2)

    logger.success(f"Best params saved to {best_params_file}")
    return study