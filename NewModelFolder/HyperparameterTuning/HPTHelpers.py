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


# ── Loss ───────────────────────────────────────────────────────────────────────

def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


# ── Training loop (tuning version) ────────────────────────────────────────────

def train_model(config, train_loader, val_loader, train_size, val_size, device,
                trial=None, max_epochs=None, patience=None, logger=None):
    """
    Lightweight training loop used during hyperparameter tuning.
    Uses quantile loss (q10/q50/q90) to match LSTMTraining.py exactly.
    No checkpoint saving — only returns best_val_loss for Optuna.
    """
    model     = LSTMForecast(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(1, (patience or 5) // 2)
    )

    best_val_loss     = np.inf
    epochs_no_improve = 0
    best_model_state  = None
    epochs            = max_epochs if max_epochs is not None else config.epochs

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0

        for enc, dec, tgt in train_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            optimizer.zero_grad()

            q10, q50, q90 = model(enc, dec)
            loss = (0.25 * quantile_loss(q10, tgt, 0.1)
                  + 0.50 * quantile_loss(q50, tgt, 0.5)
                  + 0.25 * quantile_loss(q90, tgt, 0.9))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item() * enc.size(0)

        train_loss = epoch_loss / train_size

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                q10, q50, q90 = model(enc, dec)
                loss = (0.25 * quantile_loss(q10, tgt, 0.1)
                      + 0.50 * quantile_loss(q50, tgt, 0.5)
                      + 0.25 * quantile_loss(q90, tgt, 0.9))
                val_loss_epoch += loss.item() * enc.size(0)

        val_loss = val_loss_epoch / val_size
        scheduler.step(val_loss)

        # ── Optuna pruning ─────────────────────────────────────────────────────
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.warning(f"Trial pruned at epoch {epoch} (val_loss={val_loss:.6f})")
                raise optuna.TrialPruned()

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_model_state  = copy.deepcopy(model.state_dict())
            if logger is not None:
                logger.info(
                    f"  Epoch {epoch}/{epochs} — train: {train_loss:.4f}  "
                    f"val: {val_loss:.4f}  [best]"
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger is not None:
                    logger.info(f"  Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss, model


# ── Trial entry point ──────────────────────────────────────────────────────────

def trialSuggestions(trial: Trial, patience=None, train_dataset=None, val_dataset=None,
                     device=None, local=False, logger=None, epoch=None, workers=None):

    config = Config()

    config.hidden_size     = trial.suggest_int('hidden_size', 32, 512, step=32)
    config.num_layers      = trial.suggest_int('num_layers', 1, 5)
    config.dropout         = trial.suggest_float('dropout', 0.05, 0.6)
    config.context_dropout = trial.suggest_float('context_dropout', 0.025, 0.4)
    config.batch_size      = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    config.learning_rate   = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    config.device          = device

    logger.info(
        f"Trial {trial.number} — "
        f"hidden: {config.hidden_size}  layers: {config.num_layers}  "
        f"dropout: {config.dropout:.3f}  ctx_drop: {config.context_dropout:.3f}  "
        f"batch: {config.batch_size}  lr: {config.learning_rate:.2e}"
    )

    train_size = len(train_dataset)
    val_size   = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=workers,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=workers,
        persistent_workers=False,
    )

    try:
        best_val_loss, _ = train_model(
            config, train_loader, val_loader, train_size, val_size,
            device, trial=trial, max_epochs=epoch, patience=patience, logger=logger
        )
        return best_val_loss

    except optuna.TrialPruned:
        raise  # let Optuna handle it

    except Exception as e:
        logger.error(
            f"Trial {trial.number} failed.\n"
            f"Params: {config.__dict__}\n"
            f"Error: {e}\n{traceback.format_exc()}"
        )
        raise


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(local=False, filePaths=None, logger=None,
                 val_ratio=0.1, cal_ratio=0.05, test_ratio=0.1):
    """
    4-way chronological split — matches LSTMTraining.py exactly.
    train=75%  val=10%  cal=5%  test=10%

    Tuning only uses train + val. Cal and test are returned but not used
    during the search — they must stay untouched until final evaluation.
    """
    dataset_path = filePaths[0]

    dataset      = torch.load(dataset_path, weights_only=False)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total    = len(full_dataset)
    val_size   = int(n_total * val_ratio)
    cal_size   = int(n_total * cal_ratio)
    test_size  = int(n_total * test_ratio)
    train_size = n_total - val_size - cal_size - test_size

    i1 = train_size
    i2 = i1 + val_size
    i3 = i2 + cal_size
    i4 = i3 + test_size

    train_dataset = Subset(full_dataset, range(0,  i1))
    val_dataset   = Subset(full_dataset, range(i1, i2))
    cal_dataset   = Subset(full_dataset, range(i2, i3))
    test_dataset  = Subset(full_dataset, range(i3, i4))

    logger.info(
        f"Dataset split — train: {train_size}  val: {val_size}  "
        f"cal: {cal_size}  test: {test_size}  (total: {n_total})"
    )

    # Tuning only needs train + val. Cal/test returned for completeness.
    return train_dataset, val_dataset, cal_dataset, test_dataset