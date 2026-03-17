import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import LSTMForecast


def load_and_split_dataset(dataset_path, val_ratio=0.1, test_ratio=0.1):

    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    # Train/Val/Test Split (chronological — no data leakage)
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))

    return train_dataset, val_dataset, train_size, val_size


def train_epoch(model, train_loader, optimizer, criterion, device, train_size):

    model.train()
    epoch_loss = 0

    for enc, dec, tgt in train_loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        optimizer.zero_grad()
        output, _ = model(enc, dec)
        loss = criterion(output, tgt)
        loss.backward()
        # Gradient clipping to prevent gradient explosion with 168-step sequences
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        epoch_loss += loss.item() * enc.size(0)

    train_loss = epoch_loss / train_size
    return train_loss


def validate_epoch(model, val_loader, criterion, device, val_size):
    model.eval()
    val_loss_epoch = 0

    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            output, _ = model(enc, dec)
            val_loss_epoch += criterion(output, tgt).item() * enc.size(0)

    val_loss = val_loss_epoch / val_size
    return val_loss


def save_checkpoint(model, optimizer, config, epoch, val_loss, train_losses, val_losses, model_save_path):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'epoch': epoch,
        'val_loss': val_loss,
        'train_losses': train_losses.copy(),
        'val_losses': val_losses.copy(),
    }, model_save_path)


def train_model(config, train_loader, val_loader, train_size, val_size,
                model_save_path, logger=None, patience=5):

    # Model, Loss, Optimizer
    model = LSTMForecast(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=(patience/2))

    # Early stopping
    patience = patience
    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    logger.info(f"Starting training for {config.epochs} epochs...") #TODO: Daniel

    # Training loop
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, train_size)
        train_losses.append(train_loss)

        val_loss = validate_epoch(model, val_loader, criterion, config.device, val_size)

        val_losses.append(val_loss)

        scheduler.step(val_loss)

        
        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Best epoch = {best_epoch}")
            save_checkpoint(model, optimizer, config, epoch, val_loss, train_losses, val_losses, model_save_path)
        else:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Best epoch = {best_epoch}")
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Save final loss curves into the checkpoint
    checkpoint = torch.load(model_save_path, weights_only=False)
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    torch.save(checkpoint, model_save_path)


    logger.success(f"Training complete. Best model saved at epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f})")

    return best_val_loss, train_losses, val_losses
