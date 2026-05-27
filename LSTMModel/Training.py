import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    Train the LSTM model with Adam optimizer and ReduceLROnPlateau scheduler.

    Gradient clipping is applied each step to prevent exploding gradients.

    Returns:
        train_losses : list of average training loss per epoch
        val_losses   : list of average validation loss per epoch
    """
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses   = []
    early_state = {}

    for epoch in range(epochs):
        train_loss = _run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = _run_val_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}]  "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}")

        stop, early_state = _early_stopping(val_loss, model, early_state, patience=50, min_delta=1e-5)
        if stop:
            print("Early stopping triggered.")
            model.load_state_dict(early_state["best_weights"])
            break

    return train_losses, val_losses


def evaluate_model(model, test_loader, target_scaler, device):
    """
    Run the model on the test set and report performance metrics.

    Predictions and actuals are inverse-transformed back to the original scale
    before metrics are computed.

    Returns:
        predictions : np.ndarray of predicted values in original scale
        actuals     : np.ndarray of actual values in original scale
        metrics     : dict with keys mse, rmse, mae, mape
    """
    model.eval()

    predictions = []
    actuals     = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    # Inverse transform back to original scale
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals     = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    metrics = _compute_metrics(predictions, actuals)

    print("\nTest Set Metrics:")
    print(f"  MSE  : {metrics['mse']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")

    return predictions, actuals, metrics


# ---------------------------------------------------------------------------
# Private helpers — not intended to be called directly from main.py
# ---------------------------------------------------------------------------

def _run_train_epoch(model, loader, criterion, optimizer, device):
    """Run one full training epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def _run_val_epoch(model, loader, criterion, device):
    """Run one full validation epoch and return the average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = criterion(model(X_batch), y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def _compute_metrics(predictions, actuals):
    """Compute regression metrics between predictions and actuals."""
    mse  = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


def _early_stopping(val_loss, model, state, patience, min_delta=1e-5):
    """
    Early stopping helper.

    Args:
        val_loss : current validation loss
        model    : torch model
        state    : dict storing early-stop state
        patience : epochs to wait without improvement
        min_delta: minimum improvement threshold

    Returns:
        stop_training : bool
        state         : updated state dict
    """

    # First call initialization
    if not state:
        state["best_loss"] = float("inf")
        state["epochs_no_improve"] = 0
        state["best_weights"] = None

    # Check improvement
    if val_loss < state["best_loss"] - min_delta:
        state["best_loss"] = val_loss
        state["epochs_no_improve"] = 0
        state["best_weights"] = model.state_dict()
        return False, state

    # No improvement
    state["epochs_no_improve"] += 1

    if state["epochs_no_improve"] >= patience:
        return True, state

    return False, state
