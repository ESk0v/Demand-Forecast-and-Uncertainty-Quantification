import pathlib
from matplotlib import pyplot as plt
import torch
import numpy as np
import sklearn.preprocessing._data as skdata

from LSTMModel import LSTMModel
from Data import load_dataset

# Paths relative to this script

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / 'LSTMModels' / 'lstm_model.pth'
DATASET_PATH = SCRIPT_DIR / 'Files' / 'RingkøbingData.csv'

# =============================================================================
# Main plotting functions
# =============================================================================

def residuals_plot(model_name='LSTM', save_path=None, show_plots=True):
    """
    Load model, evaluate at all horizons, and plot residuals.

    Args:
        model_name: 'LSTM' or 'Baseline'
        save_path: Path to save the plot
        show_plots: Whether to display the plot
    """
    preds_list, acts_list = _get_predictions_and_actual(model_name)
    horizons = ['1h', '24h', '168h']

    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 4), sharey=True)

    if n_horizons == 1:
        axes = [axes]

    colors = ['blue', 'green', 'red']  # match Predicted vs Actual

    for i, horizon in enumerate(horizons):
        predictions = preds_list[i]
        actuals = acts_list[i]
        residuals = predictions - actuals
        mape = _compute_mape(actuals, predictions)

        ax = axes[i]
        # --- Scatter plot for residuals ---
        ax.scatter(np.arange(len(residuals)), residuals, s=4, alpha=0.2, color=colors[i])
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)  # zero line
        ax.set_title(f'{model_name} - {horizon} Horizon Residuals', fontsize=14)
        ax.set_xlabel('Sample Index', fontsize=12)
        if i == 0:
            ax.set_ylabel('Residual (MWh)', fontsize=12)
        ax.legend([f"Residuals\nMAPE: {mape:.2f}%"], fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Residuals over time plot saved to {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def predictions_vs_actual_plot(model_name='LSTM', save_path=None, show_plots=True):
    """
    Load model, evaluate at all horizons, and plot predictions vs actuals.

    Args:
        model_name: 'LSTM' or 'Baseline'
        save_path: Path to save the plot
        show_plots: Whether to display the plot
    """
    preds_list, acts_list = _get_predictions_and_actual(model_name)
    horizons = ['1h', '24h', '168h']

    colors = ['blue', 'green', 'red']
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    fig.suptitle(f'{model_name} Model Evaluation', fontsize=16, fontweight='bold')

    for i, horizon in enumerate(horizons):
        actuals = acts_list[i]
        predictions = preds_list[i]
        mape = _compute_mape(actuals, predictions)

        # --- Scatter plot ---
        ax1 = axes[0, i]
        ax1.scatter(actuals, predictions, alpha=0.5, s=10, color=colors[i])
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')  # black dashed line
        ax1.set_xlabel('Actual Energy Usage (MWh)', fontsize=12)
        ax1.set_ylabel('Predicted Energy Usage (MWh)', fontsize=12)
        ax1.set_title(f'{horizon} Horizon: Predicted vs Actual', fontsize=14)
        ax1.legend(title=f"MAPE: {mape:.2f}%", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # --- Line plot (full series) ---
        ax2 = axes[1, i]
        ax2.plot(actuals, 'b-', label='Actual', linewidth=0.09, alpha=0.9)          # thinner + more transparent
        ax2.plot(predictions, 'r-', label='Predicted', linewidth=0.09, alpha=1)     # thinner + more transparent
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Energy Usage (MWh)', fontsize=12)
        ax2.set_title(f'{horizon} Horizon: Full Series', fontsize=14)
        ax2.legend(title=f"MAPE: {mape:.2f}%", fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Predictions vs Actuals plot saved to {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def training_validation_plot(train_losses, val_losses, save_path=None, show_plots=True):
    """
    Plot training vs validation losses (or any metric) over epochs.

    Args:
        train_losses : list or np.ndarray of training losses
        val_losses   : list or np.ndarray of validation losses
        save_path    : str or Path to save figure (optional)
        show_plots   : bool, whether to show the plot
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot training vs validation
    ax.plot(epochs, train_losses, 'b-', label='Training', linewidth=1.5, alpha=0.9)
    ax.plot(epochs, val_losses,   'r-', label='Validation', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training vs Validation plot saved to {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def first_week_prediction_plot(model_name='LSTM', save_path=None, show_plots=True):

    # --- Load dataset ---
    samples, targets, _ = load_dataset(DATASET_PATH, False)

    predictions = []
    actual_week = []

    if model_name.upper() == 'LSTM':
        # --- Load LSTM model ---
        model, feature_scaler, target_scaler = _load_model()
        model.eval()

        # Build first 168-hour forecast sequentially
        for i in range(168):
            sample_i = samples[i]      # (168, 9)
            target_i = targets[i]      # scalar

            sample_scaled = feature_scaler.transform(sample_i)
            sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = model(sample_tensor).squeeze().item()

            pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]

            predictions.append(pred)
            actual_week.append(target_i)

    elif model_name.upper() == 'BASELINE':
        # Baseline: predict each hour using the previous hour's value
        for i in range(168):
            actual_week.append(targets[i])
            if i == 0:
                # No previous hour available, use the same value
                predictions.append(targets[i])
            else:
                predictions.append(targets[i - 1])

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'LSTM' or 'Baseline'.")

    predictions = np.array(predictions)
    actual_week = np.array(actual_week)

    hours = np.arange(168)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(hours, actual_week, label='Actual', linewidth=1.5, alpha=0.9)
    ax.plot(hours, predictions, label='Predicted', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy Usage (MWh)')
    ax.set_title(f'{model_name} - First 168 Hour Forecast')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"First week prediction plot saved to {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def first_week_prediction_mc_plot(save_path=None, show_plots=True, mc_passes=50):
    """
    Predict the first 168-hour forecast with Monte Carlo Dropout uncertainty.
    
    Args:
        save_path: Path to save the plot (optional)
        show_plots: Whether to show the plot
        mc_passes: Number of stochastic forward passes for MC Dropout
    """

    # --- Load ---
    model, feature_scaler, target_scaler = _load_model()
    model.train()  # <-- keep dropout active for MC Dropout

    samples, targets, _ = load_dataset(DATASET_PATH, False)

    predictions_mean = []
    predictions_std = []
    actual_week = []

    # Build first 168-hour forecast sequentially
    for i in range(168):
        sample_i = samples[i]      # (168, 9)
        target_i = targets[i]      # scalar

        # Scale features
        sample_scaled = feature_scaler.transform(sample_i)
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(0)

        # --- Monte Carlo predictions ---
        mc_preds = []
        for _ in range(mc_passes):
            with torch.no_grad():
                pred_scaled = model(sample_tensor).squeeze().item()
            pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            mc_preds.append(pred)

        mc_preds = np.array(mc_preds)
        predictions_mean.append(mc_preds.mean())
        predictions_std.append(mc_preds.std())
        actual_week.append(target_i)

    predictions_mean = np.array(predictions_mean)
    predictions_std = np.array(predictions_std)
    actual_week = np.array(actual_week)
    hours = np.arange(168)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(hours, actual_week, label='Actual', linewidth=1.5, alpha=0.9)
    ax.plot(hours, predictions_mean, label='Predicted', linewidth=1.5, alpha=0.9)
    ax.fill_between(hours,
                    predictions_mean - 2 * predictions_std,
                    predictions_mean + 2 * predictions_std,
                    color='orange', alpha=0.3, label='Confidence Interval (2σ)')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy Usage (MWh)')
    ax.set_title('First 168 Hour Forecast with MC Dropout Uncertainty')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# Helper functions
# =============================================================================

def _compute_mape(actuals, predictions):
    """Compute MAPE, handling NaN and zero values."""
    valid_mask = ~(np.isnan(actuals) | np.isnan(predictions) | (actuals == 0))
    if valid_mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100

def _evaluate_baseline_horizon(targets, horizon):
    """
    Baseline model: predict t+horizon using the value at t.
    (Naive persistence forecast over the full horizon.)

    Args:
        targets: Array of actual energy values
        horizon: How many hours ahead (0-based: 0=1h, 23=24h, 167=168h)

    Returns:
        predictions, actuals arrays
    """
    h = horizon + 1  # convert 0-based to actual step count

    # Predict value at t+h using value at t
    predictions = targets[:-h]
    actuals = targets[h:]

    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    # Remove NaN pairs
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    return predictions, actuals

def _evaluate_horizon(model, samples, targets, feature_scaler, target_scaler, horizon=0):
    predictions = []
    actuals = []

    max_idx = len(samples) - horizon

    model.eval()  # important

    for i in range(max_idx):
        sample = samples[i]
        actual = targets[i + horizon]

        # --- Normalize input ---
        n_features = sample.shape[1]
        sample_flat = sample.reshape(-1, n_features)
        sample_scaled = feature_scaler.transform(sample_flat)
        sample_scaled = sample_scaled.reshape(1, 168, n_features)

        sample_tensor = torch.FloatTensor(sample_scaled)

        # --- Predict ---
        with torch.no_grad():
            prediction_scaled = model(sample_tensor).numpy().flatten()[0]

        prediction = target_scaler.inverse_transform(
            [[prediction_scaled]]
        )[0][0]

        predictions.append(prediction)
        actuals.append(actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    return predictions, actuals

def _get_predictions_and_actual(model_name='LSTM'):
    """
    Get predictions and actuals for all horizons based on model type.

    Args:
        model_name: 'LSTM' or 'Baseline'

    Returns:
        Tuple of (preds_list, acts_list) for horizons [1h, 24h, 168h]
    """
    # Load dataset (needed for both models)
    samples, targets, _ = load_dataset(DATASET_PATH, False)

    if model_name.upper() == 'LSTM':
        # Load LSTM model and evaluate
        model, feature_scaler, target_scaler = _load_model(False)

        pred_1h, act_1h = _evaluate_horizon(model, samples, targets, feature_scaler, target_scaler, horizon=0)
        pred_24h, act_24h = _evaluate_horizon(model, samples, targets, feature_scaler, target_scaler, horizon=23)
        pred_168h, act_168h = _evaluate_horizon(model, samples, targets, feature_scaler, target_scaler, horizon=167)

    elif model_name.upper() == 'BASELINE':
        # Baseline: predict using previous hour's value
        pred_1h, act_1h = _evaluate_baseline_horizon(targets, horizon=0)
        pred_24h, act_24h = _evaluate_baseline_horizon(targets, horizon=23)
        pred_168h, act_168h = _evaluate_baseline_horizon(targets, horizon=167)

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'LSTM' or 'Baseline'.")

    return [pred_1h, pred_24h, pred_168h], [act_1h, act_24h, act_168h]

def _load_model(verbose=True):

    try:
        with torch.serialization.safe_globals([skdata.StandardScaler]):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    except Exception:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    input_size  = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers  = checkpoint['num_layers']

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feature_scaler = checkpoint['feature_scaler']
    target_scaler  = checkpoint['target_scaler']

    if verbose:
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Num layers: {num_layers}")

    return model, feature_scaler, target_scaler

def _save_plot(save_path, show_plots):
    """Save the current figure if a path is given, then display it."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show_plots: plt.show()