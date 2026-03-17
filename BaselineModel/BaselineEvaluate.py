"""
BaselineEvaluate.py — Plotting and evaluation functions for the Baseline model.

These are self-contained copies of the functions originally in
LSTMModel/EvaluateModel.py, stripped of all LSTM-specific logic so that
the BaselineModel folder has no dependencies outside itself.
"""

import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / 'Files' / 'RingkøbingData.csv'


# =============================================================================
# Private helpers
# =============================================================================

def _load_abvaerk():
    """Load the raw abvaerk column from the CSV dataset."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["dateTime"])
    abvaerk = df["abvaerk"].interpolate(method='linear').bfill().ffill().to_numpy(dtype=np.float32)
    return abvaerk


def _compute_mape(actuals, predictions):
    """Compute MAPE, handling NaN and zero values."""
    valid_mask = ~(np.isnan(actuals) | np.isnan(predictions) | (actuals == 0))
    if valid_mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100


def _evaluate_baseline_horizon(targets, horizon):
    """
    Baseline persistence forecast: predict t+horizon using the value at t.

    Args:
        targets : 1-D array of actual energy values
        horizon : 0-based horizon (0 = 1 h ahead, 23 = 24 h ahead, 167 = 168 h ahead)

    Returns:
        predictions, actuals — matched arrays with NaN pairs removed
    """
    h = horizon + 1

    predictions = targets[:-h]
    actuals     = targets[h:]

    min_len     = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals     = actuals[:min_len]

    valid_mask  = ~(np.isnan(predictions) | np.isnan(actuals))
    return predictions[valid_mask], actuals[valid_mask]


def _get_predictions_and_actuals():
    """Return (preds_list, acts_list) for the three standard horizons [1h, 24h, 168h]."""
    targets = _load_abvaerk()

    pred_1h,   act_1h   = _evaluate_baseline_horizon(targets, horizon=0)
    pred_24h,  act_24h  = _evaluate_baseline_horizon(targets, horizon=23)
    pred_168h, act_168h = _evaluate_baseline_horizon(targets, horizon=167)

    return [pred_1h, pred_24h, pred_168h], [act_1h, act_24h, act_168h]


# =============================================================================
# Public plot functions  (API-compatible with EvaluateModel.py)
# =============================================================================

def residuals_plot(model_name='Baseline', save_path=None, show_plots=True):
    """
    Evaluate at all horizons and plot residuals.

    Args:
        model_name : label shown in plot titles
        save_path  : path to save the figure (optional)
        show_plots : whether to display the figure
    """
    preds_list, acts_list = _get_predictions_and_actuals()
    horizons = ['1h', '24h', '168h']
    colors   = ['blue', 'green', 'red']

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    for i, horizon in enumerate(horizons):
        predictions = preds_list[i]
        actuals     = acts_list[i]
        residuals   = predictions - actuals
        mape        = _compute_mape(actuals, predictions)

        ax = axes[i]
        ax.scatter(np.arange(len(residuals)), residuals, s=4, alpha=0.2, color=colors[i])
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(f'{model_name} - {horizon} Horizon Residuals', fontsize=14)
        ax.set_xlabel('Sample Index', fontsize=12)
        if i == 0:
            ax.set_ylabel('Residual (MWh)', fontsize=12)
        ax.legend([f"Residuals\nMAPE: {mape:.2f}%"], fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Residuals plot saved to {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def predictions_vs_actual_plot(model_name='Baseline', save_path=None, show_plots=True):
    """
    Evaluate at all horizons and plot predictions vs actuals.

    Args:
        model_name : label shown in plot titles
        save_path  : path to save the figure (optional)
        show_plots : whether to display the figure
    """
    preds_list, acts_list = _get_predictions_and_actuals()
    horizons = ['1h', '24h', '168h']
    colors   = ['blue', 'green', 'red']

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'{model_name} Model Evaluation', fontsize=16, fontweight='bold')

    for i, horizon in enumerate(horizons):
        actuals     = acts_list[i]
        predictions = preds_list[i]
        mape        = _compute_mape(actuals, predictions)

        # Scatter
        ax1 = axes[0, i]
        ax1.scatter(actuals, predictions, alpha=0.5, s=10, color=colors[i])
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
                 label='Perfect Prediction')
        ax1.set_xlabel('Actual Energy Usage (MWh)', fontsize=12)
        ax1.set_ylabel('Predicted Energy Usage (MWh)', fontsize=12)
        ax1.set_title(f'{horizon} Horizon: Predicted vs Actual', fontsize=14)
        ax1.legend(title=f"MAPE: {mape:.2f}%", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Full series line
        ax2 = axes[1, i]
        ax2.plot(actuals,     'b-', label='Actual',    linewidth=0.09, alpha=0.9)
        ax2.plot(predictions, 'r-', label='Predicted', linewidth=0.09, alpha=1.0)
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


def first_week_prediction_plot(model_name='Baseline', save_path=None, show_plots=True):
    """
    Plot the first 168-hour persistence forecast against actuals.

    Args:
        model_name : label shown in plot titles
        save_path  : path to save the figure (optional)
        show_plots : whether to display the figure
    """
    targets = _load_abvaerk()

    actual_week = []
    predictions = []

    for i in range(168):
        actual_week.append(targets[i])
        predictions.append(targets[i] if i == 0 else targets[i - 1])

    predictions = np.array(predictions)
    actual_week = np.array(actual_week)
    hours       = np.arange(168)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, actual_week, label='Actual',    linewidth=1.5, alpha=0.9)
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

