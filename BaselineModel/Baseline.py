import pathlib
import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

from BaselineEvaluate import (
    residuals_plot,
    predictions_vs_actual_plot,
    first_week_prediction_plot,
)

# Paths
PLOTS_DIR    = SCRIPT_DIR / 'BaselinePlots'
DATASET_PATH = SCRIPT_DIR / 'Files' / 'RingkøbingData.csv'

# --- Output toggles ---
SAVE_PLOTS   = True
SHOW_PLOTS   = False

# =============================================================================
# Baseline Algorithm
# =============================================================================

def load_abvaerk():
    """Load the abvaerk (energy usage) values from the CSV dataset."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["dateTime"])
    abvaerk = df["abvaerk"].interpolate(method='linear').bfill().ffill().to_numpy(dtype=np.float32)

    nan_count = np.isnan(abvaerk).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in targets")

    print(f"Loaded {len(abvaerk)} abvaerk values")
    print(f"Value range: {np.nanmin(abvaerk):.2f} to {np.nanmax(abvaerk):.2f}")

    return abvaerk


def baseline_predict(abvaerk, horizon):
    """
    Baseline model: predict t+horizon using the value at t.
    (Naive persistence forecast over the full horizon.)

    Args:
        abvaerk: Array of actual energy values
        horizon: How many hours ahead to predict (0-based: 0=1h, 23=24h, 167=168h)

    Returns:
        predictions: Array of predicted values
        actuals: Array of actual values at the predicted times
    """
    h = horizon + 1  # convert 0-based to actual step count

    # Predict value at t+h using value at t
    predictions = abvaerk[:-h]
    actuals = abvaerk[h:]

    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    # Remove pairs where either value is NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    return predictions, actuals


def evaluate_baseline():
    """Evaluate the baseline model at 1h, 24h, and 168h horizons."""
    abvaerk = load_abvaerk()

    # Evaluate at each horizon (0-based to match LSTM)
    pred_1h, act_1h = baseline_predict(abvaerk, horizon=0)
    pred_24h, act_24h = baseline_predict(abvaerk, horizon=23)
    pred_168h, act_168h = baseline_predict(abvaerk, horizon=167)

    print(f"\nValid predictions after NaN filtering:")
    print(f"  1h: {len(pred_1h)} samples")
    print(f"  24h: {len(pred_24h)} samples")
    print(f"  168h: {len(pred_168h)} samples")

    # Calculate metrics for each horizon
    print("\nBaseline Model Metrics:")
    print("-" * 40)

    for name, preds, acts in [
        ("1h", pred_1h, act_1h),
        ("24h", pred_24h, act_24h),
        ("168h", pred_168h, act_168h)
    ]:
        if len(preds) == 0:
            print(f"\n{name} Horizon: No valid predictions!")
            continue

        mse = np.mean((preds - acts) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - acts))

        # Avoid division by zero in MAPE
        non_zero_mask = acts != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((acts[non_zero_mask] - preds[non_zero_mask]) / acts[non_zero_mask])) * 100
        else:
            mape = float('nan')

        print(f"\n{name} Horizon:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    return {
        '1h': (pred_1h, act_1h),
        '24h': (pred_24h, act_24h),
        '168h': (pred_168h, act_168h)
    }

# =============================================================================
# Main
# =============================================================================

def main():
    """Run baseline model evaluation and generate plots."""

    # Evaluate baseline and print metrics
    results = evaluate_baseline()

    # Create save directory
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(exist_ok=True)

    save = lambda filename: plots_dir / filename if SAVE_PLOTS else None

    # Generate plots using the unified functions with model_name='Baseline'
    predictions_vs_actual_plot(
        model_name='Baseline',
        save_path=save('baseline_predictions_vs_actuals.png'),
        show_plots=SHOW_PLOTS
    )

    residuals_plot(
        model_name='Baseline',
        save_path=save('baseline_residuals.png'),
        show_plots=SHOW_PLOTS
    )

    first_week_prediction_plot(
        model_name='Baseline',
        save_path=save('baseline_first_week_prediction.png'),
        show_plots=SHOW_PLOTS
    )

if __name__ == '__main__':
    main()
