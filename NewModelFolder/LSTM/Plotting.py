"""
Plotting — Evaluation plots for the trained LSTM model.
=========================================================
Loads the saved checkpoint and dataset, rebuilds the model, runs inference
on the validation and test sets, and generates all evaluation plots.

Can be run independently after training is complete:
    python3 Plotting.py --local
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# LSTMModel.py lives in the parent directory (NewModelFolder/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config, LSTMForecast
from LSTM.GenerateREADME import generate_evaluation_readme

# Dataset starts at 2023-01-01 01:00, one row per hour, no gaps
DATASET_START   = pd.Timestamp("2023-01-01 01:00")
ENCODER_HISTORY = 168  # must match DatasetCreation.py

DAY_HOURS  = [24, 48, 72, 96, 120, 144, 168]
DAY_LABELS = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']
# Fixed y-range for residual diagnostics panel C (variance ratio)
VAR_RATIO_YLIM = (0.0, 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_model(model_dir):
    """Scan model_dir for the highest model_vN folder and return its .pth path."""
    existing = [f for f in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, f)) and f.startswith("model_v")]
    versions = []
    for f in existing:
        try:
            versions.append(int(f.replace("model_v", "")))
        except ValueError:
            pass
    if not versions:
        raise FileNotFoundError(f"No versioned run folders found in {model_dir}")
    latest_version = max(versions)
    run_folder = os.path.join(model_dir, f"model_v{latest_version}")
    return os.path.join(run_folder, f"model_v{latest_version}.pth")


def _add_day_markers(ax):
    """Add vertical day-boundary lines and labels to an axes."""
    for h, lbl in zip(DAY_HOURS, DAY_LABELS):
        ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.annotate(lbl, xy=(h, 1.02), xycoords=('data', 'axes fraction'),
                    ha='center', fontsize=8, color='gray')


def _date_label_for(idx, test_start_global_idx):
    """Return a human-readable date range string for a test-set sample index."""
    global_idx   = test_start_global_idx + idx
    window_start = DATASET_START + pd.Timedelta(hours=global_idx + ENCODER_HISTORY)
    window_end   = window_start + pd.Timedelta(hours=167)
    return (f"{window_start.strftime('%Y-%m-%d %H:%M')} "
            f"→ {window_end.strftime('%Y-%m-%d %H:%M')}"), window_start


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────


def plot_forecast_windows(preds_h, targets_h, test_start_global_idx, save_path):
    """Plot 2 — Up to three example 168-hour forecast windows from the test set.

    When only one sample is provided a single panel is shown instead of three.
    """
    n_test_samples = preds_h.shape[0]
    n_panels       = min(n_test_samples, 3)
    sample_indices = np.linspace(0, n_test_samples - 1, n_panels, dtype=int)
    hours          = np.arange(1, 169)

    fig, axes = plt.subplots(1, n_panels, figsize=(max(8, 22 // 3 * n_panels), 6),
                             squeeze=False)
    axes = axes[0]   # shape (n_panels,)

    for ax, idx in zip(axes, sample_indices):
        label, _ = _date_label_for(idx, test_start_global_idx)
        window_mae = np.mean(np.abs(targets_h[idx] - preds_h[idx]))
        ax.plot(hours, targets_h[idx], label='Actual',    linewidth=1.5, color='blue')
        ax.plot(hours, preds_h[idx],   label='Predicted', linewidth=1.5, color='red', alpha=0.8)
        ax.set_title(f"{label}\n(MAE: {window_mae:.4f})", fontsize=9)
        ax.set_xlabel("Forecast Hour", fontsize=9)
        ax.set_ylabel("abvaerk (MWh)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Example 168-Hour Forecast Windows (Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(preds_h, targets_h, save_path):
    """Plot 3 — Actual vs Predicted scatter at three forecast horizons."""
    rng = np.random.default_rng(42)

    # Fixed axes for all panels (and across model runs on same dataset):
    # derive from actual test-set values only so model output spread doesn't
    # change the visual scale.
    a_min = float(np.min(targets_h))
    a_max = float(np.max(targets_h))
    pad   = 0.03 * (a_max - a_min if a_max > a_min else 1.0)
    fixed_lims = [a_min - pad, a_max + pad]

    def _scatter_panel(ax, horizon_idx, horizon_label):
        actual = targets_h[:, horizon_idx]
        pred   = preds_h[:, horizon_idx]
        jitter = rng.normal(0, 0.02, size=actual.shape)
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2     = 1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10)
        ax.scatter(actual + jitter, pred, alpha=0.3, s=4, color='steelblue')
        ax.plot(fixed_lims, fixed_lims, 'k--', linewidth=1.0, label='y = x (perfect)')
        ax.set_xlim(fixed_lims)
        ax.set_ylim(fixed_lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Actual abvaerk (MWh)", fontsize=11)
        ax.set_ylabel("Predicted abvaerk (MWh)", fontsize=11)
        ax.set_title(f"Actual vs Predicted — {horizon_label}\nR² = {r2:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axes_sc = plt.subplots(1, 3, figsize=(22, 7))
    _scatter_panel(axes_sc[0], horizon_idx=0,   horizon_label="1h Ahead (horizon 0)")
    _scatter_panel(axes_sc[1], horizon_idx=23,  horizon_label="24h Ahead (horizon 23)")
    _scatter_panel(axes_sc[2], horizon_idx=167, horizon_label="168h Ahead (horizon 167)")

    fig.suptitle("Actual vs Predicted at Three Forecast Horizons (Test Set)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residual_diagnostics(preds_h, targets_h, test_start_global_idx, save_path):
    """Plot 4 — Residual diagnostics: MAE over time, quantile heatmap, variance ratio.

    Panel A — MAE per window over time.
      Reveals non-stationarity: error spikes at winter peaks, season
      transitions, or holidays that the model failed to generalise.

    Panel B — Quantile error heatmap (horizon × percentile).
      Shows the full distribution of absolute errors at every forecast
      step, not just the mean. Reveals whether tail errors grow faster
      than the median at longer horizons — critical for planning.

    Panel C — Predicted vs actual variance ratio per horizon.
      Ratio near 1.0 means the model's spread matches reality.
      Ratio < 1 → under-dispersed (mean regression).
      Ratio > 1 → over-dispersed.
    """
    n_test_samples = preds_h.shape[0]

    mae_per_window = np.mean(np.abs(targets_h - preds_h), axis=1)
    window_dates   = [DATASET_START + pd.Timedelta(hours=(test_start_global_idx + i + ENCODER_HISTORY))
                      for i in range(n_test_samples)]

    abs_errors  = np.abs(targets_h - preds_h)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    heatmap     = np.percentile(abs_errors, percentiles, axis=0)

    pred_var   = np.var(preds_h,   axis=0)
    actual_var = np.var(targets_h, axis=0)
    var_ratio  = pred_var / np.where(actual_var == 0, 1e-10, actual_var)

    fig = plt.figure(figsize=(18, 14))
    gs_r = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                             height_ratios=[1, 1.2])

    # Panel A — residuals over time (spans full top row)
    ax_a = fig.add_subplot(gs_r[0, :])
    ax_a.plot(window_dates, mae_per_window, linewidth=0.8, color='steelblue', alpha=0.7)
    rolling = pd.Series(mae_per_window).rolling(window=168, min_periods=1).mean().values
    ax_a.plot(window_dates, rolling, linewidth=1.5, color='red', label='168-sample rolling mean')
    ax_a.set_xlabel("Forecast Start Date", fontsize=11)
    ax_a.set_ylabel("MAE (MWh)", fontsize=11)
    ax_a.set_title("Forecast Error Over Time — MAE per Window (Test Set)", fontsize=12)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Panel B — quantile heatmap (bottom-left)
    ax_b = fig.add_subplot(gs_r[1, 0])
    im = ax_b.imshow(heatmap, aspect='auto', origin='lower',
                     extent=[1, 168, -0.5, len(percentiles) - 0.5],
                     cmap='YlOrRd')
    ax_b.set_yticks(range(len(percentiles)))
    ax_b.set_yticklabels([f"p{p}" for p in percentiles], fontsize=9)
    ax_b.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax_b.set_ylabel("Error Percentile", fontsize=11)
    ax_b.set_title("Absolute Error Distribution by Horizon\n(quantile heatmap)", fontsize=12)
    for h in DAY_HOURS:
        ax_b.axvline(x=h, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.colorbar(im, ax=ax_b, label='Absolute Error (MWh)')

    # Panel C — variance ratio (bottom-right)
    ax_c = fig.add_subplot(gs_r[1, 1])
    ax_c.plot(range(1, 169), var_ratio, color='darkorange', linewidth=1.4)
    ax_c.axhline(1.0, color='black', linestyle='--', linewidth=0.9,
                 label='ratio = 1 (perfect dispersion)')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio < 1.0), alpha=0.2, color='steelblue',
                      label='under-dispersed (mean regression)')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio > 1.0), alpha=0.2, color='red',
                      label='over-dispersed')
    for h in DAY_HOURS:
        ax_c.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_c.set_ylim(*VAR_RATIO_YLIM)
    ax_c.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax_c.set_ylabel("Var(predicted) / Var(actual)", fontsize=11)
    ax_c.set_title("Predicted vs Actual Variance Ratio\nper Horizon", fontsize=12)
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    fig.suptitle("Residual Diagnostics (Test Set)", fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_horizon_metrics(preds_h, targets_h, encoder_data, train_size, val_size,
                             demand_mean, demand_std, save_path):
    """Plot 5 — Per-horizon RMSE/MAE/R² vs persistence baseline.

    The persistence baseline repeats the last known encoder abvaerk value for
    all 168 future horizons.  If the LSTM doesn't beat it, it adds no value.
    """
    mse_per_horizon  = np.mean((preds_h - targets_h) ** 2, axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)
    mae_per_horizon  = np.mean(np.abs(preds_h - targets_h), axis=0)

    ss_res_h = np.sum((targets_h - preds_h) ** 2, axis=0)
    ss_tot_h = np.sum((targets_h - targets_h.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_horizon = 1 - ss_res_h / np.where(ss_tot_h == 0, 1e-10, ss_tot_h)

    # Persistence baseline — last encoder abvaerk value (col 0), rescaled
    test_encoder = encoder_data[train_size + val_size:]
    last_known   = test_encoder[:, -1, 0].detach().cpu().numpy() * demand_std + demand_mean
    persist_pred = np.tile(last_known[:, None], (1, 168))

    persist_mse  = np.mean((persist_pred - targets_h) ** 2, axis=0)
    persist_rmse = np.sqrt(persist_mse)
    persist_mae  = np.mean(np.abs(persist_pred - targets_h), axis=0)
    persist_r2   = 1 - np.sum((targets_h - persist_pred) ** 2, axis=0) / np.where(ss_tot_h == 0, 1e-10, ss_tot_h)

    fig, axes_h = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    hours_r = range(1, 169)

    metrics = [
        (axes_h[0], rmse_per_horizon, persist_rmse, 'RMSE (MWh)', 'purple'),
        (axes_h[1], mae_per_horizon,  persist_mae,  'MAE (MWh)',  'red'),
        (axes_h[2], r2_per_horizon,   persist_r2,   'R²',         'darkorange'),
    ]
    for ax, lstm_vals, pers_vals, ylabel, color in metrics:
        ax.plot(hours_r, lstm_vals, color=color,  linewidth=1.2, label='LSTM')
        ax.plot(hours_r, pers_vals, color='gray', linewidth=1.0, linestyle='--',
                label='Persistence baseline')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _add_day_markers(ax)

    # R² = 0 means no better than always predicting the mean
    axes_h[2].axhline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6,
                      label='R² = 0 (no better than mean)')
    axes_h[2].legend(fontsize=9)

    axes_h[0].set_title("Per-Horizon Forecast Error vs Persistence Baseline (Test Set)", fontsize=14)
    axes_h[2].set_xlabel("Forecast Horizon (hours)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(filePaths=None, logger=None, run_dir=None):
    """
    Generate evaluation plots from a trained model checkpoint.
    The train/val loss plot is handled by LSTMTraining.py during training.

    Args:
        filePaths : [dataset_path, model_path]
        logger    : Logger instance (optional).
        run_dir   : Directory under which a Plots/ sub-folder is created.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    model_path   = filePaths[1]
    dataset_path = filePaths[0]
    plot_dir     = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    test_plot_path      = os.path.join(plot_dir, "test_predictions.png")
    scatter_plot_path   = os.path.join(plot_dir, "actual_vs_predicted.png")
    residuals_plot_path = os.path.join(plot_dir, "residuals.png")
    horizon_plot_path   = os.path.join(plot_dir, "per_horizon_metrics.png")

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    best_epoch = checkpoint['epoch']
    if logger:
        logger.info(f"Checkpoint loaded (best epoch: {best_epoch}, val_loss: {checkpoint['val_loss']:.4f})")

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset      = torch.load(dataset_path, weights_only=False)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    val_ratio, test_ratio = 0.1, 0.1
    n_total    = len(full_dataset)
    test_size  = int(n_total * test_ratio)
    val_size   = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))
    config       = Config()
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------
    # Rebuild model and run inference
    # -----------------------------
    model = LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds_h, all_targets_h = [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            mu, log_var = model(enc, dec)
            all_preds_h.append(mu.cpu().numpy())
            all_targets_h.append(tgt.numpy())

    preds_h   = np.concatenate(all_preds_h,   axis=0)
    targets_h = np.concatenate(all_targets_h, axis=0)

    # -----------------------------
    # Rescale to raw MWh
    # -----------------------------
    if "demand_mean" in dataset and "demand_std" in dataset:
        demand_mean = float(dataset["demand_mean"])
        demand_std  = float(dataset["demand_std"])
    else:
        all_targets = target_data.detach().cpu().numpy()
        demand_mean = float(all_targets.mean())
        demand_std  = float(all_targets.std())

    preds_h   = preds_h   * demand_std + demand_mean
    targets_h = targets_h * demand_std + demand_mean

    test_start_global_idx = train_size + val_size

    # -----------------------------
    # Call each plot function
    # -----------------------------

    plot_forecast_windows(preds_h, targets_h, test_start_global_idx, test_plot_path)
    if logger:
        logger.success("Saved: forecast windows plot")

    plot_actual_vs_predicted(preds_h, targets_h, scatter_plot_path)
    if logger:
        logger.success("Saved: actual vs predicted scatter plot")

    plot_residual_diagnostics(preds_h, targets_h, test_start_global_idx, residuals_plot_path)
    if logger:
        logger.success("Saved: residual diagnostics plot")

    plot_per_horizon_metrics(preds_h, targets_h, encoder_data, train_size, val_size,
                             demand_mean, demand_std, horizon_plot_path)
    if logger:
        logger.success("Saved: per-horizon metrics plot")

    # -----------------------------
    # Generate README
    # -----------------------------
    generate_evaluation_readme(plot_dir, best_epoch, checkpoint['val_loss'], preds_h.shape[0],
                               train_size, val_size, test_size, n_total,
                               model_filename=os.path.basename(model_path))
