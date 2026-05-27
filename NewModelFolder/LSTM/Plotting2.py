"""
Plotting — Evaluation plots for the trained LSTM model.
=========================================================
Loads the saved checkpoint and dataset, rebuilds the model, runs inference
on the test set, and generates all evaluation plots.

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config, LSTMForecast
from LSTM.GenerateREADME import generate_evaluation_readme

DATASET_START   = pd.Timestamp("2023-01-01 01:00")
ENCODER_HISTORY = 168

DAY_HOURS  = [24, 48, 72, 96, 120, 144, 168]
DAY_LABELS = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']
VAR_RATIO_YLIM = (0.0, 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_model(model_dir):
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
    for h, lbl in zip(DAY_HOURS, DAY_LABELS):
        ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.annotate(lbl, xy=(h, 1.02), xycoords=('data', 'axes fraction'),
                    ha='center', fontsize=8, color='gray')


def _date_label_for(idx, test_start_global_idx):
    global_idx   = test_start_global_idx + idx
    window_start = DATASET_START + pd.Timedelta(hours=global_idx + ENCODER_HISTORY)
    window_end   = window_start + pd.Timedelta(hours=167)
    return (f"{window_start.strftime('%Y-%m-%d %H:%M')} "
            f"→ {window_end.strftime('%Y-%m-%d %H:%M')}"), window_start


def _get_split_indices(n_total, val_ratio=0.20, cal_ratio=0.10, test_ratio=0.10):
    """
    Reproduce the exact same split used in LSTMTraining.py.

    val and cal both cover the full valcal pool (they overlap intentionally —
    neither set touches model weights so there is no leakage).

    Returns (train_size, valcal_size, valcal_size, test_size)
    where valcal_size is used for both val and cal index ranges.
    """
    test_size   = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size  = n_total - valcal_size - test_size
    return train_size, valcal_size, valcal_size, test_size


def _run_inference(model, dataset, config, batch_size=None):
    """
    Run full inference over a Subset/Dataset.
    Returns (q10, q50, q90, targets) as numpy arrays, still normalised.
    """
    bs     = batch_size or config.batch_size
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    all_q10, all_q50, all_q90, all_tgt = [], [], [], []
    model.eval()
    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            q10, q50, q90 = model(enc, dec)
            all_q10.append(q10.cpu().numpy())
            all_q50.append(q50.cpu().numpy())
            all_q90.append(q90.cpu().numpy())
            all_tgt.append(tgt.numpy())
    return (np.concatenate(all_q10),
            np.concatenate(all_q50),
            np.concatenate(all_q90),
            np.concatenate(all_tgt))


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_windows(q10, q50, q90, targets_h, test_start_global_idx, save_path, u_alpha=None):
    """
    Plot example forecast windows (raw median + intervals), optionally with per-horizon conformal calibration.
    Plots all examples stacked vertically for more width per plot.
    Prints the total width for each horizon after calibration (sanity check).
    """
    n_test_samples = q50.shape[0]
    n_panels = min(n_test_samples, 3)
    sample_indices = np.linspace(0, n_test_samples - 1, n_panels, dtype=int)
    hours = np.arange(1, 169)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), squeeze=False)
    axes = axes[:, 0]

    for ax_idx, (ax, idx) in enumerate(zip(axes, sample_indices)):
        label, _ = _date_label_for(idx, test_start_global_idx)
        window_mae = np.mean(np.abs(targets_h[idx] - q50[idx]))

        ax.plot(hours, targets_h[idx], label='Actual', linewidth=1.5, color='blue')
        ax.plot(hours, q50[idx], label='Median (q50)', linewidth=1.5, color='red', alpha=0.8)
        ax.fill_between(hours, q10[idx], q90[idx], color='red', alpha=0.2, label='Raw q10-q90 interval')

        if u_alpha is not None:
            u_alpha_1d = np.array(u_alpha).flatten()
            q10_cal = q10[idx] - u_alpha_1d
            q90_cal = q90[idx] + u_alpha_1d
            ax.fill_between(hours, q10_cal, q90_cal, color='green', alpha=0.2,
                            label='Per-horizon calibrated interval')
            interval_width = q90_cal - q10_cal
            ax.plot(hours, interval_width, color='black', linestyle=':', linewidth=1.2,
                    label='Calibrated interval width')

            if ax_idx == 0:
                total_width = (q90[idx] - q10[idx]) + 2 * u_alpha_1d
                print("\n--- Sanity Check: Raw width, alpha, total width ---")
                print("Hour | Raw Width | Alpha | Total Width")
                for h, (rw, a, tw) in enumerate(zip(q90[idx] - q10[idx], u_alpha_1d, total_width), start=1):
                    print(f"{h:3d} | {rw:.4f} | {a:.4f} | {tw:.4f}")

        ax.set_title(f"{label}\n(MAE: {window_mae:.4f})", fontsize=10)
        ax.set_xlabel("Forecast Hour", fontsize=10)
        ax.set_ylabel("abvaerk (MWh)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Example 168-Hour Forecast Windows (Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(preds_h, targets_h, encoder_data, train_size, val_size, cal_size,
                             demand_mean, demand_std, save_path):
    rng = np.random.default_rng(42)

    a_min = float(np.min(targets_h))
    a_max = float(np.max(targets_h))
    pad   = 0.03 * (a_max - a_min if a_max > a_min else 1.0)
    fixed_lims = [a_min - pad, a_max + pad]

    test_encoder = encoder_data[train_size + val_size + cal_size:]
    last_known   = test_encoder[:, -1, 0].detach().cpu().numpy() * demand_std + demand_mean
    persist_pred = np.tile(last_known[:, None], (1, 168))

    def _scatter_panel(ax, actual, pred, horizon_label, model_name, color):
        jitter = rng.normal(0, 0.02, size=actual.shape)
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2     = 1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10)
        ax.scatter(actual + jitter, pred, alpha=0.3, s=4, color=color)
        ax.plot(fixed_lims, fixed_lims, 'k--', linewidth=1.0, label='y = x (perfect)')
        ax.set_xlim(fixed_lims)
        ax.set_ylim(fixed_lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Actual abvaerk (MWh)", fontsize=11)
        ax.set_ylabel("Predicted abvaerk (MWh)", fontsize=11)
        ax.set_title(f"{model_name} — {horizon_label}\nR² = {r2:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    horizons = [
        (0,   "1h Ahead (horizon 0)"),
        (23,  "24h Ahead (horizon 23)"),
        (167, "168h Ahead (horizon 167)"),
    ]
    for col, (h_idx, h_label) in enumerate(horizons):
        _scatter_panel(axes[0, col], targets_h[:, h_idx], preds_h[:, h_idx],
                       horizon_label=h_label, model_name="LSTM", color='steelblue')
    for col, (h_idx, h_label) in enumerate(horizons):
        _scatter_panel(axes[1, col], targets_h[:, h_idx], persist_pred[:, h_idx],
                       horizon_label=h_label, model_name="Persistence Baseline", color='coral')

    fig.suptitle("Actual vs Predicted at Three Forecast Horizons (Test Set)\nLSTM vs Persistence Baseline",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residual_diagnostics(preds_h, targets_h, test_start_global_idx, save_path):
    n_test_samples = preds_h.shape[0]

    mae_per_window = np.mean(np.abs(targets_h - preds_h), axis=1)
    window_dates = [DATASET_START + pd.Timedelta(hours=(test_start_global_idx + i + ENCODER_HISTORY))
                    for i in range(n_test_samples)]

    abs_errors  = np.abs(targets_h - preds_h)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    heatmap     = np.percentile(abs_errors, percentiles, axis=0)

    pred_var   = np.var(preds_h, axis=0)
    actual_var = np.var(targets_h, axis=0)
    var_ratio  = pred_var / np.where(actual_var == 0, 1e-10, actual_var)

    fig = plt.figure(figsize=(18, 14))
    gs_r = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2])

    ax_a = fig.add_subplot(gs_r[0, :])
    ax_a.plot(window_dates, mae_per_window, linewidth=0.8, color='steelblue', alpha=0.7)
    rolling = pd.Series(mae_per_window).rolling(window=168, min_periods=1).mean().values
    ax_a.plot(window_dates, rolling, linewidth=1.5, color='red', label='168-sample rolling mean')
    ax_a.set_xlabel("Forecast Start Date", fontsize=11)
    ax_a.set_ylabel("MAE (MWh)", fontsize=11)
    ax_a.set_title("Forecast Error Over Time — MAE per Window (Test Set)", fontsize=12)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    ax_b = fig.add_subplot(gs_r[1, 0])
    im = ax_b.imshow(heatmap, aspect='auto', origin='lower',
                     extent=[1, 168, -0.5, len(percentiles)-0.5], cmap='YlOrRd')
    ax_b.set_yticks(range(len(percentiles)))
    ax_b.set_yticklabels([f"p{p}" for p in percentiles], fontsize=9)
    ax_b.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax_b.set_ylabel("Error Percentile", fontsize=11)
    ax_b.set_title("Absolute Error Distribution by Horizon\n(quantile heatmap)", fontsize=12)
    for h in DAY_HOURS:
        ax_b.axvline(x=h, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.colorbar(im, ax=ax_b, label='Absolute Error (MWh)')

    ax_c = fig.add_subplot(gs_r[1, 1])
    ax_c.plot(range(1, 169), var_ratio, color='darkorange', linewidth=1.4)
    ax_c.axhline(1.0, color='black', linestyle='--', linewidth=0.9, label='ratio = 1 (perfect dispersion)')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio < 1.0), alpha=0.2, color='steelblue', label='under-dispersed')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio > 1.0), alpha=0.2, color='red', label='over-dispersed')
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


def plot_per_horizon_metrics(preds_h, targets_h, encoder_data, train_size, val_size, cal_size,
                             demand_mean, demand_std, save_path):
    mse_per_horizon  = np.mean((preds_h - targets_h) ** 2, axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)
    mae_per_horizon  = np.mean(np.abs(preds_h - targets_h), axis=0)

    ss_res_h = np.sum((targets_h - preds_h) ** 2, axis=0)
    ss_tot_h = np.sum((targets_h - targets_h.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_horizon = 1 - ss_res_h / np.where(ss_tot_h == 0, 1e-10, ss_tot_h)

    test_encoder = encoder_data[train_size + val_size + cal_size:]
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

    axes_h[2].axhline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6,
                      label='R² = 0 (no better than mean)')
    axes_h[2].legend(fontsize=9)
    axes_h[0].set_title("Per-Horizon Forecast Error vs Persistence Baseline (Test Set)", fontsize=14)
    axes_h[2].set_xlabel("Forecast Horizon (hours)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_quantile_coverage(q10_h, q90_h, targets_h, u_alpha, save_path,
                           q10_test=None, q90_test=None, targets_test=None):
    """
    Coverage per horizon for:
      - Raw q10-q90 interval (cal set, no calibration)
      - Conformal-calibrated interval on cal set  (honest fit check)
      - Conformal-calibrated interval on test set (honest generalization check)

    The gap between the cal calibrated line and the test calibrated line tells
    you how well the conformal thresholds transfer to unseen data.

    Args:
        q10_h, q90_h, targets_h : cal set arrays, shape (N_cal, 168), in MWh
        u_alpha                  : per-horizon calibration offsets, shape (168,), in MWh
        save_path                : output path
        q10_test, q90_test,
        targets_test             : test set arrays, shape (N_test, 168), in MWh — optional
    """
    hours   = range(1, 169)
    u_alpha = np.array(u_alpha).reshape(1, -1)   # (1, 168) for broadcasting

    # ── Raw coverage (cal set, no calibration applied) ────────────────────────
    raw_coverage = ((targets_h >= q10_h) & (targets_h <= q90_h)).mean(axis=0) * 100

    # ── Calibrated coverage on cal set ────────────────────────────────────────
    q10_cal_cal = q10_h - u_alpha
    q90_cal_cal = q90_h + u_alpha
    cal_coverage = ((targets_h >= q10_cal_cal) & (targets_h <= q90_cal_cal)).mean(axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(hours, raw_coverage,
            color='steelblue', linewidth=1.5,
            label='Raw q10–q90 coverage (cal set, uncalibrated)')

    ax.plot(hours, cal_coverage,
            color='green', linewidth=1.5,
            label='Calibrated coverage — cal set (fits own data)')

    # ── Calibrated coverage on test set (honest generalization) ───────────────
    if q10_test is not None and q90_test is not None and targets_test is not None:
        q10_cal_test = q10_test - u_alpha
        q90_cal_test = q90_test + u_alpha
        test_coverage = ((targets_test >= q10_cal_test) & (targets_test <= q90_cal_test)).mean(axis=0) * 100

        ax.plot(hours, test_coverage,
                color='crimson', linewidth=1.5, linestyle='--',
                label='Calibrated coverage — test set (generalization)')

    ax.axhline(90, color='orange', linestyle='--', linewidth=1.0,
               label='Nominal 90% (calibrated target)')
    ax.axhline(80, color='black',  linestyle='--', linewidth=1.0,
               label='Nominal 80% (raw interval target)')

    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title(
        "Prediction Interval Coverage per Horizon\n"
        "Raw  |  Calibrated on Cal Set  |  Calibrated on Test Set",
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for h in DAY_HOURS:
        ax.axvline(x=h, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pinball_loss(q10_h, q50_h, q90_h, targets_h, save_path):
    def pinball(y_true, y_pred, q):
        diff = y_true - y_pred
        return np.mean(np.maximum(q * diff, (q - 1) * diff), axis=0)

    hours   = range(1, 169)
    pin_q10 = pinball(targets_h, q10_h, 0.1)
    pin_q50 = pinball(targets_h, q50_h, 0.5)
    pin_q90 = pinball(targets_h, q90_h, 0.9)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hours, pin_q10, label='q10', color='red')
    ax.plot(hours, pin_q50, label='q50', color='blue')
    ax.plot(hours, pin_q90, label='q90', color='green')
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Pinball Loss", fontsize=12)
    ax.set_title("Per-Horizon Pinball Loss (Quantile Forecast)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for h in DAY_HOURS:
        ax.axvline(x=h, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_lstm_weather_impact(q10, q50, q90,
                             q10_zero, q50_zero, q90_zero,
                             targets_h,
                             test_start_global_idx, save_path,
                             u_alpha=None, idx=0):
    hours = np.arange(1, 169)
    label, _ = _date_label_for(idx, test_start_global_idx)
    mae_normal = np.mean(np.abs(targets_h[idx] - q50[idx]))
    mae_zero   = np.mean(np.abs(targets_h[idx] - q50_zero[idx]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(hours, targets_h[idx], label='Actual', linewidth=2)
    ax.plot(hours, q50[idx], label='LSTM Median (q50)', linewidth=2)
    ax.fill_between(hours, q10[idx], q90[idx], alpha=0.2, label='Raw q10–q90 interval')
    if u_alpha is not None:
        u_alpha_1d = np.array(u_alpha).flatten()
        ax.fill_between(hours, q10[idx] - u_alpha_1d, q90[idx] + u_alpha_1d,
                        alpha=0.2, label='Calibrated interval')
    ax.set_title(f"LSTM Forecast (With Weather)\n{label} (MAE: {mae_normal:.4f})")
    ax.set_ylabel("abvaerk (MWh)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hours, targets_h[idx], label='Actual', linewidth=2)
    ax.plot(hours, q50_zero[idx], label='LSTM Median (weather = 0)', linewidth=2)
    ax.fill_between(hours, q10_zero[idx], q90_zero[idx], alpha=0.2, label='Raw q10–q90 interval')
    if u_alpha is not None:
        u_alpha_1d = np.array(u_alpha).flatten()
        ax.fill_between(hours, q10_zero[idx] - u_alpha_1d, q90_zero[idx] + u_alpha_1d,
                        alpha=0.2, label='Calibrated interval')
    ax.set_title(f"LSTM Forecast (Weather Removed)\n(MAE: {mae_zero:.4f})")
    ax.set_xlabel("Forecast Hour")
    ax.set_ylabel("abvaerk (MWh)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Weather Impact on 168-Hour Forecast", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(filePaths=None, logger=None, run_dir=None):
    """
    Generate evaluation plots from a trained model checkpoint.

    Args:
        filePaths : [dataset_path, model_path]
        logger    : Logger instance (optional).
        run_dir   : Directory under which a Plots/ sub-folder is created.
    """
    model_path   = filePaths[1]
    dataset_path = filePaths[0]
    plot_dir     = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    test_plot_path      = os.path.join(plot_dir, "test_predictions.png")
    scatter_plot_path   = os.path.join(plot_dir, "actual_vs_predicted.png")
    residuals_plot_path = os.path.join(plot_dir, "residuals.png")
    horizon_plot_path   = os.path.join(plot_dir, "per_horizon_metrics.png")
    coverage_plot_path  = os.path.join(plot_dir, "quantile_coverage.png")
    pinball_plot_path   = os.path.join(plot_dir, "pinball_loss.png")
    weather_plot_path   = os.path.join(plot_dir, "weather_impact.png")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    best_epoch = checkpoint['epoch']

    u_alpha_t = checkpoint.get('conformal_u_alpha', None)
    if u_alpha_t is None:
        print("No per-horizon u_alpha found — using scalar fallback")
        u_alpha_t = 0.0
    else:
        if isinstance(u_alpha_t, torch.Tensor):
            u_alpha_t = u_alpha_t.cpu().numpy()
        elif isinstance(u_alpha_t, list):
            u_alpha_t = np.array(u_alpha_t)
        if np.isscalar(u_alpha_t):
            print("u_alpha is scalar:", u_alpha_t)
        else:
            print("u_alpha vector length:", len(u_alpha_t))
            for value in u_alpha_t:
                print(value)

    # ── Load dataset ───────────────────────────────────────────────────────────
    dataset      = torch.load(dataset_path, weights_only=False)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    n_total = len(full_dataset)
    train_size, valcal_size, _, test_size = _get_split_indices(n_total)

    # Chronological boundaries
    cal_start  = train_size                      # cal  = full valcal pool
    cal_end    = train_size + valcal_size
    test_start = cal_end
    test_end   = test_start + test_size

    if logger:
        logger.info(
            f"Split — train: {train_size}  val: {valcal_size}  "
            f"cal: {valcal_size}  test: {test_size}  (total: {n_total})"
        )

    cal_dataset  = Subset(full_dataset, range(cal_start,  cal_end))
    test_dataset = Subset(full_dataset, range(test_start, test_end))
    config       = Config()

    # ── Build model ───────────────────────────────────────────────────────────
    model = LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ── Rescale helpers ────────────────────────────────────────────────────────
    if "demand_mean" in dataset and "demand_std" in dataset:
        demand_mean = float(dataset["demand_mean"])
        demand_std  = float(dataset["demand_std"])
    else:
        all_t = target_data.detach().cpu().numpy()
        demand_mean = float(all_t.mean())
        demand_std  = float(all_t.std())

    def rescale(arr):
        return arr * demand_std + demand_mean

    # ── Test set inference ─────────────────────────────────────────────────────
    q10_n, q50_n, q90_n, tgt_n = _run_inference(model, test_dataset, config)
    q10_h     = rescale(q10_n)
    q50_h     = rescale(q50_n)
    q90_h     = rescale(q90_n)
    targets_h = rescale(tgt_n)

    # ── Cal set inference (for coverage plot) ──────────────────────────────────
    q10_cal_n, _, q90_cal_n, tgt_cal_n = _run_inference(model, cal_dataset, config)
    q10_cal_h  = rescale(q10_cal_n)
    q90_cal_h  = rescale(q90_cal_n)
    tgt_cal_h  = rescale(tgt_cal_n)

    # ── Rescale u_alpha to MWh ─────────────────────────────────────────────────
    u_alpha_mwh = u_alpha_t * demand_std

    test_start_global_idx = test_start

    # ── Forecast windows plot ──────────────────────────────────────────────────
    plot_forecast_windows(q10_h, q50_h, q90_h, targets_h,
                          test_start_global_idx, test_plot_path, u_alpha=u_alpha_mwh)
    if logger:
        logger.success("Saved: forecast windows plot")

    # ── Weather impact plot ────────────────────────────────────────────────────
    with torch.no_grad():
        idx = 0
        enc, dec, tgt = test_dataset[idx]
        enc = enc.unsqueeze(0).to(config.device)
        dec = dec.unsqueeze(0).to(config.device)

        print("\nBEFORE (same sample used in plot):")
        print(tuple(dec[0, 0, :11].detach().cpu().numpy()))

        dec_zero = dec.clone()
        dec_zero[:, :, 6:11] = 0

        print("\nAFTER (weather removed, same sample):")
        print(tuple(dec_zero[0, 0, :11].detach().cpu().numpy()))

        q10_z, q50_z, q90_z = model(enc, dec_zero)

    q10_zero_h = rescale(q10_z.cpu().numpy())
    q50_zero_h = rescale(q50_z.cpu().numpy())
    q90_zero_h = rescale(q90_z.cpu().numpy())

    plot_lstm_weather_impact(
        q10_h, q50_h, q90_h,
        q10_zero_h, q50_zero_h, q90_zero_h,
        targets_h,
        test_start_global_idx,
        weather_plot_path,
        u_alpha=u_alpha_mwh
    )
    if logger:
        logger.success("Saved: weather impact plot")

    # ── Scatter plot ───────────────────────────────────────────────────────────
    plot_actual_vs_predicted(q50_h, targets_h, encoder_data,
                             train_size, valcal_size, valcal_size,
                             demand_mean, demand_std, scatter_plot_path)
    if logger:
        logger.success("Saved: actual vs predicted scatter plot")

    # ── Residuals ──────────────────────────────────────────────────────────────
    plot_residual_diagnostics(q50_h, targets_h, test_start_global_idx, residuals_plot_path)
    if logger:
        logger.success("Saved: residual diagnostics plot")

    # ── Per-horizon metrics ────────────────────────────────────────────────────
    plot_per_horizon_metrics(q50_h, targets_h, encoder_data,
                             train_size, valcal_size, valcal_size,
                             demand_mean, demand_std, horizon_plot_path)
    if logger:
        logger.success("Saved: per-horizon metrics plot")

    # ── Coverage plot — cal set + test set ────────────────────────────────────
    plot_quantile_coverage(
        q10_cal_h, q90_cal_h, tgt_cal_h,   # cal set — fits own calibration data
        u_alpha_mwh,                         # already in MWh
        coverage_plot_path,
        q10_test=q10_h,                      # test set — honest generalization
        q90_test=q90_h,
        targets_test=targets_h
    )
    if logger:
        logger.success("Saved: quantile coverage plot (raw + calibrated)")

    # ── Pinball loss ───────────────────────────────────────────────────────────
    plot_pinball_loss(q10_h, q50_h, q90_h, targets_h, pinball_plot_path)
    if logger:
        logger.success("Saved: pinball loss plot")

    # ── README ─────────────────────────────────────────────────────────────────
    generate_evaluation_readme(
        plot_dir, best_epoch, checkpoint['val_loss'], q50_h.shape[0],
        train_size, valcal_size, test_size, n_total,
        model_filename=os.path.basename(model_path)
    )