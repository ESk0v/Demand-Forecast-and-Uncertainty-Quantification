import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config, LSTMForecast


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATASET_START        = pd.Timestamp("2023-01-01 01:00")
ENCODER_HISTORY      = 168
FORECAST_HOURS       = 168
DAY_HOURS            = [24, 48, 72, 96, 120, 144, 168]
DAY_LABELS           = ["1d", "2d", "3d", "4d", "5d", "6d", "7d"]
WEATHER_DECODER_COLS = slice(6, 11)   # decoder indices that carry weather forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Split helper — must mirror LSTMTraining.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def get_split_indices(n_total, val_ratio=1/12, cal_ratio=1/12, test_ratio=1/6):
    """
    Returns (train_size, valcal_size, test_size).

    val and cal both span the full valcal pool — they overlap intentionally
    since neither set touches model weights (no leakage).

      test_start = train_size + valcal_size
    """
    test_size   = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size  = n_total - valcal_size - test_size
    return train_size, valcal_size, test_size


# ─────────────────────────────────────────────────────────────────────────────
# Data / model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model_path):
    """Load and return the checkpoint dict from disk."""
    return torch.load(model_path, map_location="cpu", weights_only=False)


def load_dataset(dataset_path):
    """
    Load dataset tensors.
    Returns (raw_dict, full_TensorDataset, encoder, decoder, target).
    """
    raw          = torch.load(dataset_path, weights_only=False)
    encoder_data = raw["encoder"]
    decoder_data = raw["decoder"]
    target_data  = raw["target"]
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)
    return raw, full_dataset, encoder_data, decoder_data, target_data


def build_model(config, checkpoint):
    """Instantiate LSTMForecast, load weights, and set to eval mode."""
    model = LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_demand_stats(raw_dataset, target_data):
    """
    Return (demand_mean, demand_std) from dataset metadata if present,
    otherwise compute from the target array.
    """
    if "demand_mean" in raw_dataset and "demand_std" in raw_dataset:
        return float(raw_dataset["demand_mean"]), float(raw_dataset["demand_std"])
    arr = target_data.detach().cpu().numpy()
    return float(arr.mean()), float(arr.std())


def get_u_alpha(checkpoint, demand_std):
    """
    Extract per-horizon conformal offsets from checkpoint and rescale to MWh.
    Returns a 1-D numpy array of shape (168,), or 0.0 if not present.
    """
    u = checkpoint.get("conformal_u_alpha", None)
    if u is None:
        return 0.0
    if isinstance(u, torch.Tensor):
        u = u.cpu().numpy()
    elif isinstance(u, list):
        u = np.array(u)
    return u * demand_std


def rescale(arr, mean, std):
    """Denormalise a numpy array back to raw MWh."""
    return arr * std + mean


def run_inference(model, dataset_subset, config):
    """
    Run full inference over a Subset/Dataset.
    Returns (q10, q50, q90, targets) as normalised numpy arrays.
    """
    loader = DataLoader(dataset_subset, batch_size=config.batch_size, shuffle=False)
    q10s, q50s, q90s, tgts = [], [], [], []
    with torch.no_grad():
        for enc, dec, tgt in loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            q10, q50, q90 = model(enc, dec)
            q10s.append(q10.cpu().numpy())
            q50s.append(q50.cpu().numpy())
            q90s.append(q90.cpu().numpy())
            tgts.append(tgt.numpy())
    return (
        np.concatenate(q10s),
        np.concatenate(q50s),
        np.concatenate(q90s),
        np.concatenate(tgts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Axis / drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_day_markers(ax):
    """Vertical day-boundary lines + labels along the top of an axis."""
    for h, lbl in zip(DAY_HOURS, DAY_LABELS):
        ax.axvline(x=h, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)
        ax.annotate(
            lbl,
            xy=(h, 1.01),
            xycoords=("data", "axes fraction"),
            ha="center", fontsize=8, color="gray",
        )


def window_date_label(sample_idx, test_start_global_idx):
    """Human-readable date-range string for a test window."""
    global_idx   = test_start_global_idx + sample_idx
    window_start = DATASET_START + pd.Timedelta(hours=global_idx + ENCODER_HISTORY)
    window_end   = window_start + pd.Timedelta(hours=FORECAST_HOURS - 1)
    return (
        f"{window_start.strftime('%Y-%m-%d %H:%M')} "
        f"→ {window_end.strftime('%Y-%m-%d %H:%M')}"
    )


def draw_forecast_panel(ax, hours, actual, q50, q10, q90, u_alpha, title, mae):
    """
    Draw a single forecast panel:
      - Actual demand (blue)
      - Median forecast (orange)
      - Raw q10–q90 band
      - Calibrated band (if u_alpha is a vector)
    """
    ax.plot(hours, actual, label="Actual",       color="steelblue",  linewidth=1.8)
    ax.plot(hours, q50,    label="Median (q50)", color="darkorange", linewidth=1.8)

    ax.fill_between(hours, q10, q90,
                    color="darkorange", alpha=0.20, label="Raw q10–q90 interval")

    if u_alpha is not None and not np.isscalar(u_alpha):
        ax.fill_between(
            hours, q10 - u_alpha, q90 + u_alpha,
            color="darkorange", alpha=0.10, label="Calibrated interval",
        )

    add_day_markers(ax)
    ax.set_title(f"{title}\nMAE: {mae:.4f} MWh", fontsize=10)
    ax.set_ylabel("abvaerk (MWh)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: weather impact (2-panel)
# ─────────────────────────────────────────────────────────────────────────────

def _weather_removed_inference(model, test_dataset, config, sample_idx):
    """
    Re-run inference on one test sample with weather decoder columns zeroed.
    Returns (q10, q50, q90) as 1-D normalised numpy arrays.
    """
    enc, dec, _ = test_dataset[sample_idx]
    enc = enc.unsqueeze(0).to(config.device)
    dec = dec.clone().unsqueeze(0).to(config.device)
    dec[:, :, WEATHER_DECODER_COLS] = 0.0

    with torch.no_grad():
        q10, q50, q90 = model(enc, dec)

    return (
        q10.squeeze(0).cpu().numpy(),
        q50.squeeze(0).cpu().numpy(),
        q90.squeeze(0).cpu().numpy(),
    )


def plot_weather_impact(
    q10_h, q50_h, q90_h,
    targets_h,
    test_dataset,
    model,
    config,
    u_alpha,
    demand_mean,
    demand_std,
    test_start_global_idx,
    save_path,
    sample_idx=0,
):
    
    hours = np.arange(1, FORECAST_HOURS + 1)
    label = window_date_label(sample_idx, test_start_global_idx)

    # ── With-weather data ─────────────────────────────────────────────────────
    actual = targets_h[sample_idx]
    q10    = q10_h[sample_idx]
    q50    = q50_h[sample_idx]
    q90    = q90_h[sample_idx]
    mae_w  = float(np.mean(np.abs(actual - q50)))

    # ── Weather-removed inference + rescale ───────────────────────────────────
    q10_z_n, q50_z_n, q90_z_n = _weather_removed_inference(
        model, test_dataset, config, sample_idx
    )
    q10_z = rescale(q10_z_n, demand_mean, demand_std)
    q50_z = rescale(q50_z_n, demand_mean, demand_std)
    q90_z = rescale(q90_z_n, demand_mean, demand_std)
    mae_z = float(np.mean(np.abs(actual - q50_z)))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Weather Impact on 168-Hour Forecast", fontsize=14, fontweight="bold")

    draw_forecast_panel(
        axes[0], hours, actual, q50, q10, q90, u_alpha,
        title=f"LSTM Forecast (With Weather)\n{label}",
        mae=mae_w,
    )

    draw_forecast_panel(
        axes[1], hours, actual, q50_z, q10_z, q90_z, u_alpha,
        title="LSTM Forecast (Weather Removed)",
        mae=mae_z,
    )

    axes[1].set_xlabel("Forecast Hour", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: per-horizon coverage (raw / calibrated cal set / calibrated test set)
# ─────────────────────────────────────────────────────────────────────────────
 
def _coverage_per_horizon(q10, q90, targets, u_alpha):
    """
    Compute per-horizon empirical coverage for a given set of predictions.
 
    Args:
        q10, q90  : arrays of shape (N, 168) in MWh
        targets   : array of shape (N, 168) in MWh
        u_alpha   : per-horizon offsets (168,) in MWh, or scalar 0.0
 
    Returns:
        raw_cov : (168,) coverage of the raw [q10, q90] interval
        cal_cov : (168,) coverage of the calibrated [q10-u, q90+u] interval
    """
    u = np.array(u_alpha).reshape(1, -1) if not np.isscalar(u_alpha) else u_alpha
 
    raw_cov = ((targets >= q10) & (targets <= q90)).mean(axis=0) * 100
    cal_cov = ((targets >= q10 - u) & (targets <= q90 + u)).mean(axis=0) * 100
 
    return raw_cov, cal_cov
 
 
def _coverage_summary(label, coverage_arr):
    """Print h1 / h24 / h72 / h168 coverage for quick inspection."""
    print(
        f"  {label:40s} "
        f"h1: {coverage_arr[0]:.1f}%  "
        f"h24: {coverage_arr[23]:.1f}%  "
        f"h72: {coverage_arr[71]:.1f}%  "
        f"h168: {coverage_arr[-1]:.1f}%"
    )
 
 
def plot_coverage(
    q10_cal, q90_cal, targets_cal,
    q10_test, q90_test, targets_test,
    u_alpha,
    save_path,
    nominal_raw=0.80,
    nominal_cal=0.90,
):
    """
    Per-horizon coverage plot with three lines:
      Blue        — raw q10–q90 coverage on the cal set (uncalibrated)
      Green       — calibrated coverage on the cal set  (fits own data)
      Crimson (--)— calibrated coverage on the test set (honest generalisation)
 
    The gap between green and crimson is the key diagnostic: how well do the
    conformal thresholds transfer to unseen data?
 
    Args:
        q10_cal, q90_cal, targets_cal   : cal set arrays in MWh  (N_cal, 168)
        q10_test, q90_test, targets_test: test set arrays in MWh (N_test, 168)
        u_alpha   : per-horizon conformal offsets in MWh (168,)
        save_path : full output path for the PNG
        nominal_raw : reference line for the raw interval (default 0.80)
        nominal_cal : reference line for the calibrated interval (default 0.90)
    """
    hours = np.arange(1, FORECAST_HOURS + 1)
 
    # ── Compute coverage ──────────────────────────────────────────────────────
    raw_cov,      cal_set_cov  = _coverage_per_horizon(q10_cal,  q90_cal,  targets_cal,  u_alpha)
    _,            test_set_cov = _coverage_per_horizon(q10_test, q90_test, targets_test, u_alpha)
 
    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nCoverage summary:")
    _coverage_summary("Raw q10–q90 (cal set)",           raw_cov)
    _coverage_summary("Calibrated — cal set  (own data)", cal_set_cov)
    _coverage_summary("Calibrated — test set (unseen)",   test_set_cov)
 
    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
 
    ax.plot(hours, raw_cov,
            color="steelblue", linewidth=1.5,
            label="Raw q10–q90 (cal set, uncalibrated)")
 
    ax.plot(hours, cal_set_cov,
            color="green", linewidth=1.5,
            label="Calibrated — cal set (fits own data)")
 
    ax.plot(hours, test_set_cov,
            color="crimson", linewidth=1.5, linestyle="--",
            label="Calibrated — test set (generalisation)")
 
    ax.axhline(nominal_cal * 100, color="orange", linestyle="--", linewidth=1.0,
               label=f"Nominal {nominal_cal*100:.0f}% (calibrated target)")
    ax.axhline(nominal_raw * 100, color="black",  linestyle="--", linewidth=1.0,
               label=f"Nominal {nominal_raw*100:.0f}% (raw interval target)")
 
    add_day_markers(ax)
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title(
        "Prediction Interval Coverage per Horizon\n"
        "Raw  |  Calibrated on Cal Set  |  Calibrated on Test Set",
        fontsize=13,
    )
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(filePaths=None, logger=None, run_dir=None):
    dataset_path = filePaths[0]
    model_path   = filePaths[1]
    plot_dir     = os.path.join(run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    checkpoint = load_checkpoint(model_path)
    raw_dataset, full_dataset, encoder_data, decoder_data, target_data = load_dataset(dataset_path)

    # ── Split ──────────────────────────────────────────────────────────────────
    n_total    = len(full_dataset)
    train_size, valcal_size, test_size = get_split_indices(n_total)
    cal_start  = train_size
    cal_end    = train_size + valcal_size
    test_start = cal_end
    test_end   = test_start + test_size

    if logger:
        logger.info(
            f"Split — train: {train_size}  valcal: {valcal_size}  "
            f"test: {test_size}  (total: {n_total})"
        )

    cal_dataset  = Subset(full_dataset, range(cal_start,  cal_end))
    test_dataset = Subset(full_dataset, range(test_start, test_end))

    # ── Model ──────────────────────────────────────────────────────────────────
    config = Config()
    model  = build_model(config, checkpoint)

    # ── Demand stats + conformal offsets ───────────────────────────────────────
    demand_mean, demand_std = get_demand_stats(raw_dataset, target_data)
    u_alpha                 = get_u_alpha(checkpoint, demand_std)

    # ── Test set inference + rescale ───────────────────────────────────────────
    q10_n, q50_n, q90_n, tgt_n = run_inference(model, test_dataset, config)
    q10_h     = rescale(q10_n, demand_mean, demand_std)
    q50_h     = rescale(q50_n, demand_mean, demand_std)
    q90_h     = rescale(q90_n, demand_mean, demand_std)
    targets_h = rescale(tgt_n, demand_mean, demand_std)

    # ── Cal set inference + rescale ────────────────────────────────────────────
    q10_cal_n, _, q90_cal_n, tgt_cal_n = run_inference(model, cal_dataset, config)
    q10_cal_h = rescale(q10_cal_n, demand_mean, demand_std)
    q90_cal_h = rescale(q90_cal_n, demand_mean, demand_std)
    tgt_cal_h = rescale(tgt_cal_n, demand_mean, demand_std)

    # ── Plot: weather impact ───────────────────────────────────────────────────
    plot_weather_impact(
        q10_h, q50_h, q90_h,
        targets_h,
        test_dataset,
        model,
        config,
        u_alpha,
        demand_mean,
        demand_std,
        test_start_global_idx=test_start,
        save_path=os.path.join(plot_dir, "weather_impact.png"),
        sample_idx=0,
    )
    if logger:
        logger.success("Saved: weather impact plot")

    # ── Plot: coverage ─────────────────────────────────────────────────────────
    plot_coverage(
        q10_cal_h, q90_cal_h, tgt_cal_h,
        q10_h,     q90_h,     targets_h,
        u_alpha,
        save_path=os.path.join(plot_dir, "coverage.png"),
    )
    if logger:
        logger.success("Saved: coverage plot")