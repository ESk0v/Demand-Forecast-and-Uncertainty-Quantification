import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def _interpolate_abvaerk_knn(df, k=4, temp_range_std=2.0):
    """
    Interpolate missing abvaerk values using day-of-week preference + local trend blending.

    Strategy:
    1. For each missing value, find neighbors from SAME HOUR + SAME WEEKDAY in past data
    2. Further filter by temperature similarity and season
    3. Use weighted average of nearest neighbors
    4. Blend with local trend (70% KNN + 30% local trend) for smoothness

    This preserves both daily heating patterns AND weekday/weekend differences,
    while keeping interpolated values smooth with their neighbors.
    """
    df = df.copy()
    missing_mask = df['abvaerk'].isna()

    if not missing_mask.any():
        return df

    valid_mask = ~missing_mask
    missing_indices = np.where(missing_mask)[0]

    # Temperature stats
    temp_std = df['toutdoor'].std() if df['toutdoor'].std() > 1e-8 else 1.0

    for missing_idx in missing_indices:
        current_date = df.iloc[missing_idx]['dateTime']
        current_hour = current_date.hour
        current_weekday = current_date.weekday()
        current_month = current_date.month
        current_temp = df.iloc[missing_idx]['toutdoor']

        # Month range: current month ±1 (seasonal consistency)
        month_range = [(current_month - 2) % 12 + 1, current_month, (current_month) % 12 + 1]

        # Strategy 1: Same hour + SAME WEEKDAY + same season + temperature
        past_mask = (
            (valid_mask) &
            (df['dateTime'] < current_date) &  # Only past data
            (df['dateTime'].dt.hour == current_hour) &  # SAME HOUR OF DAY
            (df['dateTime'].dt.weekday == current_weekday) &  # SAME WEEKDAY (critical!)
            (df['dateTime'].dt.month.isin(month_range))  # Same season
        )

        # Temperature similarity filter around current temperature (±temp_range_std std devs)
        if not pd.isna(current_temp):
            temp_min = current_temp - temp_range_std * temp_std
            temp_max = current_temp + temp_range_std * temp_std
            past_mask = past_mask & (df['toutdoor'] >= temp_min) & (df['toutdoor'] <= temp_max)

        # Fallback 1: If no same-weekday match, try same hour + season only
        if not past_mask.any():
            past_mask = (
                (valid_mask) &
                (df['dateTime'] < current_date) &
                (df['dateTime'].dt.hour == current_hour) &
                (df['dateTime'].dt.month.isin(month_range))
            )

        # Fallback 2: If still no data, try same hour (any month/weekday)
        if not past_mask.any():
            past_mask = (
                (valid_mask) &
                (df['dateTime'] < current_date) &
                (df['dateTime'].dt.hour == current_hour)
            )

        # Fallback 3: Use all past data
        if not past_mask.any():
            past_mask = (valid_mask) & (df['dateTime'] < current_date)

        if not past_mask.any():
            # Absolute fallback: use all valid data
            past_mask = valid_mask

        y_valid = df.loc[past_mask, 'abvaerk'].values

        if len(y_valid) == 0:
            continue

        # Compute KNN interpolation
        if len(y_valid) < k:
            # Not enough neighbors, use simple average
            knn_value = np.mean(y_valid)
        else:
            # Compute distance based on temperature similarity
            temps_valid = df.loc[past_mask, 'toutdoor'].values

            if not pd.isna(current_temp):
                # Distance = temperature difference (prefer matches with similar temp)
                distances = np.abs(temps_valid - current_temp)

                # Find k nearest neighbors
                k_nearest_idx = np.argsort(distances)[:k]
                neighbor_values = y_valid[k_nearest_idx]
                neighbor_distances = distances[k_nearest_idx]

                # Inverse distance weighting
                weights = 1.0 / (neighbor_distances + 1e-10)
                weights /= weights.sum()
                knn_value = np.sum(weights * neighbor_values)
            else:
                # No temperature available, use average of k recent values
                knn_value = np.mean(y_valid[-k:])

        # Strategy 2: Blend with local trend for smoothness
        # Compute local trend from surrounding valid data
        local_trend = _compute_local_trend(df, missing_idx, valid_mask)

        if local_trend is not None:
            # Blend: 70% KNN + 30% local trend
            interpolated_value = 0.7 * knn_value + 0.3 * local_trend
        else:
            # No local trend available, use KNN value directly
            interpolated_value = knn_value

        df.loc[missing_idx, 'abvaerk'] = interpolated_value

    # Final fallback: linear interpolation for any remaining NaNs
    df['abvaerk'] = df['abvaerk'].interpolate(method='linear').bfill().ffill()

    return df


def _compute_local_trend(df, missing_idx, valid_mask):
    """
    Compute local trend from valid data before and after the missing point.

    Returns the trend at the missing point based on surrounding data.
    Returns None if insufficient neighbors.
    """
    # Look up to 24 hours before and after for valid data
    before_idx = None
    after_idx = None

    # Find last valid point before
    for i in range(missing_idx - 1, max(-1, missing_idx - 25), -1):
        if i >= 0 and valid_mask.iloc[i]:
            before_idx = i
            break

    # Find first valid point after
    for i in range(missing_idx + 1, min(len(df), missing_idx + 25)):
        if valid_mask.iloc[i]:
            after_idx = i
            break

    # Need at least one neighbor
    if before_idx is None and after_idx is None:
        return None

    # If only one neighbor, return its value
    if before_idx is None:
        return float(df.iloc[after_idx]['abvaerk'])
    if after_idx is None:
        return float(df.iloc[before_idx]['abvaerk'])

    # Linear interpolation between before and after
    val_before = float(df.iloc[before_idx]['abvaerk'])
    val_after = float(df.iloc[after_idx]['abvaerk'])
    dist_before = missing_idx - before_idx
    dist_after = after_idx - missing_idx
    total_dist = dist_before + dist_after

    # Weighted average based on distance
    trend_value = (val_after * dist_before + val_before * dist_after) / total_dist

    return trend_value


def main(filePaths=None, logger=None):
    csv_file    = filePaths[0]
    output_path = filePaths[1]

    encoder_history = 168  # 1 week of past data
    forecast_length = 168  # 1 week forecast

    df = pd.read_csv(csv_file, parse_dates=["dateTime"])
    logger.info(f"Loaded dataset")

    # -----------------------------
    # Interpolate missing values
    # -----------------------------
    # abvaerk: use KNN interpolation (preserves daily patterns)
    df = _interpolate_abvaerk_knn(df, k=5)
    logger.info("Interpolated abvaerk using KNN (temporal + temperature matching)")

    # Forecast features: linear interpolation (these are forecasts, not measurements)
    forecast_features = ['toutdoor', 'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover']
    for col in forecast_features:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # -----------------------------
    # Build forecast column lists
    # -----------------------------
    temperature_cols = [f"temperature_{i}"      for i in range(forecast_length)]
    humidity_cols    = [f"relativeHumidity_{i}"  for i in range(forecast_length)]
    wind_cols        = [f"windSpeed_{i}"         for i in range(forecast_length)]
    precip_cols      = [f"precipitation_{i}"     for i in range(forecast_length)]
    cloud_cols       = [f"cloudCover_{i}"        for i in range(forecast_length)]
    forecast_cols_all = [temperature_cols, humidity_cols, wind_cols, precip_cols, cloud_cols]

    # -----------------------------
    # Train/val/test split boundary
    # -----------------------------
    val_ratio  = 0.1
    test_ratio = 0.1

    n_total_windows = len(df) - encoder_history - forecast_length
    test_size  = int(n_total_windows * test_ratio)
    val_size   = int(n_total_windows * val_ratio)
    train_size = n_total_windows - val_size - test_size

    # Store boundary index only — do NOT slice train_df here.
    # Slicing before transforms creates a pandas view that may not
    # reflect in-place modifications to df (log transforms applied below).
    # All normalisation stats are computed from df.iloc[:train_size] AFTER
    # transforms to guarantee correctness.
    train_end = train_size

    logger.info(
        f"Split sizes (windows): train={train_size} | val={val_size} | test={test_size} | "
        f"Normalisation computed on rows 0–{train_size - 1} (training rows only)"
    )

    # -----------------------------
    # Normalisation — fit on training rows only, apply to full df
    # Stats computed directly from df.iloc[:train_end] to avoid
    # pandas view/copy ambiguity with pre-sliced train_df
    # -----------------------------

    # demand
    demand_mean = float(df.iloc[:train_end]['abvaerk'].mean())
    demand_std  = float(df.iloc[:train_end]['abvaerk'].std())
    df['abvaerk'] = (df['abvaerk'] - demand_mean) / demand_std

    # encoder outdoor temperature
    toutdoor_mean = float(df.iloc[:train_end]['toutdoor'].mean())
    toutdoor_std  = float(df.iloc[:train_end]['toutdoor'].std())
    df['toutdoor'] = (df['toutdoor'] - toutdoor_mean) / toutdoor_std

    # temperature forecast columns
    temp_mean = float(df.iloc[:train_end][temperature_cols].values.mean())
    temp_std  = float(df.iloc[:train_end][temperature_cols].values.std())
    df[temperature_cols] = (df[temperature_cols] - temp_mean) / temp_std

    # humidity forecast columns
    hum_mean = float(df.iloc[:train_end][humidity_cols].values.mean())
    hum_std  = float(df.iloc[:train_end][humidity_cols].values.std())
    df[humidity_cols] = (df[humidity_cols] - hum_mean) / hum_std

    # cloud cover forecast columns
    cloud_mean = float(df.iloc[:train_end][cloud_cols].values.mean())
    cloud_std  = float(df.iloc[:train_end][cloud_cols].values.std())
    df[cloud_cols] = (df[cloud_cols] - cloud_mean) / cloud_std

    # wind speed — log1p THEN compute stats from training rows only
    # Critical: stats must be computed AFTER log transform is applied to df
    # so that df.iloc[:train_end] reflects the transformed values
    df[wind_cols] = np.log1p(df[wind_cols])
    wind_log_mean = float(df.iloc[:train_end][wind_cols].values.mean())
    wind_log_std  = float(df.iloc[:train_end][wind_cols].values.std())
    df[wind_cols] = (df[wind_cols] - wind_log_mean) / wind_log_std

    # precipitation — log1p THEN compute stats from training rows only
    df[precip_cols] = np.log1p(df[precip_cols])
    precip_log_mean = float(df.iloc[:train_end][precip_cols].values.mean())
    precip_log_std  = float(df.iloc[:train_end][precip_cols].values.std())
    df[precip_cols] = (df[precip_cols] - precip_log_mean) / precip_log_std

    norm_stats = {
        'demand_mean':     demand_mean,
        'demand_std':      demand_std,
        'toutdoor_mean':   toutdoor_mean,
        'toutdoor_std':    toutdoor_std,
        'temp_mean':       temp_mean,
        'temp_std':        temp_std,
        'hum_mean':        hum_mean,
        'hum_std':         hum_std,
        'cloud_mean':      cloud_mean,
        'cloud_std':       cloud_std,
        'wind_log_mean':   wind_log_mean,
        'wind_log_std':    wind_log_std,
        'precip_log_mean': precip_log_mean,
        'precip_log_std':  precip_log_std,
    }

    logger.info(
        f"Normalisation stats — demand: mean={demand_mean:.4f}, std={demand_std:.4f} | "
        f"toutdoor: mean={toutdoor_mean:.4f}, std={toutdoor_std:.4f} | "
        f"temp: mean={temp_mean:.4f}, std={temp_std:.4f} | "
        f"hum: mean={hum_mean:.4f}, std={hum_std:.4f} | "
        f"cloud: mean={cloud_mean:.4f}, std={cloud_std:.4f} | "
        f"wind_log: mean={wind_log_mean:.4f}, std={wind_log_std:.4f} | "
        f"precip_log: mean={precip_log_mean:.4f}, std={precip_log_std:.4f}"
    )

    # -----------------------------
    # Time features
    # -----------------------------
    def get_time_features(df):
        hour    = df['dateTime'].dt.hour
        weekday = df['dateTime'].dt.weekday
        month   = df['dateTime'].dt.month

        time_features = pd.DataFrame({
            'hour_sin':    np.sin(2 * np.pi * hour    / 24),
            'hour_cos':    np.cos(2 * np.pi * hour    / 24),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7),
            'month_sin':   np.sin(2 * np.pi * month   / 12),
            'month_cos':   np.cos(2 * np.pi * month   / 12),
        }, index=df.index)

        return pd.concat([df, time_features], axis=1)

    df = get_time_features(df)

    # -----------------------------
    # Build tensors
    # -----------------------------
    encoder_features      = ['abvaerk', 'toutdoor', 'hour_sin', 'hour_cos',
                              'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']
    decoder_time_features = ['hour_sin', 'hour_cos',
                              'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']

    encoder_data = []
    decoder_data = []
    target_data  = []

    logger.info("Building encoder/decoder/target tensors...")

    for i in tqdm(range(n_total_windows), disable=True):
        enc_start = i
        enc_end   = i + encoder_history
        dec_end   = enc_end + forecast_length

        # Encoder: historical window
        encoder_slice = df.iloc[enc_start:enc_end][encoder_features].values.astype(np.float32)

        # Decoder time features: future timestamps
        decoder_time_slice = df.iloc[enc_end:dec_end][decoder_time_features].values.astype(np.float32)

        # Decoder forecast: use the single forecast issued at enc_end
        forecast_row = df.iloc[enc_end]
        decoder_forecast_slice = np.zeros((forecast_length, 5), dtype=np.float32)
        for j, cols in enumerate(forecast_cols_all):
            decoder_forecast_slice[:, j] = forecast_row[cols].values[:forecast_length]

        # Last known demand value at encoder boundary — already normalised
        # NOTE: Removed from decoder to prevent data leakage.
        # Including the last demand value directly in decoder inputs lets the model
        # trivially learn to output a flat persistence baseline, defeating the purpose
        # of having a neural network. The model should learn patterns from time + weather.

        decoder_slice = np.concatenate([
            decoder_time_slice,      # (168, 6)
            decoder_forecast_slice,  # (168, 5)
        ], axis=1)                   # → (168, 12)

        # Target: future demand
        target_slice = df.iloc[enc_end:dec_end]['abvaerk'].values.astype(np.float32)

        encoder_data.append(encoder_slice)
        decoder_data.append(decoder_slice)
        target_data.append(target_slice)

    encoder_tensor = torch.from_numpy(np.stack(encoder_data))
    decoder_tensor = torch.from_numpy(np.stack(decoder_data))
    target_tensor  = torch.from_numpy(np.stack(target_data))

    torch.save({
        'encoder': encoder_tensor,
        'decoder': decoder_tensor,
        'target':  target_tensor,
        **norm_stats,
    }, output_path)

    logger.info(
        f"Tensors saved — encoder: {encoder_tensor.shape}, "
        f"decoder: {decoder_tensor.shape}, target: {target_tensor.shape}"
    )

    return output_path
