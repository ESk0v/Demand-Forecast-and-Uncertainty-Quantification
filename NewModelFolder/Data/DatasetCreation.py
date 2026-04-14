from fastapi import logger
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def main(local=False, filePaths=None, logger=None):
    csv_file    = filePaths[0]
    output_path = filePaths[1]

    encoder_history = 168  # 1 week of past data
    forecast_length = 168  # 1 week forecast

    df = pd.read_csv(csv_file, parse_dates=["dateTime"])
    logger.info(f"Loaded dataset")

    # -----------------------------
    # Interpolate missing values
    # -----------------------------
    df['abvaerk'] = df['abvaerk'].interpolate(method='linear').bfill().ffill()

    forecast_features = ['toutdoor', 'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover']
    for col in forecast_features:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    df = add_forecast_uncertainty_noise(
        df,
        forecast_length=forecast_length,
        max_noise_pct=0.80,
        logger=logger
    )

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
    val_ratio  = 0.25
    test_ratio = 0.15

    n_total_windows = len(df) - encoder_history - forecast_length
    test_size  = int(n_total_windows * test_ratio)
    val_size   = int(n_total_windows * val_ratio)
    train_size = n_total_windows - val_size - test_size

    train_end = train_size

    logger.info(
        f"Split sizes (windows): train={train_size} | val={val_size} | test={test_size} | "
        f"Normalisation computed on rows 0–{train_size - 1} (training rows only)"
    )

    # -----------------------------
    # Normalisation — fit on training rows only, apply to full df
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

    # wind speed — log1p THEN normalise
    df[wind_cols] = np.log1p(df[wind_cols])
    wind_log_mean = float(df.iloc[:train_end][wind_cols].values.mean())
    wind_log_std  = float(df.iloc[:train_end][wind_cols].values.std())
    df[wind_cols] = (df[wind_cols] - wind_log_mean) / wind_log_std

    # precipitation — log1p THEN normalise
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
    # Demand lag features — computed AFTER normalisation
    # so they are on the same scale as abvaerk.
    # bfill() handles the first rows where shift produces NaN.
    # -----------------------------
    df = df.copy()
    df['lag_1h']   = df['abvaerk'].shift(1).bfill()
    df['lag_24h']  = df['abvaerk'].shift(24).bfill()
    df['lag_168h'] = df['abvaerk'].shift(168).bfill()

    # -----------------------------
    # Horizon index — same for every window
    # 0.0 at hour 1, 1.0 at hour 168
    # Gives the decoder an explicit "how far into the future am I" signal
    # -----------------------------
    horizon_index = np.linspace(0.0, 1.0, forecast_length, dtype=np.float32).reshape(-1, 1)

    # -----------------------------
    # Feature lists
    #
    # Encoder (11 features):
    #   abvaerk, toutdoor, hour_sin, hour_cos,
    #   weekday_sin, weekday_cos, month_sin, month_cos,
    #   lag_1h, lag_24h, lag_168h
    #
    # Decoder (15 features):
    #   6  time features (hour/weekday/month sin+cos)
    #   5  weather forecast (temp, humidity, wind, precip, cloud)
    #   1  last known demand (constant across horizon)
    #   1  horizon index (0.0 → 1.0)
    #   1  lag_24h  (demand yesterday same hour, constant across horizon)
    #   1  lag_168h (demand last week same hour, constant across horizon)
    # -----------------------------
    encoder_features = [
        'abvaerk',
        'toutdoor',
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos',
        'lag_1h',    # demand 1 hour ago
        'lag_24h',   # demand yesterday same hour
        'lag_168h',  # demand last week same hour
    ]

    decoder_time_features = [
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos',
    ]

    encoder_data = []
    decoder_data = []
    target_data  = []

    logger.info("Building encoder/decoder/target tensors...")

    for i in tqdm(range(n_total_windows), disable=True):
        enc_start = i
        enc_end   = i + encoder_history
        dec_end   = enc_end + forecast_length

        # ── Encoder: historical window ─────────────────────────────────────────
        # Shape: (168, 11)
        encoder_slice = df.iloc[enc_start:enc_end][encoder_features].values.astype(np.float32)

        # ── Decoder time features: future timestamps ───────────────────────────
        # Shape: (168, 6)
        decoder_time_slice = df.iloc[enc_end:dec_end][decoder_time_features].values.astype(np.float32)

        # ── Decoder weather forecast ───────────────────────────────────────────
        # Shape: (168, 5)
        forecast_row = df.iloc[enc_end]
        decoder_forecast_slice = np.zeros((forecast_length, 5), dtype=np.float32)
        for j, cols in enumerate(forecast_cols_all):
            decoder_forecast_slice[:, j] = forecast_row[cols].values[:forecast_length]

        # ── Last known demand — repeated constant ──────────────────────────────
        # The demand value at the final encoder timestep.
        # Shape: (168, 1)
        last_demand     = float(df.iloc[enc_end - 1]['abvaerk'])
        last_demand_col = np.full((forecast_length, 1), last_demand, dtype=np.float32)

        # ── Lag features for decoder — repeated constants ──────────────────────
        # lag_24h:  demand at same clock hour yesterday relative to forecast start
        # lag_168h: demand at same clock hour last week relative to forecast start
        # Using max(0, ...) guards against very early windows in the dataset.
        # Shape: (168, 1) each
        lag_24h  = float(df.iloc[max(0, enc_end - 24)]['abvaerk'])
        lag_168h = float(df.iloc[max(0, enc_end - 168)]['abvaerk'])
        lag_24h_col  = np.full((forecast_length, 1), lag_24h,  dtype=np.float32)
        lag_168h_col = np.full((forecast_length, 1), lag_168h, dtype=np.float32)

        # ── Assemble decoder input ─────────────────────────────────────────────
        # Shape: (168, 15)
        decoder_slice = np.concatenate([
            decoder_time_slice,      # (168, 6)  future time features
            decoder_forecast_slice,  # (168, 5)  weather forecast
            last_demand_col,         # (168, 1)  last known demand
            horizon_index,           # (168, 1)  how far into the future
            lag_24h_col,             # (168, 1)  demand yesterday same hour
            lag_168h_col,            # (168, 1)  demand last week same hour
        ], axis=1)

        # ── Target: future demand ──────────────────────────────────────────────
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

def add_forecast_uncertainty_noise(df, forecast_length=168, max_noise_pct=0.80, logger=None):
    """
    Adds structured, horizon-growing noise to forecast columns only.
    
    At step i (0-indexed):
      - amplitude = (i / (forecast_length - 1)) * max_noise_pct
      - noise     = random uniform in [-amplitude, +amplitude]
    
    Each variable (temp, humidity, wind, precip, cloud) gets
    independent noise draws — applied to raw values before normalisation.
    """
    df = df.copy()

    forecast_vars = {
        'temperature':     [f"temperature_{i}"     for i in range(forecast_length)],
        'relativeHumidity':[f"relativeHumidity_{i}" for i in range(forecast_length)],
        'windSpeed':       [f"windSpeed_{i}"        for i in range(forecast_length)],
        'precipitation':   [f"precipitation_{i}"    for i in range(forecast_length)],
        'cloudCover':      [f"cloudCover_{i}"       for i in range(forecast_length)],
    }

    for var_name, cols in forecast_vars.items():
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue

            amplitude = (i / (forecast_length - 1)) * max_noise_pct

            if i in [0, forecast_length // 2, forecast_length - 1]:
                logger.info(f"{col}: amplitude={amplitude:.3f}")
    
            noise = np.random.uniform(-amplitude, amplitude, size=len(df))
            df[col] = df[col] * (1.0 + noise)

    return df