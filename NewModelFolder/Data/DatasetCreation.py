import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def main(local=False, filePaths=None, logger=None):
    csv_file    = filePaths[0]
    output_path = filePaths[1]

    encoder_history = 168
    forecast_length = 168

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    df = pd.read_csv(csv_file, parse_dates=["dateTime"])
    _log("Loaded dataset")

    # Enforce chronological order and unique timestamps before feature logic.
    n_before = len(df)
    df = (
        df.sort_values("dateTime")
          .drop_duplicates(subset=["dateTime"], keep="first")
          .reset_index(drop=True)
    )
    n_removed = n_before - len(df)
    if n_removed > 0:
        _log(f"Removed {n_removed} duplicate timestamp rows after sorting.")

    # =====================================================
    # Required base columns for this pipeline
    # =====================================================
    required_base_cols = ['abvaerk', 'toutdoor']
    missing_base_cols = [c for c in required_base_cols if c not in df.columns]
    if missing_base_cols:
        raise ValueError(f"Missing required base columns in CSV: {missing_base_cols}")

    # =====================================================
    # Missing values
    # =====================================================
    df['abvaerk'] = df['abvaerk'].interpolate(method='linear').bfill().ffill()
    df['toutdoor'] = df['toutdoor'].interpolate(method='linear').bfill().ffill()

    # Optional unsuffixed weather columns are supported for backwards compatibility.
    # They are not required when using minimal synthetic CSV schema.
    optional_observed_weather_cols = [
        'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover'
    ]
    for col in optional_observed_weather_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # =====================================================
    # Forecast column lists
    # =====================================================
    temperature_cols = [f"temperature_{i}" for i in range(forecast_length)]
    humidity_cols    = [f"relativeHumidity_{i}" for i in range(forecast_length)]
    wind_cols        = [f"windSpeed_{i}" for i in range(forecast_length)]
    precip_cols      = [f"precipitation_{i}" for i in range(forecast_length)]
    cloud_cols       = [f"cloudCover_{i}" for i in range(forecast_length)]

    forecast_cols_all = [
        temperature_cols,
        humidity_cols,
        wind_cols,
        precip_cols,
        cloud_cols
    ]

    all_forecast_horizon_cols = [c for cols in forecast_cols_all for c in cols]
    missing_forecast_cols = [c for c in all_forecast_horizon_cols if c not in df.columns]
    if missing_forecast_cols:
        preview = missing_forecast_cols[:10]
        raise ValueError(
            f"Missing required forecast horizon columns (showing up to 10): {preview} "
            f"(total missing: {len(missing_forecast_cols)})"
        )

    for col in all_forecast_horizon_cols:
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    # =====================================================
    # Split
    # =====================================================
    val_ratio  = 0.16666
    test_ratio = 0.16666

    n_total_windows = len(df) - encoder_history - forecast_length + 1
    if n_total_windows <= 0:
        raise ValueError(
            f"Not enough rows ({len(df)}) for encoder_history={encoder_history} "
            f"and forecast_length={forecast_length}. Need at least "
            f"{encoder_history + forecast_length} rows."
        )
    test_size  = int(n_total_windows * test_ratio)
    val_size   = int(n_total_windows * val_ratio)
    train_size = n_total_windows - val_size - test_size

    train_end = train_size

    _log(f"Train={train_size}, Val={val_size}, Test={test_size}")

    # =====================================================
    # Normalisation (fit on train only)
    # =====================================================
    demand_mean = df.iloc[:train_end]['abvaerk'].mean()
    demand_std  = df.iloc[:train_end]['abvaerk'].std()
    df['abvaerk'] = (df['abvaerk'] - demand_mean) / demand_std

    toutdoor_mean = df.iloc[:train_end]['toutdoor'].mean()
    toutdoor_std  = df.iloc[:train_end]['toutdoor'].std()
    df['toutdoor'] = (df['toutdoor'] - toutdoor_mean) / toutdoor_std

    temp_mean = df.iloc[:train_end][temperature_cols].values.mean()
    temp_std  = df.iloc[:train_end][temperature_cols].values.std()
    df[temperature_cols] = (df[temperature_cols] - temp_mean) / temp_std

    hum_mean = df.iloc[:train_end][humidity_cols].values.mean()
    hum_std  = df.iloc[:train_end][humidity_cols].values.std()
    df[humidity_cols] = (df[humidity_cols] - hum_mean) / hum_std

    cloud_mean = df.iloc[:train_end][cloud_cols].values.mean()
    cloud_std  = df.iloc[:train_end][cloud_cols].values.std()
    df[cloud_cols] = (df[cloud_cols] - cloud_mean) / cloud_std

    # Synthetic raw data may contain slight negative values from generation noise.
    # Guard log1p against invalid values.
    df[wind_cols] = np.log1p(df[wind_cols].clip(lower=0.0))
    wind_mean = df.iloc[:train_end][wind_cols].values.mean()
    wind_std  = df.iloc[:train_end][wind_cols].values.std()
    df[wind_cols] = (df[wind_cols] - wind_mean) / wind_std

    df[precip_cols] = np.log1p(df[precip_cols].clip(lower=0.0))
    precip_mean = df.iloc[:train_end][precip_cols].values.mean()
    precip_std  = df.iloc[:train_end][precip_cols].values.std()
    df[precip_cols] = (df[precip_cols] - precip_mean) / precip_std

    # =====================================================
    # Time features
    # =====================================================
    hour    = df['dateTime'].dt.hour
    weekday = df['dateTime'].dt.weekday
    month   = df['dateTime'].dt.month

    df['hour_sin']    = np.sin(2*np.pi*hour/24)
    df['hour_cos']    = np.cos(2*np.pi*hour/24)
    df['weekday_sin'] = np.sin(2*np.pi*weekday/7)
    df['weekday_cos'] = np.cos(2*np.pi*weekday/7)
    df['month_sin']   = np.sin(2*np.pi*month/12)
    df['month_cos']   = np.cos(2*np.pi*month/12)

    # =====================================================
    # Lag features
    # =====================================================
    df['lag_1h']   = df['abvaerk'].shift(1).bfill()
    df['lag_24h']  = df['abvaerk'].shift(24).bfill()
    df['lag_168h'] = df['abvaerk'].shift(168).bfill()

    horizon_index = np.linspace(0, 1, forecast_length, dtype=np.float32).reshape(-1, 1)

    # =====================================================
    # Feature lists
    # =====================================================
    encoder_features = [
        'abvaerk', 'toutdoor',
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos'
    ]

    decoder_time_features = [
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos'
    ]

    encoder_data = []
    decoder_data = []
    target_data  = []

    _log("Building samples (NO NOISE)...")

    # =====================================================
    # BUILD DATASET
    # =====================================================
    for i in tqdm(range(n_total_windows), disable=True):

        enc_start = i
        enc_end   = i + encoder_history
        dec_end   = enc_end + forecast_length

        # Encoder
        encoder_slice = df.iloc[enc_start:enc_end][encoder_features].values.astype(np.float32)

        # Decoder time features
        decoder_time_slice = df.iloc[enc_end:dec_end][decoder_time_features].values.astype(np.float32)

        # =================================================
        # Forecast features (CLEAN - NO NOISE)
        # =================================================
        forecast_row = df.iloc[enc_end]

        decoder_forecast_slice = np.zeros((forecast_length, 5), dtype=np.float32)

        for j, cols in enumerate(forecast_cols_all):
            base_series = forecast_row[cols].values[:forecast_length].astype(np.float32)
            decoder_forecast_slice[:, j] = base_series

        # Static decoder features (unchanged)
        last_demand = float(df.iloc[enc_end - 1]['abvaerk'])

        lag_24h  = float(df.iloc[max(0, enc_end - 24)]['abvaerk'])
        lag_168h = float(df.iloc[max(0, enc_end - 168)]['abvaerk'])

        lag_24h_col  = np.full((forecast_length, 1), lag_24h, dtype=np.float32)
        lag_168h_col = np.full((forecast_length, 1), lag_168h, dtype=np.float32)

        # Decoder assembly
        decoder_slice = np.concatenate([
            decoder_time_slice,
            decoder_forecast_slice,
            horizon_index
        ], axis=1)

        # Target
        target_slice = df.iloc[enc_end:dec_end]['abvaerk'].values.astype(np.float32)

        encoder_data.append(encoder_slice)
        decoder_data.append(decoder_slice)
        target_data.append(target_slice)

    # =====================================================
    # SAVE
    # =====================================================
    encoder_tensor = torch.from_numpy(np.stack(encoder_data))
    decoder_tensor = torch.from_numpy(np.stack(decoder_data))
    target_tensor  = torch.from_numpy(np.stack(target_data))

    torch.save({
        "encoder": encoder_tensor,
        "decoder": decoder_tensor,
        "target": target_tensor,
        "demand_mean": demand_mean,
        "demand_std": demand_std
    }, output_path)

    _log(f"Saved tensors: {encoder_tensor.shape}")

    return output_path
