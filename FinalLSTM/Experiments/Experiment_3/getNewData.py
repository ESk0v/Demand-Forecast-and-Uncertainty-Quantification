import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# Observed historical weather (reanalysis, long history)
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Prognostic-style historical weather (forecast-model based archive)
HISTORICAL_FORECAST_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    """Robust z-score based on median and IQR."""
    median = np.nanmedian(values)
    iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
    scale = max(iqr / 1.349, 1e-6)
    return (values - median) / scale


def _repeat_to_length(base_values: np.ndarray, target_length: int) -> np.ndarray:
    repeats = int(np.ceil(target_length / len(base_values)))
    return np.tile(base_values, repeats)[:target_length]


def _extract_http_error_reason(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "reason" in payload:
            return str(payload["reason"])
        return json.dumps(payload)[:300]
    except Exception:
        return response.text[:300]


def fetch_open_meteo_hourly(
    *,
    api_url: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    hourly_params: list[str],
    model_candidates: list[str | None],
    label: str,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch hourly data from Open-Meteo.
    Tries models in order and falls back if a model is unavailable.
    """
    base_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_params),
        "timezone": timezone,
    }

    errors = []
    for model in model_candidates:
        params = base_params.copy()
        model_name = "default" if model is None else model
        if model is not None:
            params["models"] = model

        print(f"Fetching {label} from {api_url} with model={model_name} ...")
        response = requests.get(api_url, params=params, timeout=180)
        if not response.ok:
            reason = _extract_http_error_reason(response)
            errors.append(f"model={model_name}: HTTP {response.status_code} ({reason})")
            print(f"  -> failed ({response.status_code}), trying next model if available.")
            continue

        payload = response.json()
        if "hourly" not in payload or "time" not in payload["hourly"]:
            errors.append(f"model={model_name}: missing 'hourly.time' in response")
            print("  -> missing hourly payload, trying next model if available.")
            continue

        df = pd.DataFrame(payload["hourly"])
        if df.empty:
            errors.append(f"model={model_name}: empty hourly dataframe")
            print("  -> empty dataframe, trying next model if available.")
            continue

        df["time"] = pd.to_datetime(df["time"])
        print(f"  -> success ({len(df)} rows)")
        return df, model_name

    details = "\n".join(errors) if errors else "No attempts were made."
    raise RuntimeError(f"Could not fetch {label} from Open-Meteo.\n{details}")


def load_ringkobing_abvaerk(ringkobing_path: Path) -> np.ndarray:
    if not ringkobing_path.exists():
        raise FileNotFoundError(f"Ringkobing file not found: {ringkobing_path}")

    df = pd.read_csv(ringkobing_path, usecols=["abvaerk"])
    values = pd.to_numeric(df["abvaerk"], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("No valid 'abvaerk' values found in RingkobingData.csv")
    return values


def synthesize_abvaerk_from_ringkobing_and_weather(
    observed_df: pd.DataFrame,
    ringkobing_abvaerk: np.ndarray,
) -> np.ndarray:
    """
    Baseline:
      - Repeat real Ringkøbing abvaerk to 6-year length.

    Weather tweak (deterministic):
      - Increase demand for colder effective temperature (HDD style).
      - Increase demand for wind / gust / humidity / precipitation / cloud.
      - Decrease demand for solar gains.
      - Re-center to keep long-run mean stable vs baseline.
    """
    n = len(observed_df)
    baseline = _repeat_to_length(ringkobing_abvaerk.astype(float), n)

    temp = observed_df["temperature"].to_numpy(dtype=float)
    apparent = observed_df["apparentTemperature"].to_numpy(dtype=float)
    humidity = observed_df["relativeHumidity"].to_numpy(dtype=float)
    precip = observed_df["precipitation"].to_numpy(dtype=float)
    snow = observed_df["snowfall"].to_numpy(dtype=float)
    wind = observed_df["windSpeed"].to_numpy(dtype=float)
    gust = observed_df["windGusts"].to_numpy(dtype=float)
    solar = observed_df["shortwaveRadiation"].to_numpy(dtype=float)
    cloud = observed_df["cloudCover"].to_numpy(dtype=float)
    pressure = observed_df["pressureMsl"].to_numpy(dtype=float)

    gust_excess = np.maximum(gust - wind, 0.0)
    effective_outdoor = apparent - 0.05 * gust_excess
    effective_core = (
        pd.Series(effective_outdoor).ewm(span=48, adjust=False).mean().to_numpy()
    )
    heating_need = np.maximum(0.0, 17.0 - effective_core)

    wet_load = np.log1p(precip * 2.0 + snow * 5.0)

    weather_signal = (
        0.65 * _robust_zscore(heating_need)
        + 0.10 * _robust_zscore(wind)
        + 0.06 * _robust_zscore(gust_excess)
        + 0.07 * _robust_zscore(humidity)
        + 0.06 * _robust_zscore(wet_load)
        + 0.04 * _robust_zscore(cloud)
        - 0.17 * _robust_zscore(solar)
        + 0.03 * _robust_zscore(pressure - np.nanmedian(pressure))
        + 0.04 * _robust_zscore(np.maximum(0.0, 14.0 - temp))
    )

    multiplier = np.clip(1.0 + 0.12 * weather_signal, 0.65, 1.55)
    adjusted = baseline * multiplier

    baseline_mean = float(np.nanmean(baseline))
    adjusted_mean = float(np.nanmean(adjusted))
    if adjusted_mean > 0:
        adjusted *= baseline_mean / adjusted_mean

    return np.maximum(0.0, adjusted)


def fetch_ml_weather_data():
    # Coordinates for Aalborg, Denmark
    latitude = 57.0488
    longitude = 9.9217

    # 6 consecutive years of data
    start_date = "2017-01-01"
    end_date = "2022-12-31"
    timezone = "Europe/Copenhagen"

    # Weather feature set (observed + prognosis)
    weather_params = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "pressure_msl",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
    ]

    # Keep feature names aligned with existing downstream style.
    rename_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "relativeHumidity",
        "dew_point_2m": "dewPoint",
        "apparent_temperature": "apparentTemperature",
        "precipitation": "precipitation",
        "rain": "rain",
        "snowfall": "snowfall",
        "cloud_cover": "cloudCover",
        "cloud_cover_low": "cloudCoverLow",
        "cloud_cover_mid": "cloudCoverMid",
        "cloud_cover_high": "cloudCoverHigh",
        "pressure_msl": "pressureMsl",
        "surface_pressure": "surfacePressure",
        "wind_speed_10m": "windSpeed",
        "wind_direction_10m": "windDirection",
        "wind_gusts_10m": "windGusts",
        "shortwave_radiation": "shortwaveRadiation",
        "direct_radiation": "directRadiation",
        "diffuse_radiation": "diffuseRadiation",
    }

    # Model choices are tried in order; this makes the script robust to model availability.
    observed_model_candidates = ["era5_seamless", "era5", None]
    prognosis_model_candidates = ["ecmwf_ifs", "best_match", None]

    ringkobing_path = Path("../../../NewModelFolder/Files/RingkøbingData.csv")
    output_file = "Aalborg_Weather_2017_2022_Formatted.csv"
    max_horizon = 168  # 168 hours = 1 week

    print(f"Fetching weather for Aalborg from {start_date} to {end_date}...")

    try:
        observed_raw, observed_model_used = fetch_open_meteo_hourly(
            api_url=ARCHIVE_API_URL,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            hourly_params=weather_params,
            model_candidates=observed_model_candidates,
            label="observed weather",
        )

        try:
            prognosis_raw, prognosis_model_used = fetch_open_meteo_hourly(
                api_url=HISTORICAL_FORECAST_API_URL,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                timezone=timezone,
                hourly_params=weather_params,
                model_candidates=prognosis_model_candidates,
                label="prognostic weather",
            )
        except RuntimeError as forecast_err:
            print(
                "Historical-forecast endpoint failed for this full period. "
                "Falling back to archive endpoint for prognosis columns."
            )
            print(f"Reason: {forecast_err}")
            prognosis_raw, prognosis_model_used = fetch_open_meteo_hourly(
                api_url=ARCHIVE_API_URL,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                timezone=timezone,
                hourly_params=weather_params,
                model_candidates=["ecmwf_ifs", "best_match", None],
                label="prognostic weather (archive fallback)",
            )

        observed = observed_raw.rename(columns=rename_map).copy()
        prognosis = prognosis_raw.rename(columns=rename_map).copy()

        # Align time-index intersection to ensure strict row-wise consistency.
        observed = observed.set_index("time")
        prognosis = prognosis.set_index("time")
        common_index = observed.index.intersection(prognosis.index)

        if len(common_index) <= max_horizon:
            raise ValueError(
                f"Insufficient overlapping timestamps after alignment: {len(common_index)}"
            )

        observed = observed.loc[common_index].sort_index().reset_index()
        prognosis = prognosis.loc[common_index].sort_index().reset_index()

        print(
            f"Observed model used: {observed_model_used} | "
            f"Prognosis model used: {prognosis_model_used}"
        )
        print(f"Aligned rows: {len(observed)}")

        print("Loading Ringkøbing baseline and synthesizing abvaerk...")
        ringkobing_abvaerk = load_ringkobing_abvaerk(ringkobing_path)
        observed["abvaerk"] = synthesize_abvaerk_from_ringkobing_and_weather(
            observed,
            ringkobing_abvaerk,
        )
        print("abvaerk synthesis complete.")

        print(
            "Transforming to final format with prognosis horizons "
            "(var_0...var_168) ..."
        )
        structured_data = {}

        # Metadata + target
        structured_data["dateTime"] = observed["time"].iloc[:-max_horizon].values
        structured_data["abvaerk"] = observed["abvaerk"].iloc[:-max_horizon].values

        # Base observed values for quick reference
        structured_data["toutdoor"] = observed["temperature"].iloc[:-max_horizon].values
        structured_data["solarRadiation"] = (
            observed["shortwaveRadiation"].iloc[:-max_horizon].values
        )

        # Prognosis horizon features (decoder future inputs)
        for var in rename_map.values():
            base_series = prognosis[var]
            for h in range(max_horizon + 1):  # 0...168
                col_name = f"{var}_{h}"
                structured_data[col_name] = (
                    base_series.shift(-h).iloc[:-max_horizon].values
                )

        df_final = pd.DataFrame(structured_data)
        df_final.dropna(inplace=True)
        df_final.to_csv(output_file, index=False)

        print("\nData fetched and formatted successfully.")
        print(f"Total rows created: {len(df_final)}")
        print("\nPreview of formatted columns (first 10):")
        print(df_final.iloc[:5, :10])
        print(f"\nSaved formatted data to: {output_file}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


if __name__ == "__main__":
    fetch_ml_weather_data()
