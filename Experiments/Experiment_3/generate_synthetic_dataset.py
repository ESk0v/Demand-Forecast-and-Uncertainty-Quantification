import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

HORIZON = 168
WX_VARS = ["temperature", "relativeHumidity", "windSpeed", "precipitation", "cloudCover"]
BASE_COLS = ["dateTime", "abvaerk", "toutdoor"]
OBS_EXTRA_VARS = ["dewPoint", "surfacePressure", "shortwaveRadiation", "windGust", "windDirection"]


SCENARIO_PROFILES = {
    # Mixed baseline with random parameter jitter (recommended first run).
    "S00_mixed": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.0, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        demand_level=1.0, demand_sens=1.0,
        break_scale=0.55, ramp_prob=0.0012, ramp_amp=2.8,
    ),
    "S01_cold_snap_blocks": dict(
        cold_mult=2.0, mild_mult=0.7, storm_mult=1.1, wet_mult=1.0,
        sigma_mult=1.10, temp_bias=0.45, rh_bias=1.0, wet_bias=0.0,
        demand_level=1.05, demand_sens=1.18,
        break_scale=0.65, ramp_prob=0.0014, ramp_amp=3.0,
    ),
    "S02_warm_front_reversal": dict(
        cold_mult=0.9, mild_mult=1.4, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.12, temp_bias=0.60, rh_bias=1.0, wet_bias=0.05,
        demand_level=0.95, demand_sens=1.05,
        break_scale=0.45, ramp_prob=0.0013, ramp_amp=2.6,
    ),
    "S03_wind_storm_heatloss": dict(
        cold_mult=1.1, mild_mult=0.9, storm_mult=2.0, wet_mult=1.3,
        sigma_mult=1.20, temp_bias=0.40, rh_bias=1.1, wet_bias=0.20,
        demand_level=1.0, demand_sens=1.15,
        break_scale=0.60, ramp_prob=0.0018, ramp_amp=3.4,
    ),
    "S04_wet_atlantic_train": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.4, wet_mult=2.1,
        sigma_mult=1.18, temp_bias=0.35, rh_bias=1.3, wet_bias=0.35,
        demand_level=1.0, demand_sens=1.08,
        break_scale=0.55, ramp_prob=0.0014, ramp_amp=2.9,
    ),
    "S05_shoulder_season_flip": dict(
        cold_mult=1.2, mild_mult=1.2, storm_mult=1.1, wet_mult=1.1,
        sigma_mult=1.15, temp_bias=0.55, rh_bias=1.0, wet_bias=0.1,
        demand_level=1.0, demand_sens=1.10,
        break_scale=0.55, ramp_prob=0.0015, ramp_amp=2.7,
    ),
    "S06_holiday_schedule_shift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.0, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        demand_level=0.92, demand_sens=0.98,
        break_scale=0.50, ramp_prob=0.0012, ramp_amp=2.5,
    ),
    "S07_sensor_drift_toutdoor": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.06, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        demand_level=1.0, demand_sens=1.05,
        break_scale=0.55, ramp_prob=0.0013, ramp_amp=2.8,
    ),
    "S08_forecast_bias_drift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.08, temp_bias=0.95, rh_bias=1.45, wet_bias=0.2,
        demand_level=1.0, demand_sens=1.0,
        break_scale=0.55, ramp_prob=0.0012, ramp_amp=2.8,
    ),
    "S09_structural_break_asset_shift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.03, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        demand_level=1.06, demand_sens=1.22,
        break_scale=1.10, ramp_prob=0.0017, ramp_amp=3.2,
    ),
    "S10_rare_extremes_outliers": dict(
        cold_mult=1.1, mild_mult=0.9, storm_mult=1.3, wet_mult=1.2,
        sigma_mult=1.25, temp_bias=0.5, rh_bias=1.2, wet_bias=0.25,
        demand_level=1.03, demand_sens=1.15,
        break_scale=0.70, ramp_prob=0.0025, ramp_amp=4.2,
    ),
}


REGIME_NAMES = ["normal", "cold", "mild", "storm", "wet", "transition"]
REGIME_DUR_H = np.array([72, 120, 96, 24, 36, 48], dtype=np.float64)
REGIME_WIND_SHIFT = np.array([0.0, 0.8, -0.5, 5.0, 2.0, 0.5], dtype=np.float64)
REGIME_CLOUD_SHIFT = np.array([0.0, 6.0, -10.0, 14.0, 19.0, -2.0], dtype=np.float64)
REGIME_WET_SHIFT = np.array([0.0, 0.05, -0.03, 0.28, 0.55, 0.08], dtype=np.float64)


@dataclass
class SynthConfig:
    start: str = "2020-01-01 00:00:00"
    end: str = "2025-12-31 23:00:00"
    scenario: str = "S00_mixed"
    seed: int = 2026
    output_csv: str = "FinalLSTM/Experiments/Experiment_3/data/synthetic_S00_mixed.csv"
    latitude: float = 56.476
    longitude: float = 8.459
    request_timeout_sec: int = 120


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def temp_climatology(day_of_year, hour):
    day = np.asarray(day_of_year, dtype=np.float64)
    hod = np.asarray(hour, dtype=np.float64)
    seasonal = 8.5 + 10.0 * np.sin(2.0 * np.pi * (day - 205.0) / 365.25)
    amp = 1.8 + 1.6 * np.cos(2.0 * np.pi * (day - 172.0) / 365.25)
    diurnal = amp * np.sin(2.0 * np.pi * (hod - 8.0) / 24.0)
    return seasonal + diurnal


def seasonal_wind(day_of_year):
    day = np.asarray(day_of_year, dtype=np.float64)
    return 1.8 + 1.1 * np.cos(2.0 * np.pi * (day - 10.0) / 365.25)


def seasonal_cloud(day_of_year):
    day = np.asarray(day_of_year, dtype=np.float64)
    return 64.0 + 10.0 * np.cos(2.0 * np.pi * (day - 25.0) / 365.25)


def profile_with_jitter(base_profile, rng):
    profile = dict(base_profile)
    for key in ("sigma_mult", "demand_level", "demand_sens", "temp_bias", "rh_bias"):
        profile[key] = float(profile[key] * np.exp(rng.normal(0.0, 0.06)))
    return profile


def regime_weights(month, profile):
    winter = 1.0 if month in (11, 12, 1, 2, 3) else 0.0
    shoulder = 1.0 if month in (3, 4, 10, 11) else 0.0

    # normal, cold, mild, storm, wet, transition
    w = np.array([
        0.44,
        0.14 + 0.12 * winter,
        0.12 + 0.10 * (1.0 - winter),
        0.08 + 0.03 * winter,
        0.12 + 0.04 * winter,
        0.10 + 0.06 * shoulder,
    ], dtype=np.float64)

    w[1] *= profile["cold_mult"]
    w[2] *= profile["mild_mult"]
    w[3] *= profile["storm_mult"]
    w[4] *= profile["wet_mult"]

    w = np.clip(w, 1e-9, None)
    return w / w.sum()


def generate_regimes(ts, profile, rng):
    n = len(ts)
    regimes = np.zeros(n, dtype=np.int16)
    t = 0
    while t < n:
        month = int(ts[t].month)
        probs = regime_weights(month, profile)
        r = int(rng.choice(len(REGIME_NAMES), p=probs))
        mean_dur = REGIME_DUR_H[r]
        dur = int(max(6, rng.geometric(1.0 / mean_dur)))
        t_end = min(n, t + dur)
        regimes[t:t_end] = r
        t = t_end
    return regimes


def _http_json(url, params, timeout_sec=120, retries=3):
    query = urlencode(params)
    full_url = f"{url}?{query}"
    last_error = None
    for k in range(retries):
        try:
            with urlopen(full_url, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if k + 1 < retries:
                continue
            raise RuntimeError(f"HTTP request failed for {full_url}: {exc}") from exc
    raise RuntimeError(f"HTTP request failed for {full_url}: {last_error}")


def _fetch_open_meteo_hourly(ts, cfg, base_url, hourly_vars):
    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "start_date": ts[0].strftime("%Y-%m-%d"),
        "end_date": ts[-1].strftime("%Y-%m-%d"),
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    payload = _http_json(base_url, params, timeout_sec=cfg.request_timeout_sec)
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"Open-Meteo response missing hourly time for {base_url}.")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df = df.set_index("time").reindex(ts)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.interpolate(limit_direction="both")
    df = df.reset_index().rename(columns={"index": "dateTime"})
    return df


def fetch_open_meteo_observed(ts, cfg):
    # Observed/reanalysis weather (used for target generation and toutdoor).
    # Includes extra drivers used only inside demand simulation.
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "cloud_cover",
        "dew_point_2m",
        "surface_pressure",
        "shortwave_radiation",
        "wind_gusts_10m",
        "wind_direction_10m",
    ]
    df = _fetch_open_meteo_hourly(
        ts,
        cfg,
        base_url="https://archive-api.open-meteo.com/v1/archive",
        hourly_vars=hourly_vars,
    )
    df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relativeHumidity",
            "wind_speed_10m": "windSpeed",
            "precipitation": "precipitation",
            "cloud_cover": "cloudCover",
            "dew_point_2m": "dewPoint",
            "surface_pressure": "surfacePressure",
            "shortwave_radiation": "shortwaveRadiation",
            "wind_gusts_10m": "windGust",
            "wind_direction_10m": "windDirection",
        },
        inplace=True,
    )
    df["windSpeed"] = df["windSpeed"].clip(lower=0.0)
    df["windGust"] = df["windGust"].clip(lower=0.0)
    df["precipitation"] = df["precipitation"].clip(lower=0.0)
    df["cloudCover"] = df["cloudCover"].clip(0.0, 100.0)
    df["relativeHumidity"] = df["relativeHumidity"].clip(20.0, 100.0)
    df["windDirection"] = df["windDirection"].mod(360.0)
    return df[["dateTime", "temperature", "relativeHumidity", "windSpeed", "precipitation", "cloudCover"] + OBS_EXTRA_VARS]


def forecast_row(t, obs_arrays, day_t, hour_t, regime_t, profile, rng):
    h = np.arange(HORIZON, dtype=np.float64)
    future_hour = (hour_t + h.astype(np.int32)) % 24
    future_day = ((day_t - 1 + ((hour_t + h.astype(np.int32)) // 24)) % 365) + 1
    f_ratio = h / (HORIZON - 1)

    temp = obs_arrays["temperature"]
    rh = obs_arrays["relativeHumidity"]
    wind = obs_arrays["windSpeed"]
    precip = obs_arrays["precipitation"]
    cloud = obs_arrays["cloudCover"]

    def past_slope(series, back, cap):
        if t <= 0:
            return 0.0
        k = min(back, t)
        s = (series[t] - series[t - k]) / max(1, k)
        return float(np.clip(s, -cap, cap))

    temp_slope = past_slope(temp, back=6, cap=0.45)
    rh_slope = past_slope(rh, back=8, cap=0.35)
    wind_slope = past_slope(wind, back=6, cap=0.30)
    cloud_slope = past_slope(cloud, back=8, cap=2.00)
    prec_slope = past_slope(precip, back=4, cap=0.20)

    temp_clim = temp_climatology(future_day, future_hour)
    rh_clim = np.clip(82.0 - 0.8 * (temp_clim - 8.0), 25.0, 100.0)
    wind_clim = np.clip(4.5 + seasonal_wind(future_day) + 0.15 * (future_hour >= 12), 0.0, 35.0)
    cloud_clim = np.clip(seasonal_cloud(future_day) + 8.0 * np.sin(2.0 * np.pi * (future_hour - 4.0) / 24.0), 0.0, 100.0)
    prec_clim = np.clip(0.02 + 0.002 * np.maximum(0.0, cloud_clim - 60.0), 0.0, 1.2)

    w_temp = np.exp(-h / 22.0)
    w_rh = np.exp(-h / 34.0)
    w_wind = np.exp(-h / 28.0)
    w_cloud = np.exp(-h / 30.0)
    w_prec = np.exp(-h / 18.0)

    temp_bias = profile["temp_bias"] * np.sin(2.0 * np.pi * future_day / 365.25 + 0.4) * f_ratio
    rh_bias = profile["rh_bias"] * np.sin(2.0 * np.pi * future_hour / 24.0) * f_ratio

    temp_mu = w_temp * (temp[t] + temp_slope * h) + (1.0 - w_temp) * temp_clim + temp_bias
    rh_mu = w_rh * (rh[t] + rh_slope * h) + (1.0 - w_rh) * rh_clim + 0.04 * (cloud[t] - 60.0) + rh_bias
    wind_mu = w_wind * (wind[t] + wind_slope * h) + (1.0 - w_wind) * (wind_clim + 0.06 * REGIME_WIND_SHIFT[regime_t])
    cloud_mu = w_cloud * (cloud[t] + cloud_slope * h) + (1.0 - w_cloud) * (cloud_clim + 0.10 * REGIME_CLOUD_SHIFT[regime_t])
    prec_mu = w_prec * (precip[t] + prec_slope * np.minimum(h, 10.0)) + (1.0 - w_prec) * prec_clim

    temp_sigma = (0.14 + 1.85 * (f_ratio ** 1.25)) * profile["sigma_mult"]
    rh_sigma = (0.35 + 6.4 * (f_ratio ** 1.35)) * profile["sigma_mult"]
    wind_sigma = (0.14 + 2.7 * (f_ratio ** 1.30)) * profile["sigma_mult"]
    cloud_sigma = (1.00 + 14.0 * (f_ratio ** 1.20)) * profile["sigma_mult"]
    prec_sigma = (0.02 + 0.55 * (f_ratio ** 1.50)) * profile["sigma_mult"]

    temp_hat = np.clip(temp_mu + rng.normal(0.0, temp_sigma, HORIZON), -25.0, 35.0)
    rh_hat = np.clip(rh_mu + rng.normal(0.0, rh_sigma, HORIZON), 20.0, 100.0)
    wind_hat = np.clip(wind_mu + rng.normal(0.0, wind_sigma, HORIZON), 0.0, 35.0)
    cloud_hat = np.clip(cloud_mu + rng.normal(0.0, cloud_sigma, HORIZON), 0.0, 100.0)

    wet_term = REGIME_WET_SHIFT[regime_t] + profile["wet_bias"]
    p_occ = sigmoid(-2.8 + 0.030 * cloud_hat + 0.018 * (rh_hat - 70.0) + 0.050 * wind_hat + wet_term)
    is_wet = rng.random(HORIZON) < p_occ
    prec_amt = np.clip(prec_mu + rng.normal(0.0, prec_sigma, HORIZON), 0.0, 20.0)
    prec_dry_noise = np.maximum(0.0, rng.normal(0.0, 0.01 + 0.06 * (f_ratio ** 1.5), HORIZON))
    prec_hat = np.where(is_wet, prec_amt, prec_dry_noise)
    prec_hat = np.clip(prec_hat, 0.0, 20.0)

    # Anchor short lead-time forecasts close to current observed values.
    # This produces realistic low near-term errors and clearer horizon growth.
    temp_hat[0] = np.clip(temp[t] + rng.normal(0.0, 0.12), -25.0, 35.0)
    rh_hat[0] = np.clip(rh[t] + rng.normal(0.0, 0.35), 20.0, 100.0)
    wind_hat[0] = np.clip(wind[t] + rng.normal(0.0, 0.15), 0.0, 35.0)
    prec_hat[0] = np.clip(precip[t] + rng.normal(0.0, 0.02), 0.0, 20.0)
    cloud_hat[0] = np.clip(cloud[t] + rng.normal(0.0, 1.2), 0.0, 100.0)

    return {
        "temperature": temp_hat.astype(np.float32),
        "relativeHumidity": rh_hat.astype(np.float32),
        "windSpeed": wind_hat.astype(np.float32),
        "precipitation": prec_hat.astype(np.float32),
        "cloudCover": cloud_hat.astype(np.float32),
    }


def generate_causal_forecasts(obs_df, ts, regimes, profile, rng):
    n = len(obs_df)
    day = ts.dayofyear.to_numpy(dtype=np.int16)
    hour = ts.hour.to_numpy(dtype=np.int16)
    obs_arrays = {v: obs_df[v].to_numpy(dtype=np.float64) for v in WX_VARS}
    fc = {v: np.empty((n, HORIZON), dtype=np.float32) for v in WX_VARS}

    for t in range(n):
        row = forecast_row(t, obs_arrays, int(day[t]), int(hour[t]), int(regimes[t]), profile, rng)
        for v in WX_VARS:
            fc[v][t, :] = row[v]
    return fc


def simulate_abvaerk(obs_df, ts, regimes, profile, rng):
    n = len(obs_df)
    temp = obs_df["temperature"].to_numpy(dtype=np.float64)
    rh = obs_df["relativeHumidity"].to_numpy(dtype=np.float64)
    wind = obs_df["windSpeed"].to_numpy(dtype=np.float64)
    precip = obs_df["precipitation"].to_numpy(dtype=np.float64)
    cloud = obs_df["cloudCover"].to_numpy(dtype=np.float64)
    dew = obs_df["dewPoint"].to_numpy(dtype=np.float64)
    pressure = obs_df["surfacePressure"].to_numpy(dtype=np.float64)
    swrad = obs_df["shortwaveRadiation"].to_numpy(dtype=np.float64)
    gust = obs_df["windGust"].to_numpy(dtype=np.float64)
    wdir = obs_df["windDirection"].to_numpy(dtype=np.float64)

    dow = ts.dayofweek.to_numpy(dtype=np.int16)
    hod = ts.hour.to_numpy(dtype=np.int16)
    month = ts.month.to_numpy(dtype=np.int16)
    day = ts.day.to_numpy(dtype=np.int16)

    is_holiday = ((month == 12) & (day >= 24)) | ((month == 1) & (day == 1))

    abv = np.zeros(n, dtype=np.float64)
    t_eff = temp[0]
    event_state = 0.0

    for t in range(n):
        # Effective temperature with thermal inertia, wind chill and solar relief.
        t_eff = 0.94 * t_eff + 0.06 * (temp[t] - 0.18 * wind[t] - 0.00035 * swrad[t] + 0.01 * cloud[t])
        hdd = max(0.0, 16.5 - t_eff)

        morning = 1.05 * math.exp(-0.5 * ((hod[t] - 7.0) / 2.2) ** 2)
        evening = 0.95 * math.exp(-0.5 * ((hod[t] - 18.0) / 3.0) ** 2)
        daily_profile = 0.85 + morning + evening

        weekend_mult = 0.94 if dow[t] >= 5 else 1.0
        holiday_mult = 0.90 if is_holiday[t] else 1.0
        cold_factor = 1.0 + 0.12 * (regimes[t] == 1) - 0.05 * (regimes[t] == 2)
        dir_proxy = 0.08 if (wdir[t] >= 220.0 and wdir[t] <= 320.0) else 0.0

        if rng.random() < profile["ramp_prob"]:
            event_state += max(0.0, rng.normal(0.70 * profile["ramp_amp"], 0.60))
        event_state *= 0.94

        base_term = 3.6 * profile["demand_level"] + daily_profile
        weather_term = (
            0.52 * profile["demand_sens"] * cold_factor * hdd
            + 0.05 * (hdd ** 1.15)
            + 0.10 * wind[t]
            + 0.06 * math.sqrt(precip[t] + 1.0)
            + 0.03 * gust[t]
            + 0.015 * max(0.0, rh[t] - 70.0)
            + 0.020 * max(0.0, dew[t] - 2.0)
            + 0.010 * max(0.0, 1015.0 - pressure[t])
            + 0.0010 * cloud[t]
            + dir_proxy
            - 0.00035 * swrad[t]
        )
        noise_std = 0.24 + 0.03 * hdd + 0.12 * float(regimes[t] in (3, 4))

        raw = (base_term + weather_term + event_state) * weekend_mult * holiday_mult + rng.normal(0.0, noise_std)
        ar = 0.58 * (abv[t - 1] if t > 0 else raw) + 0.22 * (abv[t - 24] if t >= 24 else raw)
        abv[t] = np.clip(0.45 * raw + 0.55 * ar, 0.0, 45.0)

    return abv.astype(np.float32)


def assemble_dataframe(ts, obs_df, abvaerk, forecasts):
    data = {
        "dateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "abvaerk": abvaerk,
        "toutdoor": obs_df["toutdoor"].to_numpy(dtype=np.float32),
    }
    for v in WX_VARS:
        mat = forecasts[v]
        for h in range(HORIZON):
            data[f"{v}_{h}"] = mat[:, h]
    out = pd.DataFrame(data)
    return out


def metadata_summary(df):
    keys = ["abvaerk", "toutdoor", "temperature_0", "relativeHumidity_0", "windSpeed_0", "precipitation_0", "cloudCover_0"]
    stats = df[keys].agg(["min", "max", "mean"]).T
    return {
        "date_start": str(df["dateTime"].iloc[0]),
        "date_end": str(df["dateTime"].iloc[-1]),
        "rows": int(len(df)),
        "stats": stats.round(4),
    }


def forecast_error_growth(forecasts, obs_df, prefix):
    truth = obs_df[prefix].to_numpy(dtype=np.float64)
    mat = forecasts[prefix]
    maes = []
    for h in range(HORIZON):
        pred = mat[:, h].astype(np.float64, copy=False)
        if h == 0:
            err = np.abs(pred - truth)
        else:
            err = np.abs(pred[:-h] - truth[h:])
        maes.append(float(np.mean(err)))
    return np.asarray(maes, dtype=np.float64)


def validate_dataset(df, forecasts, obs_df):
    # Time monotonicity / duplicates / hourly cadence
    dt = pd.to_datetime(df["dateTime"], errors="raise")
    if not dt.is_monotonic_increasing:
        raise ValueError("dateTime is not strictly increasing.")
    if dt.duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    step_h = dt.diff().dropna() / pd.Timedelta(hours=1)
    if not np.allclose(step_h.to_numpy(dtype=np.float64), 1.0):
        raise ValueError("Timestamp cadence is not exactly hourly.")

    # Required schema
    required = set(BASE_COLS)
    for v in WX_VARS:
        required.update({f"{v}_{h}" for h in range(HORIZON)})
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns (first 10): {missing[:10]}")

    # Hard constraints
    if df.isna().any().any():
        raise ValueError("NaNs present in final CSV.")
    nonneg_cols = [f"windSpeed_{h}" for h in range(HORIZON)] + [f"precipitation_{h}" for h in range(HORIZON)]
    if (df[nonneg_cols] < 0).any().any():
        raise ValueError("Negative values in windSpeed/precipitation columns.")

    # Forecast error should broadly grow with horizon.
    # Use smoothed curves and a permissive monotonicity condition to avoid
    # false failures while still enforcing "generally increasing with horizon".
    for v in WX_VARS:
        curve = forecast_error_growth(forecasts, obs_df, v)
        smooth = pd.Series(curve).rolling(window=6, min_periods=1, center=True).mean().to_numpy(dtype=np.float64)
        early = float(np.mean(smooth[:24]))
        mid = float(np.mean(smooth[72:96]))
        late = float(np.mean(smooth[-24:]))
        rho = pd.Series(smooth).corr(pd.Series(np.arange(HORIZON)), method="spearman")

        # Main requirement: errors should generally grow with horizon.
        # Use late-vs-early uplift plus positive rank trend.
        ok_growth = (late > early * 1.02)
        ok_rank = (rho > 0.05)

        if not (ok_growth and ok_rank):
            raise ValueError(
                f"Forecast error growth check failed for {v}: "
                f"early={early:.4f}, mid={mid:.4f}, late={late:.4f}, spearman={rho:.3f}"
            )


def validate_feature_availability(obs_df, ts, regimes, profile):
    # Causality test: forecast row at t must not change when future observed values are altered.
    t0 = min(24 * 30, len(obs_df) - 2)
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)

    obs_arrays = {v: obs_df[v].to_numpy(dtype=np.float64) for v in WX_VARS}
    row_a = forecast_row(
        t0,
        obs_arrays,
        int(ts.dayofyear[t0]),
        int(ts.hour[t0]),
        int(regimes[t0]),
        profile,
        rng_a,
    )

    mutated = obs_df.copy()
    for v in WX_VARS:
        mutated[v] = mutated[v].astype(np.float64)
    m_rng = np.random.default_rng(999)
    m = len(mutated) - (t0 + 1)
    mutated.loc[t0 + 1:, "temperature"] = m_rng.normal(5.0, 8.0, m)
    mutated.loc[t0 + 1:, "relativeHumidity"] = np.clip(m_rng.normal(70.0, 18.0, m), 20.0, 100.0)
    mutated.loc[t0 + 1:, "windSpeed"] = np.clip(m_rng.normal(6.0, 4.0, m), 0.0, None)
    mutated.loc[t0 + 1:, "precipitation"] = np.clip(m_rng.normal(0.2, 0.5, m), 0.0, None)
    mutated.loc[t0 + 1:, "cloudCover"] = np.clip(m_rng.normal(60.0, 25.0, m), 0.0, 100.0)

    obs_arrays_b = {v: mutated[v].to_numpy(dtype=np.float64) for v in WX_VARS}
    row_b = forecast_row(
        t0,
        obs_arrays_b,
        int(ts.dayofyear[t0]),
        int(ts.hour[t0]),
        int(regimes[t0]),
        profile,
        rng_b,
    )

    for v in WX_VARS:
        if not np.allclose(row_a[v], row_b[v], atol=1e-9, rtol=0.0):
            raise ValueError(f"Feature-availability causality check failed for {v}.")


def generate_instance(cfg):
    if cfg.scenario not in SCENARIO_PROFILES:
        raise ValueError(f"Unknown scenario: {cfg.scenario}")

    ts = pd.date_range(cfg.start, cfg.end, freq="h")
    if len(ts) < 6 * 365 * 24:
        raise ValueError("Need at least 6 years of hourly data.")

    rng = np.random.default_rng(cfg.seed)
    profile = profile_with_jitter(SCENARIO_PROFILES[cfg.scenario], rng)
    regimes = generate_regimes(ts, profile, rng)

    obs_df = fetch_open_meteo_observed(ts, cfg)
    obs_df["toutdoor"] = obs_df["temperature"] + rng.normal(0.0, 0.12, len(obs_df))
    # Pseudo-forecast generation uses only observed history available up to t
    # and horizon-specific climatology/noise; no realized future values are used.
    validate_feature_availability(obs_df, ts, regimes, profile)
    forecasts = generate_causal_forecasts(obs_df, ts, regimes, profile, rng)
    abvaerk = simulate_abvaerk(obs_df, ts, regimes, profile, rng)

    df = assemble_dataframe(ts, obs_df, abvaerk, forecasts)
    validate_dataset(df, forecasts, obs_df)
    return df


def generate_and_save_single(cfg):
    df = generate_instance(cfg)
    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    meta = metadata_summary(df)
    print(f"Output: {out_path}")
    print(f"Rows: {meta['rows']}")
    print(f"Date range: {meta['date_start']} -> {meta['date_end']}")
    print(meta["stats"])
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Causal synthetic dataset generator for Experiment_3")
    parser.add_argument("--scenario", type=str, default="S00_mixed", choices=sorted(SCENARIO_PROFILES.keys()))
    parser.add_argument("--start", type=str, default="2020-01-01 00:00:00")
    parser.add_argument("--end", type=str, default="2025-12-31 23:00:00")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="FinalLSTM/Experiments/Experiment_3/data/synthetic_S00_mixed.csv")
    parser.add_argument("--request-timeout-sec", type=int, default=120)
    parser.add_argument("--latitude", type=float, default=56.476)
    parser.add_argument("--longitude", type=float, default=8.459)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SynthConfig(
        start=args.start,
        end=args.end,
        scenario=args.scenario,
        seed=args.seed,
        output_csv=args.output,
        latitude=args.latitude,
        longitude=args.longitude,
        request_timeout_sec=args.request_timeout_sec,
    )
    generate_and_save_single(cfg)


if __name__ == "__main__":
    main()
