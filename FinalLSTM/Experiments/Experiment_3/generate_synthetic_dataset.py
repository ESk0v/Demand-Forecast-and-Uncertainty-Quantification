import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 168
WX_VARS = ["temperature", "relativeHumidity", "windSpeed", "precipitation", "cloudCover"]
BASE_COLS = ["dateTime", "abvaerk", "toutdoor"]


SCENARIO_PROFILES = {
    # Mixed baseline with random parameter jitter (recommended first run).
    "S00_mixed": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.0, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        sensor_drift=False, demand_level=1.0, demand_sens=1.0,
        break_scale=0.55, ramp_prob=0.0012, ramp_amp=2.8,
    ),
    "S01_cold_snap_blocks": dict(
        cold_mult=2.0, mild_mult=0.7, storm_mult=1.1, wet_mult=1.0,
        sigma_mult=1.10, temp_bias=0.45, rh_bias=1.0, wet_bias=0.0,
        sensor_drift=False, demand_level=1.05, demand_sens=1.18,
        break_scale=0.65, ramp_prob=0.0014, ramp_amp=3.0,
    ),
    "S02_warm_front_reversal": dict(
        cold_mult=0.9, mild_mult=1.4, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.12, temp_bias=0.60, rh_bias=1.0, wet_bias=0.05,
        sensor_drift=False, demand_level=0.95, demand_sens=1.05,
        break_scale=0.45, ramp_prob=0.0013, ramp_amp=2.6,
    ),
    "S03_wind_storm_heatloss": dict(
        cold_mult=1.1, mild_mult=0.9, storm_mult=2.0, wet_mult=1.3,
        sigma_mult=1.20, temp_bias=0.40, rh_bias=1.1, wet_bias=0.20,
        sensor_drift=False, demand_level=1.0, demand_sens=1.15,
        break_scale=0.60, ramp_prob=0.0018, ramp_amp=3.4,
    ),
    "S04_wet_atlantic_train": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.4, wet_mult=2.1,
        sigma_mult=1.18, temp_bias=0.35, rh_bias=1.3, wet_bias=0.35,
        sensor_drift=False, demand_level=1.0, demand_sens=1.08,
        break_scale=0.55, ramp_prob=0.0014, ramp_amp=2.9,
    ),
    "S05_shoulder_season_flip": dict(
        cold_mult=1.2, mild_mult=1.2, storm_mult=1.1, wet_mult=1.1,
        sigma_mult=1.15, temp_bias=0.55, rh_bias=1.0, wet_bias=0.1,
        sensor_drift=False, demand_level=1.0, demand_sens=1.10,
        break_scale=0.55, ramp_prob=0.0015, ramp_amp=2.7,
    ),
    "S06_holiday_schedule_shift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.0, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        sensor_drift=False, demand_level=0.92, demand_sens=0.98,
        break_scale=0.50, ramp_prob=0.0012, ramp_amp=2.5,
    ),
    "S07_sensor_drift_toutdoor": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.06, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        sensor_drift=True, demand_level=1.0, demand_sens=1.05,
        break_scale=0.55, ramp_prob=0.0013, ramp_amp=2.8,
    ),
    "S08_forecast_bias_drift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.08, temp_bias=0.95, rh_bias=1.45, wet_bias=0.2,
        sensor_drift=False, demand_level=1.0, demand_sens=1.0,
        break_scale=0.55, ramp_prob=0.0012, ramp_amp=2.8,
    ),
    "S09_structural_break_asset_shift": dict(
        cold_mult=1.0, mild_mult=1.0, storm_mult=1.0, wet_mult=1.0,
        sigma_mult=1.03, temp_bias=0.35, rh_bias=1.0, wet_bias=0.0,
        sensor_drift=False, demand_level=1.06, demand_sens=1.22,
        break_scale=1.10, ramp_prob=0.0017, ramp_amp=3.2,
    ),
    "S10_rare_extremes_outliers": dict(
        cold_mult=1.1, mild_mult=0.9, storm_mult=1.3, wet_mult=1.2,
        sigma_mult=1.25, temp_bias=0.5, rh_bias=1.2, wet_bias=0.25,
        sensor_drift=False, demand_level=1.03, demand_sens=1.15,
        break_scale=0.70, ramp_prob=0.0025, ramp_amp=4.2,
    ),
}


REGIME_NAMES = ["normal", "cold", "mild", "storm", "wet", "transition"]
REGIME_DUR_H = np.array([72, 120, 96, 24, 36, 48], dtype=np.float64)
REGIME_TEMP_SHIFT = np.array([0.0, -4.5, 2.8, -1.0, -0.8, 0.4], dtype=np.float64)
REGIME_WIND_SHIFT = np.array([0.0, 0.8, -0.5, 5.0, 2.0, 0.5], dtype=np.float64)
REGIME_CLOUD_SHIFT = np.array([0.0, 6.0, -10.0, 14.0, 19.0, -2.0], dtype=np.float64)
REGIME_WET_SHIFT = np.array([0.0, 0.05, -0.03, 0.28, 0.55, 0.08], dtype=np.float64)


@dataclass
class SynthConfig:
    start: str = "2018-01-01 00:00:00"
    end: str = "2023-12-31 23:00:00"
    scenario: str = "S00_mixed"
    seed: int = 2026
    output_csv: str = "FinalLSTM/Experiments/Experiment_3/data/synthetic_S00_mixed.csv"
    output_dir: str = "FinalLSTM/Experiments/Experiment_3/data"
    use_open_meteo: bool = False
    latitude: float = 56.476
    longitude: float = 8.459


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


def fetch_open_meteo_observed(ts, cfg):
    # Optional path: pull observed weather from Open-Meteo archive.
    import requests

    api = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "start_date": ts[0].strftime("%Y-%m-%d"),
        "end_date": ts[-1].strftime("%Y-%m-%d"),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
        ]),
        "timezone": "UTC",
    }
    response = requests.get(api, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()["hourly"]
    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relativeHumidity",
            "wind_speed_10m": "windSpeed",
            "precipitation": "precipitation",
            "cloud_cover": "cloudCover",
        },
        inplace=True,
    )
    df = df.set_index("time").reindex(ts).interpolate(limit_direction="both").reset_index()
    df.rename(columns={"index": "dateTime"}, inplace=True)
    df["windSpeed"] = df["windSpeed"].clip(lower=0.0)
    df["precipitation"] = df["precipitation"].clip(lower=0.0)
    df["cloudCover"] = df["cloudCover"].clip(0.0, 100.0)
    df["relativeHumidity"] = df["relativeHumidity"].clip(20.0, 100.0)
    return df[["dateTime", "temperature", "relativeHumidity", "windSpeed", "precipitation", "cloudCover"]]


def simulate_observed_weather(ts, regimes, profile, rng):
    n = len(ts)
    day = ts.dayofyear.to_numpy(dtype=np.int16)
    hour = ts.hour.to_numpy(dtype=np.int16)

    temperature = np.zeros(n, dtype=np.float64)
    rh = np.zeros(n, dtype=np.float64)
    wind = np.zeros(n, dtype=np.float64)
    precip = np.zeros(n, dtype=np.float64)
    cloud = np.zeros(n, dtype=np.float64)

    for t in range(n):
        r = regimes[t]
        storm = 1.0 if r == 3 else 0.0
        season_temp = temp_climatology(day[t], hour[t])
        season_wind = seasonal_wind(day[t])
        season_cloud = seasonal_cloud(day[t])

        prev_temp = temperature[t - 1] if t > 0 else season_temp
        prev_wind = wind[t - 1] if t > 0 else max(0.5, 4.5 + season_wind)
        prev_cloud = cloud[t - 1] if t > 0 else np.clip(season_cloud, 0.0, 100.0)
        prev_rh = rh[t - 1] if t > 0 else 78.0

        temp_target = season_temp + REGIME_TEMP_SHIFT[r]
        temperature[t] = 0.92 * prev_temp + 0.08 * temp_target + rng.normal(0.0, 0.65 + 0.35 * storm)
        temperature[t] = np.clip(temperature[t], -23.0, 33.0)

        wind_target = 4.8 + season_wind + REGIME_WIND_SHIFT[r]
        wind[t] = 0.78 * prev_wind + 0.22 * wind_target + rng.normal(0.0, 1.10 + 0.55 * storm)
        wind[t] = np.clip(wind[t], 0.0, 30.0)

        cloud_target = season_cloud + REGIME_CLOUD_SHIFT[r] + 0.45 * (wind[t] - 6.0)
        cloud[t] = 0.82 * prev_cloud + 0.18 * cloud_target + rng.normal(0.0, 8.0 + 2.0 * storm)
        cloud[t] = np.clip(cloud[t], 0.0, 100.0)

        rh_target = 79.0 + 0.35 * (cloud[t] - 60.0) - 0.75 * (temperature[t] - 8.0)
        rh[t] = 0.65 * prev_rh + 0.35 * rh_target + rng.normal(0.0, 5.5)
        rh[t] = np.clip(rh[t], 25.0, 100.0)

        wet_logit = -2.8 + 0.03 * cloud[t] + 0.02 * (rh[t] - 70.0) + 0.05 * wind[t] + REGIME_WET_SHIFT[r]
        if rng.random() < sigmoid(wet_logit):
            p = rng.gamma(shape=1.55, scale=0.55) * (1.0 + 0.05 * max(0.0, wind[t] - 5.0))
            precip[t] = min(18.0, p)
        else:
            precip[t] = 0.0

    sensor_rw = np.cumsum(rng.normal(0.0, 0.0014, n)) if profile["sensor_drift"] else np.zeros(n)

    observed = pd.DataFrame({
        "dateTime": ts,
        "temperature": np.clip(temperature + rng.normal(0.0, 0.09, n), -25.0, 35.0),
        "relativeHumidity": np.clip(rh + rng.normal(0.0, 0.9, n), 20.0, 100.0),
        "windSpeed": np.clip(wind + rng.normal(0.0, 0.22, n), 0.0, None),
        "precipitation": np.clip(precip + rng.normal(0.0, 0.03, n), 0.0, None),
        "cloudCover": np.clip(cloud + rng.normal(0.0, 1.6, n), 0.0, 100.0),
    })
    observed["toutdoor"] = observed["temperature"] + sensor_rw + rng.normal(0.0, 0.15, n)
    return observed


def ar_error_path(rng, sigma, rho):
    sigma = np.asarray(sigma, dtype=np.float64)
    out = np.zeros_like(sigma)
    scale = math.sqrt(max(1e-9, 1.0 - rho * rho))
    for i in range(len(sigma)):
        prev = out[i - 1] if i > 0 else 0.0
        out[i] = rho * prev + rng.normal(0.0, sigma[i] * scale)
    return out


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

    k_temp = min(6, t) if t > 0 else 1
    k_rh = min(6, t) if t > 0 else 1
    k_wind = min(4, t) if t > 0 else 1
    k_cloud = min(6, t) if t > 0 else 1

    temp_slope = (temp[t] - temp[t - k_temp]) / max(1, k_temp) if t > 0 else 0.0
    rh_slope = (rh[t] - rh[t - k_rh]) / max(1, k_rh) if t > 0 else 0.0
    wind_slope = (np.log1p(wind[t]) - np.log1p(wind[t - k_wind])) / max(1, k_wind) if t > 0 else 0.0
    cloud_slope = (cloud[t] - cloud[t - k_cloud]) / max(1, k_cloud) if t > 0 else 0.0

    temp_clim = temp_climatology(future_day, future_hour)
    w_temp = np.exp(-h / 20.0)
    temp_mu = w_temp * (temp[t] + temp_slope * h) + (1.0 - w_temp) * temp_clim
    temp_bias = profile["temp_bias"] * np.sin(2.0 * np.pi * future_day / 365.25 + 0.4) * f_ratio
    temp_sigma = (0.30 + 1.55 * (f_ratio ** 1.25)) * profile["sigma_mult"]
    temp_hat = np.clip(temp_mu + temp_bias + ar_error_path(rng, temp_sigma, rho=0.78), -25.0, 35.0)

    rh_clim = np.clip(82.0 - 0.8 * (temp_clim - 8.0), 25.0, 100.0)
    w_rh = np.exp(-h / 26.0)
    rh_mu = w_rh * (rh[t] + rh_slope * h) + (1.0 - w_rh) * rh_clim + 0.10 * (cloud[t] - 60.0)
    rh_bias = profile["rh_bias"] * np.sin(2.0 * np.pi * future_hour / 24.0) * f_ratio
    rh_sigma = (1.8 + 7.2 * (f_ratio ** 1.15)) * profile["sigma_mult"]
    rh_hat = np.clip(rh_mu + rh_bias + ar_error_path(rng, rh_sigma, rho=0.72), 20.0, 100.0)

    wind_clim = np.log1p(np.clip(5.0 + seasonal_wind(future_day) + 0.15 * (future_hour >= 12), 0.0, 35.0))
    w_wind = np.exp(-h / 16.0)
    wind_mu = (
        w_wind * (np.log1p(wind[t]) + wind_slope * h)
        + (1.0 - w_wind) * wind_clim
        + 0.06 * REGIME_WIND_SHIFT[regime_t]
    )
    wind_sigma = (0.08 + 0.30 * (f_ratio ** 1.20)) * profile["sigma_mult"]
    wind_hat = np.expm1(wind_mu + ar_error_path(rng, wind_sigma, rho=0.67))
    wind_hat = np.clip(wind_hat, 0.0, 35.0)

    cloud_clim = np.clip(seasonal_cloud(future_day) + 8.0 * np.sin(2.0 * np.pi * (future_hour - 4.0) / 24.0), 0.0, 100.0)
    w_cloud = np.exp(-h / 18.0)
    cloud_mu = w_cloud * (cloud[t] + cloud_slope * h) + (1.0 - w_cloud) * (cloud_clim + REGIME_CLOUD_SHIFT[regime_t])
    cloud_sigma = (3.0 + 11.0 * (f_ratio ** 1.1)) * profile["sigma_mult"]
    cloud_hat = np.clip(cloud_mu + ar_error_path(rng, cloud_sigma, rho=0.70), 0.0, 100.0)

    wet_term = REGIME_WET_SHIFT[regime_t] + profile["wet_bias"]
    p_occ = sigmoid(-3.1 + 0.032 * cloud_hat + 0.020 * (rh_hat - 70.0) + 0.055 * wind_hat + wet_term)
    int_mu = np.log1p(0.05 + 0.004 * np.maximum(0.0, cloud_hat - 55.0) + 0.03 * np.maximum(0.0, wind_hat - 6.0))
    int_sigma = (0.20 + 0.50 * (f_ratio ** 1.3)) * profile["sigma_mult"]
    intensity = np.expm1(int_mu + ar_error_path(rng, int_sigma, rho=0.55))
    intensity = np.clip(intensity, 0.0, 20.0)
    prec_hat = p_occ * intensity + np.maximum(0.0, rng.normal(0.0, 0.01 + 0.06 * (f_ratio ** 1.5), HORIZON))
    prec_hat = np.clip(prec_hat, 0.0, 20.0)

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
    wind = obs_df["windSpeed"].to_numpy(dtype=np.float64)
    precip = obs_df["precipitation"].to_numpy(dtype=np.float64)
    cloud = obs_df["cloudCover"].to_numpy(dtype=np.float64)

    dow = ts.dayofweek.to_numpy(dtype=np.int16)
    hod = ts.hour.to_numpy(dtype=np.int16)
    month = ts.month.to_numpy(dtype=np.int16)
    day = ts.day.to_numpy(dtype=np.int16)

    is_holiday = ((month == 12) & (day >= 24)) | ((month == 1) & (day == 1))

    abv = np.zeros(n, dtype=np.float64)
    t_eff = temp[0]
    event_state = 0.0

    break_points = sorted(rng.choice(np.arange(24 * 120, n - 24 * 120), size=3, replace=False).tolist())
    base_shift = np.zeros(n, dtype=np.float64)
    sens_shift = np.zeros(n, dtype=np.float64)
    for bp in break_points:
        base_shift[bp:] += rng.normal(0.0, profile["break_scale"])
        sens_shift[bp:] += rng.normal(0.0, 0.035 * profile["break_scale"])

    for t in range(n):
        solar_proxy = max(0.0, math.sin(2.0 * math.pi * (hod[t] - 6.0) / 24.0)) * (1.0 - cloud[t] / 100.0) * 450.0
        t_eff = 0.93 * t_eff + 0.07 * (temp[t] - 0.22 * wind[t] - 0.006 * solar_proxy + 0.03 * cloud[t] / 100.0)
        hdd = max(0.0, 16.5 - t_eff)

        morning = 1.25 * math.exp(-0.5 * ((hod[t] - 7.0) / 2.2) ** 2)
        evening = 1.05 * math.exp(-0.5 * ((hod[t] - 18.0) / 3.0) ** 2)
        daily_profile = 0.9 + morning + evening

        weekend_mult = 0.93 if dow[t] >= 5 else 1.0
        holiday_mult = 0.88 if is_holiday[t] else 1.0
        cold_factor = 1.0 + 0.15 * (regimes[t] == 1)

        if rng.random() < profile["ramp_prob"]:
            event_state += max(0.0, rng.normal(profile["ramp_amp"], 0.9))
        event_state *= 0.93

        beta_hdd = 0.45 * profile["demand_sens"] + sens_shift[t]
        beta_hdd2 = 0.07 * profile["demand_sens"]
        base_term = 3.8 * profile["demand_level"] + base_shift[t] + daily_profile
        weather_term = beta_hdd * cold_factor * hdd + beta_hdd2 * (hdd ** 1.2) + 0.19 * wind[t] + 0.08 * math.sqrt(precip[t] + 1.0) - 0.0009 * solar_proxy
        noise_std = 0.30 + 0.04 * hdd + 0.25 * float(regimes[t] in (3, 4))

        raw = (base_term + weather_term + event_state) * weekend_mult * holiday_mult + rng.normal(0.0, noise_std)
        ar = 0.55 * (abv[t - 1] if t > 0 else raw) + 0.20 * (abv[t - 24] if t >= 24 else raw)
        abv[t] = np.clip(0.45 * raw + 0.55 * ar, 0.0, 45.0)

    return abv.astype(np.float32)


def assemble_dataframe(ts, obs_df, abvaerk, forecasts):
    out = pd.DataFrame({
        "dateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "abvaerk": abvaerk,
        "toutdoor": obs_df["toutdoor"].to_numpy(dtype=np.float32),
    })
    for v in WX_VARS:
        mat = forecasts[v]
        for h in range(HORIZON):
            out[f"{v}_{h}"] = mat[:, h]
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

    # Forecast error should broadly grow with horizon
    for v in WX_VARS:
        curve = forecast_error_growth(forecasts, obs_df, v)
        rho = pd.Series(curve).corr(pd.Series(np.arange(HORIZON)), method="spearman")
        if rho < 0.60:
            raise ValueError(f"Forecast error growth check failed for {v}: Spearman rho={rho:.3f}")


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

    if cfg.use_open_meteo:
        obs_core = fetch_open_meteo_observed(ts, cfg)
        obs_df = obs_core.copy()
        obs_df["toutdoor"] = obs_df["temperature"] + rng.normal(0.0, 0.12, len(obs_df))
    else:
        obs_df = simulate_observed_weather(ts, regimes, profile, rng)

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


def generate_and_save_all(cfg):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, scenario in enumerate(sorted(SCENARIO_PROFILES.keys())):
        per_cfg = SynthConfig(
            start=cfg.start,
            end=cfg.end,
            scenario=scenario,
            seed=cfg.seed + i * 31,
            output_csv=str(out_dir / f"synthetic_{scenario}.csv"),
            output_dir=cfg.output_dir,
            use_open_meteo=cfg.use_open_meteo,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
        )
        saved.append(generate_and_save_single(per_cfg))
    print(f"Generated {len(saved)} scenario files in: {out_dir}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Causal synthetic dataset generator for Experiment_3")
    parser.add_argument("--scenario", type=str, default="S00_mixed", choices=sorted(SCENARIO_PROFILES.keys()))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--start", type=str, default="2018-01-01 00:00:00")
    parser.add_argument("--end", type=str, default="2023-12-31 23:00:00")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="FinalLSTM/Experiments/Experiment_3/data/synthetic_S00_mixed.csv")
    parser.add_argument("--output-dir", type=str, default="FinalLSTM/Experiments/Experiment_3/data")
    parser.add_argument("--use-open-meteo", action="store_true")
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
        output_dir=args.output_dir,
        use_open_meteo=args.use_open_meteo,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    if args.all_scenarios:
        generate_and_save_all(cfg)
    else:
        generate_and_save_single(cfg)


if __name__ == "__main__":
    main()
