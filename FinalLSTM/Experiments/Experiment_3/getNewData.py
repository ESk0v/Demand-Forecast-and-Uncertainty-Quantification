import requests
import pandas as pd
import numpy as np

# Open-Meteo Historical Weather API (ERA5 Reanalysis)
# This serves perfectly as both "historical observations" and "historical prognoses" (reanalysis)
# as it estimates the atmospheric features for past timestamps using physical models based on actual observations.
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_ml_weather_data():
    # Coordinates for Aalborg, Denmark
    latitude = 57.0488
    longitude = 9.9217

    # 6 consecutive years of data
    start_date = "2017-01-01"
    end_date = "2022-12-31"

    # Hourly parameters to request from Open-Meteo
    # Added shortwave_radiation for the physical synthesis of heating demand
    hourly_params = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "shortwave_radiation"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_params),
        "timezone": "Europe/Copenhagen"
    }

    print(f"Fetching {end_date[:4]} historical weather data for Aalborg...")

    try:
        response = requests.get(ARCHIVE_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to pandas DataFrame
        df_raw = pd.DataFrame(data["hourly"])
        df_raw['time'] = pd.to_datetime(df_raw['time'])

        # Rename base columns to match desired prefixes
        rename_map = {
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'relativeHumidity',
            'precipitation': 'precipitation',
            'cloud_cover': 'cloudCover',
            'wind_speed_10m': 'windSpeed',
            'shortwave_radiation': 'shortwave_radiation'
        }
        df_raw.rename(columns=rename_map, inplace=True)

        print("Synthesizing highly realistic 'abvaerk' (District Heating Demand) using physics and heuristics...")

        # --- PHYSICS-BASED ABVAERK SYNTHESIS ---
        temp = df_raw['temperature'].values
        wind = df_raw['windSpeed'].values
        solar = df_raw['shortwave_radiation'].values

        # Wind chills the building
        wind_chill_effect = wind * 0.1

        # Sun warms the building (reduced effect compared to before)
        passive_solar = solar * 0.003

        # Instantaneous effective external temperature
        T_instant = temp - wind_chill_effect + passive_solar

        # KEY FIX: Buildings have thermal mass (inertia)! They do not cool down or heat up instantly.
        # We apply an exponential moving average (approx 36-hour span) to simulate building heat retention.
        T_inertia = pd.Series(T_instant).ewm(span=36).mean().values

        # Blend: heating systems respond slightly to current weather, but mostly to the building's core temperature
        T_effective = 0.8 * T_inertia + 0.2 * T_instant

        # Space heating demand (thermostats kick in when effective temp is below ~16C)
        space_heating = np.maximum(0, 16.0 - T_effective)

        # Scaled down the MW multiplier so the temperature delta doesn't cause massive structural swings
        space_heating_MW = space_heating * 12.0

        # Human Heuristics (Base load for hot tap water, showers, industries)
        # Increased base load so the city doesn't drop to 25 MW
        hour_of_day = df_raw['time'].dt.hour.values
        day_of_week = df_raw['time'].dt.dayofweek.values

        # Morning peak (7 AM), evening activity (6 PM) - scaled realistically
        daily_profile = 50.0 + 12.0 * np.exp(-0.5 * ((hour_of_day - 7) / 2.0) ** 2) + 8.0 * np.exp(
            -0.5 * ((hour_of_day - 18) / 3.0) ** 2)

        # Weekends have slightly lower overall demand
        weekend_penalty = np.where(day_of_week >= 5, 0.90, 1.0)

        # Add some true Gaussian chaos/noise
        np.random.seed(42)
        noise = np.random.normal(0, 3.0, len(temp))

        # Final synthesized target feature
        abvaerk_synth = (space_heating_MW + daily_profile) * weekend_penalty + noise
        df_raw['abvaerk'] = np.maximum(20.0, abvaerk_synth)  # Floor constraint
        print("Synthesis complete.")

        print("Transforming data to match 'RingkøbingData.csv' format (creating forecast horizons up to t+168)...")

        max_horizon = 168  # 168 hours = 1 week future lookahead

        # Create a new dictionary to hold our shaped data
        structured_data = {}

        # Base dateTime
        structured_data['dateTime'] = df_raw['time'].iloc[:-max_horizon].values

        # Our newly synthesized Target
        structured_data['abvaerk'] = df_raw['abvaerk'].iloc[:-max_horizon].values

        # Base historical observations
        structured_data['toutdoor'] = df_raw['temperature'].iloc[:-max_horizon].values
        structured_data['solarRadiation'] = df_raw['shortwave_radiation'].iloc[:-max_horizon].values

        # Add the rolling horizons for each variable
        for var in rename_map.values():
            for h in range(max_horizon + 1):  # 0 to 168
                # We shift the data backwards to look into the future for each row
                col_name = f"{var}_{h}"
                structured_data[col_name] = df_raw[var].shift(-h).iloc[:-max_horizon].values

        # Convert to final DataFrame
        df_final = pd.DataFrame(structured_data)

        # Drop any remaining NaN rows at the very end
        df_final.dropna(inplace=True)

        output_file = "Aalborg_Weather_2017_2022_Formatted.csv"
        df_final.to_csv(output_file, index=False)

        print("\nData fetched and formatted successfully!")
        print(f"Total rows created: {len(df_final)}")
        print("\nPreview of formatted columns (showing first few):")
        print(df_final.iloc[:5, :10])
        print(f"\nSaved formatted data to: {output_file}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        print(f"Other error occurred: {err}")


if __name__ == "__main__":
    fetch_ml_weather_data()