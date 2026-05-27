import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from MainLibrary import GetData, SavePlot

# =========================
# LOAD DATA
# =========================
Data = GetData()
Data = Data[["abvaerk", "windSpeed_0", "dateTime"]].dropna()

# Convert numeric columns
Data["abvaerk"] = pd.to_numeric(Data["abvaerk"].astype(str).str.replace(",", "."), errors="coerce")
Data["windSpeed_0"] = pd.to_numeric(Data["windSpeed_0"].astype(str).str.replace(",", "."), errors="coerce")

# Robust datetime parse (fixes your error)
Data["dt"] = pd.to_datetime(Data["dateTime"], errors="coerce")

Data = Data.dropna()

print(f"Loaded {len(Data)} rows")

# =========================
# EXTRACT HOUR OF DAY
# =========================
Data["hour"] = Data["dt"].dt.hour

# =========================
# WIND SPEED BINS (for smooth lines)
# =========================
bins = np.linspace(
    Data["windSpeed_0"].min(),
    Data["windSpeed_0"].max(),
    40
)

Data["wind_bin"] = pd.cut(Data["windSpeed_0"], bins)

# Average per (hour, wind_bin)
grouped = (
    Data.groupby(["hour", "wind_bin"])
        .agg({
            "abvaerk": "mean",
            "windSpeed_0": "mean"
        })
        .reset_index()
)

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(14, 7))

# Choose which hours to highlight
highlight_hours = [6, 12, 18, 22]   # <-- change these anytime

for hour in range(24):

    sub = grouped[grouped["hour"] == hour]

    if hour in highlight_hours:
        ax.plot(
            sub["windSpeed_0"],
            sub["abvaerk"],
            linewidth=3,
            label=f"Hour {hour}",
            zorder=5
        )
    else:
        ax.plot(
            sub["windSpeed_0"],
            sub["abvaerk"],
            linewidth=1,
            alpha=0.25,
            color="gray",
            zorder=1
        )

# =========================
# LABELS
# =========================
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Average Energy Production (MWh)")
ax.set_title("Energy vs Wind Speed — 24 Hour-of-Day Profiles")

ax.grid(True, linestyle="--", alpha=0.4)

# Legend only shows highlighted hours
ax.legend(title="Highlighted Hours")

plt.tight_layout()

SavePlot(fig, "WindSpeed_Abvaerk_24HourProfiles.png")
plt.show()
