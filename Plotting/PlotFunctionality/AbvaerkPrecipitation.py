import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from MainLibrary import GetData, SavePlot

# === LOAD DATA ===
Data = GetData()
Data = Data[["abvaerk", "precipitation_0", "dateTime"]].dropna()

# Convert comma decimals to dots and floats just in case
Data["abvaerk"] = pd.to_numeric(Data["abvaerk"].astype(str).str.replace(",", "."), errors='coerce')
Data["precipitation_0"] = pd.to_numeric(Data["precipitation_0"].astype(str).str.replace(",", "."), errors='coerce')
print(f"Loaded {len(Data)} rows of data")

# === SCATTER PLOT: Precipitation vs Abvaerk ===
correlation = np.corrcoef(Data['abvaerk'], Data['precipitation_0'])[0, 1]
fig_scatter = plt.figure(figsize=(10, 7))
plt.scatter(Data['precipitation_0'], Data['abvaerk'], alpha=0.3, s=10)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Energy Production (MWh)')
plt.title(f'Precipitation vs Energy Production (r = {correlation:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
SavePlot(fig_scatter, "Abvaerk_Precipitation_Correlation.png")
print(f"Correlation: {correlation:.3f}")

# === WEEKLY AVERAGE PLOT ===
df = Data.copy()
df['datetime_parsed'] = pd.to_datetime(df['dateTime'], format='%d/%m/%Y %H.%M')
df['week'] = df['datetime_parsed'].dt.to_period('W')

weekly = df.groupby('week').agg({
    'abvaerk': 'mean',
    'precipitation_0': 'mean'
}).reset_index()

weekly['week_str'] = weekly['week'].astype(str)

fig_weekly, ax1 = plt.subplots(figsize=(14, 6))

# Energy usage
ax1.plot(weekly['week_str'], weekly['abvaerk'], color='red', linewidth=2, marker='o', markersize=3, label='Avg Energy Production (MWh)')
ax1.set_xlabel('Week')
ax1.set_ylabel('Avg Energy Production (MWh)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticks(ax1.get_xticks()[::10])
plt.xticks(rotation=45)

# Precipitation on secondary axis
ax2 = ax1.twinx()
ax2.plot(weekly['week_str'], weekly['precipitation_0'], color='blue', linewidth=2, marker='s', markersize=3, label='Avg Precipitation (mm)')
ax2.set_ylabel('Avg Precipitation (mm)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

weekly_corr = np.corrcoef(weekly['abvaerk'], weekly['precipitation_0'])[0, 1]
plt.title(f'Weekly Average: Energy Production vs Precipitation (r = {weekly_corr:.3f})')
fig_weekly.tight_layout()
SavePlot(fig_weekly, "Abvaerk_Precipitation_WeeklyAverage.png")
print(f"Weekly correlation: {weekly_corr:.3f}")