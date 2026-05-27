import pathlib
import sys
from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from MainLibrary import GetData, SavePlot

# Load data
Data = GetData()

# Keep only needed columns
Data = Data[["dateTime", "abvaerk"]]
Data = Data.sort_values("dateTime").set_index("dateTime")

# Convert index to datetime
Data.index = pd.to_datetime(Data.index)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(14, 6))

# White background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot all data
ax.plot(Data.index, Data["abvaerk"], color="blue", linewidth=1)

# Titles and labels
ax.set_title("Abvaerk vs Dates (Hourly)")
ax.set_xlabel("Dates")
ax.set_ylabel("Abvaerk")

# Set a few x-axis ticks only (8 ticks roughly quarterly)
num_ticks = 8
tick_positions = np.linspace(0, len(Data.index)-1, num_ticks, dtype=int)
tick_labels = Data.index[tick_positions]

ax.set_xticks(tick_labels)
ax.set_xticklabels([d.strftime("%Y-%m") for d in tick_labels], rotation=45)

# Grid and layout
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
SavePlot(fig, "AbvaerkTimePlot.png")
