import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from MainLibrary import GetData, SavePlot

# Load CSV / data
Data = GetData()

# Convert 'dateTime' to datetime if it isn't already
Data["dateTime"] = pd.to_datetime(Data["dateTime"], errors="coerce")

# Select columns and sort
Data = Data[["dateTime", "abvaerk"]].sort_values("dateTime")

# Set datetime as index
Data = Data.set_index("dateTime")

# Ensure 'abvaerk' is numeric
Data["abvaerk"] = pd.to_numeric(Data["abvaerk"], errors="coerce")

# Create a date-only column
Data["date"] = Data.index.date  # This now works because index is DatetimeIndex

# Group by day and sum
daily_abvaerk = Data.groupby("date")["abvaerk"].sum()

# Plot

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(daily_abvaerk.index, daily_abvaerk.values)

ax.set_xlabel("Date")
ax.set_ylabel("Accumulated abvaerk (hours)")
ax.set_title("Daily Accumulated Abvaerk")
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
SavePlot(fig, "AccumulatedAbvaerk.png")