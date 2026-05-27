import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import dates
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from MainLibrary import GetData, SavePlot

Data = GetData()

Data = Data[["dateTime", "toutdoor", "treturn", "tforward"]]
Data = Data.sort_values("dateTime").set_index("dateTime")

Data.index = pd.to_datetime(Data.index, format="ISO8601")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(Data.index, Data["treturn"], label="Temperature In", color="steelblue") 
ax.plot(Data.index, Data["tforward"], label="Temperature Out", color="crimson") 
ax.plot(Data.index, Data["toutdoor"], label="Temperature Outside", color="seagreen") 

ax.set_xlabel("Date (YYYY-MM-DD)")
ax.set_ylabel("Temperature(°C)")
ax.set_title("Temperature over Date (Hourly)")
ax.legend()

plt.tight_layout()
SavePlot(fig,"TempDate.png") 
