import os
import inspect
import matplotlib.pyplot as plt
from pathlib import Path

def SavePlot(fig, filename, target_folder="Images"):
    """
    Save matplotlib figure inside the project 'Files' folder (where MainLibrary.py lives).

    fig : matplotlib figure
    filename : name of the image file (e.g., 'chart.png')
    target_folder : folder to save inside (default = 'Files')
    """

    # Find MainLibrary.py location
    import MainLibrary
    mainlib_path = os.path.abspath(inspect.getfile(MainLibrary))
    project_root = os.path.dirname(mainlib_path)

    # Build output folder path
    output_dir = os.path.join(project_root, target_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Full save path
    save_path = os.path.join(output_dir, filename)

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")

def GetData(filename="RingkøbingData.csv", start=None, search_folder="Files"):
    from pathlib import Path
    import pandas as pd

    if start is None:
        start = Path(__file__).resolve()

    # walk upward until project root found
    for parent in [start] + list(start.parents):
        candidate = parent / search_folder / filename
        if candidate.exists():
            return pd.read_csv(candidate, on_bad_lines="skip", engine="python")  # <-- return DataFrame

    # fallback: deep search
    project_root = start.parents[-1]
    matches = list(project_root.rglob(filename))
    if matches:
        return pd.read_csv(matches[0])  # <-- return DataFrame

    raise FileNotFoundError(f"Could not find {filename} in project.")
