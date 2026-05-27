import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from MainLibrary import SavePlot, GetData

CSV_FILE_PATH = '../../NewModelFolder/Files/RingkøbingData.csv'

def get_data(filepath=CSV_FILE_PATH):
    # """Load and clean the CSV data."""
    # # Count total lines in file (excluding header)
    # with open(filepath, 'r') as f:
    #     total_lines = sum(1 for _ in f) - 1  # -1 for header
    #
    # # Read CSV with detected delimiter, skip bad lines
    # df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip', low_memory=False)
    #
    # # Calculate and log skipped lines
    # skipped_lines = total_lines - len(df)
    # if skipped_lines > 0:
    #     print(f"Warning: {skipped_lines} lines were skipped due to parsing errors")

    df = GetData()

    # Convert comma decimals to dots for abvaerk and convert to float
    df['abvaerk'] = pd.to_numeric(df['abvaerk'].astype(str).str.replace(',', '.'), errors='coerce')

    # Convert comma decimals for cloudCover_0 (the one we need for plotting)
    df['cloudCover_0'] = pd.to_numeric(df['cloudCover_0'].astype(str).str.replace(',', '.'), errors='coerce')

    # Drop rows with NaN values in the columns we need
    df = df.dropna(subset=['abvaerk', 'cloudCover_0', 'dateTime'])

    print(f"Loaded {len(df)} rows of data")
    return df


def plot_correlation(df):
    """Scatter plot showing correlation between cloud cover and energy usage."""
    correlation = np.corrcoef(df['abvaerk'], df['cloudCover_0'])[0, 1]

    fig = plt.figure(figsize=(10, 7))
    plt.scatter(df['cloudCover_0'], df['abvaerk'], alpha=0.3, s=10)
    plt.xlabel('Cloud Cover (%)')
    plt.ylabel('Energy Usage (MWh)')
    plt.title(f'Cloud Cover vs Energy Usage (r = {correlation:.3f})')
    plt.grid(True, alpha=0.3)
    # plt.show()

    print(f"Correlation: {correlation:.3f}")
    return fig


def plot_weekly_average(df):
    """Plot weekly averages of energy usage and cloud cover to see seasonal patterns."""
    df = df.copy()
    df['datetime_parsed'] = pd.to_datetime(df['dateTime'], format='%d/%m/%Y %H.%M')
    df['week'] = df['datetime_parsed'].dt.to_period('W')

    # Group by week and calculate averages
    weekly = df.groupby('week').agg({
        'abvaerk': 'mean',
        'cloudCover_0': 'mean'
    }).reset_index()

    weekly['week_str'] = weekly['week'].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot energy usage
    ax1.plot(weekly['week_str'], weekly['abvaerk'], color='red', linewidth=2, marker='o', markersize=3, label='Avg Energy Usage (MWh)')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Avg Energy Usage (MWh)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Show fewer x-axis labels to avoid crowding
    ax1.set_xticks(ax1.get_xticks()[::10])
    plt.xticks(rotation=45)

    # Plot cloud cover on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(weekly['week_str'], weekly['cloudCover_0'], color='blue', linewidth=2, marker='s', markersize=3, label='Avg Cloud Cover (%)')
    ax2.set_ylabel('Avg Cloud Cover (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 100)

    # Calculate weekly correlation
    correlation = np.corrcoef(weekly['abvaerk'], weekly['cloudCover_0'])[0, 1]

    plt.title(f'Weekly Average: Energy Usage vs Cloud Cover (r = {correlation:.3f})')
    fig.tight_layout()
    # plt.show()

    print(f"Weekly correlation: {correlation:.3f}")
    return fig


# Run when script is executed directly
if __name__ == "__main__":
    df = get_data()

    # Plot weekly averages to see seasonal correlation
    fig1 = plot_weekly_average(df)
    fig2 = plot_correlation(df)

    # Save the plot
    SavePlot(fig1, "Abvaerk_CloudCover_WeeklyAverage.png")
    SavePlot(fig2, "Abvaerk_CloudCover_Correlation.png", "Images/Correlation")