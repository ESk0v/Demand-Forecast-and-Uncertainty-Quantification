import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ForecastDataset(Dataset):
    """PyTorch Dataset wrapping feature sequences and scalar targets."""

    def __init__(self, samples, targets):
        self.samples = torch.FloatTensor(samples)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

def load_dataset(filepath, verbose=True):
    """
    Load, validate, and clean the dataset from a CSV file.

    The CSV is expected to contain:
      - 'dateTime'                                  : timestamp (not used as a feature)
      - 'abvaerk'                                   : target column (energy in MWh)
      - 'cloudCover_0'       .. 'cloudCover_167'    : 168-hour weather forecast
      - 'precipitation_0'    .. 'precipitation_167' : 168-hour weather forecast
      - 'relativeHumidity_0' .. 'relativeHumidity_167'
      - 'temperature_0'      .. 'temperature_167'
      - 'windSpeed_0'        .. 'windSpeed_167'

    Each row becomes one sample. The five forecast groups are stacked so that
    samples has shape (n_samples, 168, 5) — seq_len=168, n_features=5.

    Returns:
        samples  : np.ndarray of shape (n_samples, 168, 5)
        targets  : np.ndarray of shape (n_samples,)
        metadata : dict with 'feature_names', 'n_features', 'n_samples'
    """
    df = pd.read_csv(filepath)

    SEQ_LEN = 168
    feature_groups = ['cloudCover', 'precipitation', 'relativeHumidity', 'temperature', 'windSpeed']

    # Build (n_samples, 168, 5) by stacking each weather-variable group
    feature_arrays = []
    for group in feature_groups:
        cols = [f"{group}_{i}" for i in range(SEQ_LEN)]
        feature_arrays.append(df[cols].values)  # (n_samples, 168)

    # Stack along last axis → (n_samples, 168, 5)
    samples = np.stack(feature_arrays, axis=2).astype(np.float32)
    targets = df['abvaerk'].values.astype(np.float32)

    if verbose:
        # --- Data quality report ---
        print("\nData quality check:")
        print(f"  Samples NaN count : {np.isnan(samples).sum()}")
        print(f"  Samples Inf count : {np.isinf(samples).sum()}")
        print(f"  Targets NaN count : {np.isnan(targets).sum()}")
        print(f"  Targets Inf count : {np.isinf(targets).sum()}")
        print(f"  Samples range     : [{np.nanmin(samples):.4f}, {np.nanmax(samples):.4f}]")
        print(f"  Targets range     : [{np.nanmin(targets):.4f}, {np.nanmax(targets):.4f}]")

    # --- Remove invalid rows ---
    valid_mask = ~np.isnan(targets) & ~np.isinf(targets)
    for i in range(samples.shape[2]):
        feature_valid = ~np.any(np.isnan(samples[:, :, i]) | np.isinf(samples[:, :, i]), axis=1)
        valid_mask &= feature_valid

    samples = samples[valid_mask]
    targets = targets[valid_mask]

    metadata = {
        'feature_names': feature_groups,
        'n_features'   : len(feature_groups),
        'n_samples'    : len(targets),
    }

    if verbose:
        # --- Summary ---
        print(f"\nLoaded dataset:")
        print(f"  Samples shape : {samples.shape}")
        print(f"  Targets shape : {targets.shape}")
        print(f"  Features      : {metadata['feature_names']}")
        print(f"  Removed       : {(~valid_mask).sum()} invalid samples")

    return samples, targets, metadata


def normalize_data(X_train, X_val, y_train, y_val):
    """
    Fit scalers on training data and apply to both train and validation sets.

    Reshapes internally to handle the 3D sample array (n_samples, seq_len, n_features).

    Returns:
        X_train_scaled, X_val_scaled   : normalized feature arrays
        y_train_scaled, y_val_scaled   : normalized target arrays
        feature_scaler, target_scaler  : fitted StandardScaler instances
    """
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]

    # Fit scalers on training data only, then apply to both splits
    feature_scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train_scaled = feature_scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled   = feature_scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)

    target_scaler  = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled   = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, feature_scaler, target_scaler