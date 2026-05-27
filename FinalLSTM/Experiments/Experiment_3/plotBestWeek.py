import importlib.util
import os
import csv
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, TensorDataset


MODEL_DIR = "Model_2_decoupled_no_detach"
CHECKPOINT_REL = os.path.join(MODEL_DIR, "original", "model.pt")
DATASET_REL = os.path.join("data", "dataset.pt")
DATE_SOURCE_CSV_REL = os.path.join("data", "RingkøbingData.csv")
PLOT_REL = os.path.join("compare_plots", "plotBestWeek.png")

VAL_RATIO = 1.0 / 12.0
CAL_RATIO = 1.0 / 12.0
TEST_RATIO = 1.0 / 6.0
ALPHA = 0.10

# Weighted rank-combination for best-week selection.
# Higher weight = metric matters more in the final choice.
WEIGHT_MAE = 1.0
WEIGHT_WINKLER = 1.0
WEIGHT_COVERAGE = 2.5
WEIGHT_COVERAGE_MAD = 2.5

# Hard-coded exclusions from best-week search (test-set row indices).
EXCLUDED_TEST_INDICES = set(range(160, 331)) | set(range(570, 701))


def _abs(path_from_this_file: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path_from_this_file))


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_test_dataset(dataset_path: str):
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    enc = data["encoder"]
    dec = data["decoder"]
    tgt = data["target"]

    n_total = len(tgt)
    test_size = int(n_total * TEST_RATIO)
    valcal_size = int(n_total * (VAL_RATIO + CAL_RATIO))
    train_size = n_total - valcal_size - test_size

    i2 = train_size + valcal_size
    i3 = i2 + test_size

    test_ds = TensorDataset(enc[i2:i3], dec[i2:i3], tgt[i2:i3])
    return data, test_ds, dict(
        n_total=n_total,
        train_size=train_size,
        val_size=valcal_size,
        cal_size=valcal_size,
        test_size=test_size,
        test_start=i2,
    )


def _load_model(checkpoint_path: str):
    model_module_path = _abs(os.path.join(MODEL_DIR, "LSTMModel.py"))
    model_module = _load_module("lstm_model_model2_best_week", model_module_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = model_module.Config()
    ckpt_cfg = checkpoint.get("config")
    if isinstance(ckpt_cfg, dict):
        for key, value in ckpt_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.device = "cpu"

    model = model_module.LSTMForecast(config).to("cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    batch_size = int(getattr(config, "batch_size", 64))
    return model, checkpoint, batch_size


def _predict_test(model, test_loader):
    q10_all, q50_all, q90_all, tgt_all = [], [], [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            q10, q50, q90 = model(enc, dec)
            q10_all.append(q10.cpu().numpy())
            q50_all.append(q50.cpu().numpy())
            q90_all.append(q90.cpu().numpy())
            tgt_all.append(tgt.cpu().numpy())
    return (
        np.concatenate(q10_all, axis=0),
        np.concatenate(q50_all, axis=0),
        np.concatenate(q90_all, axis=0),
        np.concatenate(tgt_all, axis=0),
    )


def _week_winkler(q_low, q_high, target, alpha):
    lo = np.minimum(q_low, q_high)
    hi = np.maximum(q_low, q_high)
    width = hi - lo
    penalty_low = (2.0 / alpha) * np.clip(lo - target, a_min=0.0, a_max=None)
    penalty_high = (2.0 / alpha) * np.clip(target - hi, a_min=0.0, a_max=None)
    return float(np.mean(width + penalty_low + penalty_high))


def _week_coverage_mad(q10, q90, target, alpha):
    target_coverage = (1.0 - alpha) * 100.0
    covered = ((target >= q10) & (target <= q90)).astype(np.float64)
    per_horizon_coverage = covered * 100.0
    return float(np.mean(np.abs(per_horizon_coverage - target_coverage)))


def _week_coverage_percent(q10, q90, target):
    covered = (target >= q10) & (target <= q90)
    return float(np.mean(covered) * 100.0)


def _rank_score(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks.astype(np.float64)


def _load_datetime_series(csv_path: str):
    datetimes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "dateTime" not in reader.fieldnames:
            raise ValueError(f"'dateTime' column not found in: {csv_path}")
        for row in reader:
            datetimes.append(datetime.strptime(row["dateTime"], "%Y-%m-%d %H:%M:%S"))
    return datetimes


def _map_dataset_index_to_datetime(datetimes, dataset_index: int, dataset_total: int):
    if dataset_total <= 1:
        return datetimes[0]
    frac = dataset_index / float(dataset_total - 1)
    csv_idx = int(round(frac * (len(datetimes) - 1)))
    csv_idx = max(0, min(csv_idx, len(datetimes) - 1))
    return datetimes[csv_idx]


def _plot_week(save_path, q10, q50, q90, target, week_idx, metrics, date_start, date_end):
    hours = np.arange(1, len(target) + 1)
    plt.figure(figsize=(14, 6))
    plt.plot(hours, target, color="blue", linewidth=1.5, label="Actual")
    plt.plot(hours, q50, color="red", linewidth=1.5, label="Median (q50)")
    plt.fill_between(hours, q10, q90, color="red", alpha=0.18, label="Uncertainty band")
    plt.title(
        f"Date range: {date_start.strftime('%Y-%m-%d %H:%M')} to {date_end.strftime('%Y-%m-%d %H:%M')}"
    )
    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel("Demand")
    plt.grid(True, alpha=0.3)
    data_legend = plt.legend(loc="upper left")
    metric_handles = [
        Line2D([], [], color="none", label=f"MAE: {metrics['mae']:.4f}"),
        Line2D([], [], color="none", label=f"Winkler: {metrics['winkler']:.4f}"),
        Line2D([], [], color="none", label=f"Coverage: {metrics['coverage']:.2f}%"),
        Line2D([], [], color="none", label=f"Coverage MAD: {metrics['coverage_mad']:.4f}"),
    ]
    plt.gca().add_artist(data_legend)
    plt.legend(
        handles=metric_handles,
        loc="upper right",
        frameon=True,
        title="Metrics",
        handlelength=0,
        handletextpad=0,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    checkpoint_path = _abs(CHECKPOINT_REL)
    dataset_path = _abs(DATASET_REL)
    date_source_csv_path = _abs(DATE_SOURCE_CSV_REL)
    plot_path = _abs(PLOT_REL)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.exists(date_source_csv_path):
        raise FileNotFoundError(f"Date source CSV not found: {date_source_csv_path}")

    model, checkpoint, batch_size = _load_model(checkpoint_path)
    dataset_raw, test_ds, split_info = _split_test_dataset(dataset_path)

    eval_batch_size = max(256, batch_size * 8)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)
    q10, q50, q90, tgt = _predict_test(model, test_loader)

    demand_mean = float(dataset_raw.get("demand_mean", 0.0))
    demand_std = float(dataset_raw.get("demand_std", 1.0))

    q10_h = q10 * demand_std + demand_mean
    q50_h = q50 * demand_std + demand_mean
    q90_h = q90 * demand_std + demand_mean
    tgt_h = tgt * demand_std + demand_mean

    n_weeks = q50_h.shape[0]
    mae_week = np.zeros(n_weeks, dtype=np.float64)
    winkler_week = np.zeros(n_weeks, dtype=np.float64)
    coverage_week = np.zeros(n_weeks, dtype=np.float64)
    coverage_err_week = np.zeros(n_weeks, dtype=np.float64)
    coverage_mad_week = np.zeros(n_weeks, dtype=np.float64)
    target_coverage = (1.0 - ALPHA) * 100.0

    for i in range(n_weeks):
        mae_week[i] = float(np.mean(np.abs(q50_h[i] - tgt_h[i])))
        winkler_week[i] = _week_winkler(q10_h[i], q90_h[i], tgt_h[i], alpha=ALPHA)
        coverage_week[i] = _week_coverage_percent(q10[i], q90[i], tgt[i])
        coverage_err_week[i] = abs(coverage_week[i] - target_coverage)
        coverage_mad_week[i] = _week_coverage_mad(q10[i], q90[i], tgt[i], alpha=ALPHA)

    combined_rank_score = (
        WEIGHT_MAE * _rank_score(mae_week)
        + WEIGHT_WINKLER * _rank_score(winkler_week)
        + WEIGHT_COVERAGE * _rank_score(coverage_err_week)
        + WEIGHT_COVERAGE_MAD * _rank_score(coverage_mad_week)
    )

    eligible_mask = np.ones(n_weeks, dtype=bool)
    for idx in EXCLUDED_TEST_INDICES:
        if 0 <= idx < n_weeks:
            eligible_mask[idx] = False

    if not np.any(eligible_mask):
        raise RuntimeError("No eligible test weeks left after applying exclusions.")

    masked_score = np.where(eligible_mask, combined_rank_score, np.inf)
    best_idx = int(np.argmin(masked_score))
    dataset_idx = split_info["test_start"] + best_idx
    datetimes = _load_datetime_series(date_source_csv_path)
    date_start = _map_dataset_index_to_datetime(
        datetimes=datetimes,
        dataset_index=dataset_idx,
        dataset_total=split_info["n_total"],
    )
    date_end = date_start + timedelta(hours=tgt_h.shape[1] - 1)

    metrics = {
        "mae": float(mae_week[best_idx]),
        "winkler": float(winkler_week[best_idx]),
        "coverage": float(coverage_week[best_idx]),
        "coverage_err": float(coverage_err_week[best_idx]),
        "coverage_mad": float(coverage_mad_week[best_idx]),
    }

    _plot_week(
        plot_path,
        q10=q10_h[best_idx],
        q50=q50_h[best_idx],
        q90=q90_h[best_idx],
        target=tgt_h[best_idx],
        week_idx=best_idx,
        metrics=metrics,
        date_start=date_start,
        date_end=date_end,
    )

    print("Best-week evaluation (Model_2_decoupled_no_detach/original/model.pt)")
    print("------------------------------------------------------------")
    print(f"Excluded test indices: {sorted(EXCLUDED_TEST_INDICES)}")
    print(
        f"Split sizes — train: {split_info['train_size']}  val: {split_info['val_size']}  "
        f"cal: {split_info['cal_size']}  test: {split_info['test_size']}  (total: {split_info['n_total']})"
    )
    print(f"Test week index: {best_idx} (absolute dataset index: {dataset_idx})")
    print(f"Best-week MAE: {metrics['mae']:.6f}")
    print(f"Best-week Mean Winkler Score: {metrics['winkler']:.6f}")
    print(f"Best-week Coverage: {metrics['coverage']:.6f}% (target: {target_coverage:.2f}%)")
    print(f"Best-week Coverage Error |Coverage-Target|: {metrics['coverage_err']:.6f}")
    print(f"Best-week Coverage MAD: {metrics['coverage_mad']:.6f}")
    print(f"Best-week Date range: {date_start} to {date_end}")

    ckpt_alpha = checkpoint.get("conformal_alpha", None)
    if ckpt_alpha is not None:
        print(f"Checkpoint conformal_alpha metadata: {ckpt_alpha}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
