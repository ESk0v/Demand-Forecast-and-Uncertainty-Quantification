"""
CoveragePlot/main.py
────────────────────
Trains N models (each with its own conformal alpha / coverage target)
and saves checkpoints + evaluation plots for each model.
"""

import os
import sys
import random
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# ─── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

MODEL_CONFIGS = [
    dict(
        name            = "Model_3 Coupled",
        module_dir      = "Model_3",
        dataset_path    = "data/dataset.pt",
        model_save_path = "Model_3/checkpoints/model_coupled.pt",
        conformal_alpha = 0.20,   # 1 - 0.80
        epochs          = 5,
        patience        = 5,
        training_variant = "coupled",
    ),
    dict(
        name            = "Model_3 Decoupled",
        module_dir      = "Model_3",
        dataset_path    = "data/dataset.pt",
        model_save_path = "Model_3/checkpoints/model_decoupled.pt",
        conformal_alpha = 0.20,   # 1 - 0.80
        epochs          = 5,
        patience        = 5,
        training_variant = "decoupled",
    )
]


# ─── Minimal logger ───────────────────────────────────────────────────────────
class _SimpleLogger:
    def info(self,    msg): print(f"[INFO]    {msg}")
    def success(self, msg): print(f"[SUCCESS] {msg}")
    def warning(self, msg): print(f"[WARN]    {msg}")
    def error(self,   msg): print(f"[ERROR]   {msg}")

logger = _SimpleLogger()


# ─── Plotting2 adapter ─────────────────────────────────────────────────────────
def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment3_split_indices(n_total, val_ratio=1 / 12, cal_ratio=1 / 12, test_ratio=1 / 6):
    """
    Reproduce Model_3's chronological split:
      train, val, cal, test (val and cal intentionally overlap).
    """
    test_size = int(n_total * test_ratio)
    valcal_size = int(n_total * (val_ratio + cal_ratio))
    train_size = n_total - valcal_size - test_size

    if (val_ratio + cal_ratio) == 0:
        raise ValueError("val_ratio + cal_ratio must be > 0")

    val_size = valcal_size
    cal_size = valcal_size
    return train_size, val_size, cal_size, test_size


def _run_plotting2_for_checkpoint(main_dir, abs_module_dir, dataset_path, model_save_path, model_name):
    """
    Reuse NewModelFolder/LSTM/Plotting2.py on Experiment_3 checkpoints by
    patching in the local Model_3 architecture + split logic.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")

    repo_root = os.path.abspath(os.path.join(main_dir, "..", "..", ".."))
    shared_plotting2_path = os.path.join(repo_root, "NewModelFolder", "LSTM", "Plotting2.py")
    local_model_path = os.path.join(abs_module_dir, "LSTMModel.py")

    if not os.path.isfile(shared_plotting2_path):
        logger.warning(
            f"[{model_name}] Plotting2 source not found at {shared_plotting2_path}. "
            "Skipping extended evaluation plots."
        )
        return

    if not os.path.isfile(local_model_path):
        logger.warning(
            f"[{model_name}] Local LSTMModel.py not found at {local_model_path}. "
            "Skipping extended evaluation plots."
        )
        return

    model_tag = os.path.splitext(os.path.basename(model_save_path))[0]
    plotting_run_dir = os.path.join(
        os.path.dirname(model_save_path),
        "Plotting2",
        model_tag,
    )
    os.makedirs(plotting_run_dir, exist_ok=True)

    plotting_module_name = f"exp3_plotting2_{model_tag}"
    local_model_module_name = f"exp3_model3_{model_tag}"

    plotting2_mod = _load_module_from_file(plotting_module_name, shared_plotting2_path)
    local_model_mod = _load_module_from_file(local_model_module_name, local_model_path)

    # Patch plotting module to use the Experiment_3 architecture and split.
    plotting2_mod.Config = local_model_mod.Config
    plotting2_mod.LSTMForecast = local_model_mod.LSTMForecast
    plotting2_mod._get_split_indices = _experiment3_split_indices

    # Dataset dates for this project start in 2017.
    if hasattr(plotting2_mod, "DATASET_START") and hasattr(plotting2_mod, "pd"):
        plotting2_mod.DATASET_START = plotting2_mod.pd.Timestamp("2017-01-01 00:00")

    plot_dir = os.path.join(plotting_run_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    logger.info(
        f"[{model_name}] Running extended Plotting2 evaluation "
        f"(output: {plot_dir})"
    )

    checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)

    encoder_data = dataset["encoder"]
    decoder_data = dataset["decoder"]
    target_data = dataset["target"]
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)
    n_total = len(full_dataset)

    train_size, val_size, cal_size, test_size = _experiment3_split_indices(n_total)
    valcal_end = train_size + val_size

    # val and cal overlap by design: both use the same val/cal pool.
    cal_start = train_size
    cal_end = valcal_end
    test_start = valcal_end
    test_end = test_start + test_size

    cal_dataset = Subset(full_dataset, range(cal_start, cal_end))
    test_dataset = Subset(full_dataset, range(test_start, test_end))

    config = plotting2_mod.Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = plotting2_mod.LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    q10_test_n, q50_test_n, q90_test_n, targets_test_n = plotting2_mod._run_inference(
        model, test_dataset, config
    )
    q10_cal_n, _, q90_cal_n, targets_cal_n = plotting2_mod._run_inference(
        model, cal_dataset, config
    )

    demand_mean = float(dataset.get("demand_mean", target_data.mean().item()))
    demand_std = float(dataset.get("demand_std", target_data.std().item()))

    def rescale(arr):
        return arr * demand_std + demand_mean

    q10_test_h = rescale(q10_test_n)
    q50_test_h = rescale(q50_test_n)
    q90_test_h = rescale(q90_test_n)
    targets_test_h = rescale(targets_test_n)

    q10_cal_h = rescale(q10_cal_n)
    q90_cal_h = rescale(q90_cal_n)
    targets_cal_h = rescale(targets_cal_n)

    u_alpha = checkpoint.get("conformal_u_alpha", 0.0)
    if isinstance(u_alpha, torch.Tensor):
        u_alpha = u_alpha.detach().cpu().numpy()
    elif isinstance(u_alpha, list):
        u_alpha = np.array(u_alpha)
    u_alpha_mwh = u_alpha * demand_std

    # Forecast windows
    plotting2_mod.plot_forecast_windows(
        q10_test_h,
        q50_test_h,
        q90_test_h,
        targets_test_h,
        test_start,
        os.path.join(plot_dir, "test_predictions.png"),
        u_alpha=u_alpha_mwh,
    )
    logger.success(f"[{model_name}] Saved: test_predictions.png")

    # Weather impact
    with torch.no_grad():
        enc, dec, _ = test_dataset[0]
        enc = enc.unsqueeze(0).to(config.device)
        dec = dec.unsqueeze(0).to(config.device)
        dec_zero = dec.clone()
        dec_zero[:, :, 6:11] = 0
        q10_zero, q50_zero, q90_zero = model(enc, dec_zero)

    plotting2_mod.plot_lstm_weather_impact(
        q10_test_h,
        q50_test_h,
        q90_test_h,
        rescale(q10_zero.detach().cpu().numpy()),
        rescale(q50_zero.detach().cpu().numpy()),
        rescale(q90_zero.detach().cpu().numpy()),
        targets_test_h,
        test_start,
        os.path.join(plot_dir, "weather_impact.png"),
        u_alpha=u_alpha_mwh,
    )
    logger.success(f"[{model_name}] Saved: weather_impact.png")

    # Scatter
    # Plotting2 helpers infer test offset via train + val + cal.
    # Because val/cal overlap here, pass cal=0 to keep the offset correct.
    plotting2_mod.plot_actual_vs_predicted(
        q50_test_h,
        targets_test_h,
        encoder_data,
        train_size,
        val_size,
        0,
        demand_mean,
        demand_std,
        os.path.join(plot_dir, "actual_vs_predicted.png"),
    )
    logger.success(f"[{model_name}] Saved: actual_vs_predicted.png")

    # Residuals
    plotting2_mod.plot_residual_diagnostics(
        q50_test_h,
        targets_test_h,
        test_start,
        os.path.join(plot_dir, "residuals.png"),
    )
    logger.success(f"[{model_name}] Saved: residuals.png")

    # Horizon metrics
    plotting2_mod.plot_per_horizon_metrics(
        q50_test_h,
        targets_test_h,
        encoder_data,
        train_size,
        val_size,
        0,
        demand_mean,
        demand_std,
        os.path.join(plot_dir, "per_horizon_metrics.png"),
    )
    logger.success(f"[{model_name}] Saved: per_horizon_metrics.png")

    # Coverage
    plotting2_mod.plot_quantile_coverage(
        q10_cal_h,
        q90_cal_h,
        targets_cal_h,
        u_alpha_mwh,
        os.path.join(plot_dir, "quantile_coverage.png"),
        q10_test=q10_test_h,
        q90_test=q90_test_h,
        targets_test=targets_test_h,
    )
    logger.success(f"[{model_name}] Saved: quantile_coverage.png")

    # Pinball
    plotting2_mod.plot_pinball_loss(
        q10_test_h,
        q50_test_h,
        q90_test_h,
        targets_test_h,
        os.path.join(plot_dir, "pinball_loss.png"),
    )
    logger.success(f"[{model_name}] Saved: pinball_loss.png")

    # README (if helper exists in the imported plotting module)
    if hasattr(plotting2_mod, "generate_evaluation_readme"):
        plotting2_mod.generate_evaluation_readme(
            plot_dir,
            checkpoint["epoch"],
            checkpoint["val_loss"],
            q50_test_h.shape[0],
            train_size,
            val_size,
            test_size,
            n_total,
            model_filename=os.path.basename(model_save_path),
        )
        logger.success(f"[{model_name}] Saved: README_Evaluation.md")

    logger.success(f"[{model_name}] Plotting2 evaluation complete")


# ─── Training helper ──────────────────────────────────────────────────────────
def train_one_model(cfg: dict):
    """
    Dynamically imports LSTMModel / LSTMTraining from the model-specific
    subfolder and trains the model, saving the checkpoint to disk.
    """
    import importlib

    module_dir     = cfg["module_dir"]
    main_dir       = os.path.abspath(os.path.dirname(__file__))
    abs_module_dir = os.path.abspath(os.path.join(main_dir, module_dir))

    if not os.path.isdir(abs_module_dir):
        raise FileNotFoundError(
            f"Model folder not found: {abs_module_dir}\n"
            f"Expected 'module_dir' to be a subfolder next to main.py."
        )

    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    sys.path.insert(0, abs_module_dir)

    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    lstm_training = importlib.import_module("LSTMTraining")
    lstm_model    = importlib.import_module("LSTMModel")

    load_and_split_dataset = lstm_training.load_and_split_dataset
    train_model            = lstm_training.train_model
    Config                 = lstm_model.Config

    set_seed(GLOBAL_SEED)

    dataset_path    = os.path.join(main_dir, cfg["dataset_path"])
    model_save_path = os.path.join(main_dir, cfg["model_save_path"])

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Place your dataset.pt inside {os.path.join(abs_module_dir, 'data')}"
        )

    (train_dataset, val_dataset, cal_dataset, test_dataset,
     train_size, val_size, cal_size, test_size) = load_and_split_dataset(
        dataset_path
    )

    config        = Config()
    config.epochs = cfg["epochs"]
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers   = min(4, os.cpu_count() or 1)

    def make_loader(ds, shuffle, bs):
        return DataLoader(
            ds,
            batch_size         = bs,
            shuffle            = shuffle,
            pin_memory         = (device == "cuda"),
            num_workers        = num_workers,
            persistent_workers = False,
        )

    train_loader = make_loader(train_dataset, True,  config.batch_size)
    val_loader   = make_loader(val_dataset,   False, config.batch_size * 8)
    cal_loader   = make_loader(cal_dataset,   False, config.batch_size * 8)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    training_variant = cfg.get("training_variant", "coupled")
    logger.info(f"[{cfg['name']}] Starting training … (variant={training_variant})")

    train_model(
        config, train_loader, val_loader, cal_loader,
        train_size, val_size, cal_size,
        model_save_path,
        logger          = logger,
        patience        = cfg["patience"],
        conformal_alpha = cfg["conformal_alpha"],
        training_variant = training_variant,
    )

    logger.success(f"[{cfg['name']}] Checkpoint saved → {model_save_path}")

    try:
        _run_plotting2_for_checkpoint(
            main_dir=main_dir,
            abs_module_dir=abs_module_dir,
            dataset_path=dataset_path,
            model_save_path=model_save_path,
            model_name=cfg["name"],
        )
    except Exception as exc:
        logger.warning(
            f"[{cfg['name']}] Plotting2 evaluation failed: {exc}\n"
            "Training checkpoint is saved; continuing to next model."
        )

    # Clean up imports so the next model gets a fresh load
    if abs_module_dir in sys.path:
        sys.path.remove(abs_module_dir)
    for mod_name in ("LSTMTraining", "LSTMModel"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    set_seed(GLOBAL_SEED)

    for cfg in MODEL_CONFIGS:
        logger.info(f"══ Model: {cfg['name']} ══")
        train_one_model(cfg)

    logger.success("All models trained with per-model training plots and Plotting2 evaluations.")


if __name__ == "__main__":
    main()
