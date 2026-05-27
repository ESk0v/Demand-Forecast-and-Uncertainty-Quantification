import sys
import os
from sklearn import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from pathlib import Path

from .EnsembleHelpers import _EnsemblePredict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LSTM")))
from GenerateREADME import generate_ensemble_readme


def _EvaluateModel(test_loader, models, device, demand_mean, demand_std, n_models, plot_dir):
    """
    Runs ensemble evaluation with:
    - Mean prediction
    - Total uncertainty
    - Epistemic uncertainty
    - Aleatoric uncertainty

    Only targets are rescaled back to original units (MW),
    since predictions are already in MW.
    """

    meanPredictions = []
    stdPredictions = []
    epistemicPredictions = []
    aleatoricPredictions = []
    targets = []

    # -----------------------------
    # Run Ensemble on Test Set
    # -----------------------------
    for enc, dec, tgt in test_loader:
        enc, dec = enc.to(device).float(), dec.to(device).float()

        mean_real, total_std_real, epistemic_real, aleatoric_real = \
            _EnsemblePredict(models, enc, dec, demand_mean, demand_std)

        meanPredictions.append(mean_real.cpu())
        stdPredictions.append(total_std_real.cpu())
        epistemicPredictions.append(epistemic_real.cpu())
        aleatoricPredictions.append(aleatoric_real.cpu())
        targets.append(tgt)

    # Concatenate batches
    meanPredictions = torch.cat(meanPredictions, dim=0).numpy().flatten()
    stdPredictions = torch.cat(stdPredictions, dim=0).numpy().flatten()
    epistemicPredictions = torch.cat(epistemicPredictions, dim=0).numpy().flatten()
    aleatoricPredictions = torch.cat(aleatoricPredictions, dim=0).numpy().flatten()
    targets = torch.cat(targets, dim=0).numpy().flatten()


    # -----------------------------
    # RESCALE ONLY TARGETS
    # -----------------------------
    targets_rescaled = targets * demand_std + demand_mean

    # -----------------------------
    # Select ONE WEEK of data
    # -----------------------------
    week_index = 1
    start = week_index * 168
    end = min(start + 168, len(meanPredictions))

    mean_preds_week = meanPredictions[start:end]
    std_preds_week = stdPredictions[start:end]
    targets_week = targets_rescaled[start:end]
    aleatoric_week = aleatoricPredictions[start:end]
    epistemic_week = epistemicPredictions[start:end]

    time_steps = np.arange(len(mean_preds_week))

    # -----------------------------
    # Generate Plots
    # -----------------------------
    generated_plots = _GeneratePlots(
        time_steps,
        targets_week,
        mean_preds_week,
        std_preds_week,
        week_index,
        meanPredictions,
        stdPredictions,
        targets_rescaled,
        epistemicPredictions,
        aleatoricPredictions,
        plot_dir=plot_dir,
        epistemic_week=epistemic_week,
        aleatoric_week=aleatoric_week
    )

    # -----------------------------
    # Generate Ensemble README
    # -----------------------------
    generate_ensemble_readme(
        plot_dir=str(plot_dir),
        ensemble_size=n_models,
        model_filename="model.pth",
        generated_plots=generated_plots,
    )


# ============================================================
# MASTER PLOT FUNCTION
# ============================================================

def _GeneratePlots(
    time_steps,
    targets_week,
    mean_preds_week,
    std_preds_week,
    week_index,
    meanPredictions,
    stdPredictions,
    targets,
    epistemicPredictions,
    aleatoricPredictions,
    plot_dir,
    epistemic_week,
    aleatoric_week,
):
    generated = []
    generated.append(_PlotWeekForecast(time_steps, targets_week, mean_preds_week, std_preds_week, epistemic_week, aleatoric_week, week_index, plot_dir))
    generated.append(_PlotCalibration(meanPredictions, stdPredictions, targets, plot_dir))
    generated.append(_PlotSigmaCoverage(meanPredictions, stdPredictions, targets, plot_dir))
    generated.append(_PlotUncertaintyDecomposition(time_steps, epistemic_week, aleatoric_week, plot_dir))
    generated.append(_PlotBinnedErrorVsUncertainty(meanPredictions, stdPredictions, targets, plot_dir))
    #generated.append(_PlotStandardizedResiduals(meanPredictions, stdPredictions, targets, plot_dir))
    #generated.append(_PlotUncertaintyHistogram(stdPredictions, plot_dir))
    return [f for f in generated if f is not None]

# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

# Week forecast with uncertainty
def _PlotWeekForecast(
    time_steps,
    targets_week,
    mean_preds_week,
    std_preds_week,
    epistemic_week,
    aleatoric_week,
    week_index,
    plot_dir
):
    """
    Plots one-week forecast showing epistemic and total uncertainty.

    Epistemic band = model uncertainty
    Total band = epistemic + aleatoric
    """

    total_std = std_preds_week
    epistemic_std = epistemic_week

    plt.figure(figsize=(14,5))

    # Actual data
    plt.plot(time_steps, targets_week, label="Actual abvaerk", linewidth=2)

    # Mean prediction
    plt.plot(time_steps, mean_preds_week, label="Predicted mean", linewidth=2)

    # Total uncertainty band
    plt.fill_between(
        time_steps,
        mean_preds_week - 2 * total_std,
        mean_preds_week + 2 * total_std,
        alpha=0.2,
        label="±2 Total Uncertainty"
    )

    # Epistemic uncertainty band
    plt.fill_between(
        time_steps,
        mean_preds_week - 2 * epistemic_std,
        mean_preds_week + 2 * epistemic_std,
        alpha=0.4,
        label="±2 Epistemic Uncertainty"
    )

    plt.xlabel("Time Steps")
    plt.ylabel("abvaerk (MW)")
    plt.title(f"Ensemble LSTM Forecast with Uncertainty (Week {week_index+1})")

    plt.legend()
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_path = plot_dir / f"week_forecast_uncertainty_week{week_index+1}.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name

# Calibration Plot
def _PlotCalibration(meanPredictions, stdPredictions, targets, plot_dir):
    """
    Checks empirical coverage of prediction intervals.
    All inputs are in MW.
    """

    z_values = np.linspace(0.5, 3.0, 20)
    empirical_coverages = []

    for z in z_values:
        lower = meanPredictions - z * stdPredictions
        upper = meanPredictions + z * stdPredictions

        coverage = np.mean((targets >= lower) & (targets <= upper))
        empirical_coverages.append(coverage)

    theoretical_coverages = 2 * (0.5 * (1 + erf(z_values / np.sqrt(2)))) - 1

    plt.figure(figsize=(6,6))
    plt.plot(theoretical_coverages, empirical_coverages, marker="o")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Theoretical Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Curve")
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "calibration_curve.png"

    plt.savefig(save_path)
    plt.close()
    return save_path.name

def _PlotSigmaCoverage(meanPredictions, stdPredictions, targets, plot_dir):
    """
    Plots coverage as a function of sigma (z-value).

    X-axis: sigma (0–3)
    Y-axis: coverage probability

    Shows:
        - empirical coverage
        - theoretical Gaussian coverage
    """

    z_values = np.arange(0.0, 3.5, 0.5)
    empirical_coverages = []

    for z in z_values:
        lower = meanPredictions - z * stdPredictions
        upper = meanPredictions + z * stdPredictions

        coverage = np.mean((targets >= lower) & (targets <= upper))
        empirical_coverages.append(coverage)

    theoretical_coverages = 2 * (0.5 * (1 + erf(z_values / np.sqrt(2)))) - 1

    plt.figure(figsize=(6,6))

    plt.plot(z_values, empirical_coverages, marker="o", label="Empirical")
    plt.plot(z_values, theoretical_coverages, linestyle="--", label="Theoretical")

    plt.xlabel("Sigma (Standard Deviations)")
    plt.ylabel("Coverage Probability")
    plt.title("Coverage vs Sigma")

    plt.xticks(z_values)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "sigma_coverage_curve.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name

def _PlotUncertaintyDecomposition(time_steps, epistemic_week, aleatoric_week, plot_dir):

    plt.figure(figsize=(12,5))

    plt.plot(time_steps, epistemic_week, label="Epistemic uncertainty")
    plt.plot(time_steps, aleatoric_week, label="Aleatoric uncertainty")

    plt.xlabel("Time Steps")
    plt.ylabel("Standard deviation (MW)")
    plt.title("Uncertainty Decomposition (Epistemic vs Aleatoric)")

    plt.legend()
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "uncertainty_decomposition.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name

def _PlotBinnedErrorVsUncertainty(meanPredictions, stdPredictions, targets, plot_dir, bins=10):

    abs_errors = np.abs(meanPredictions - targets)

    bin_edges = np.linspace(stdPredictions.min(), stdPredictions.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    avg_errors = []

    for i in range(bins):
        mask = (stdPredictions >= bin_edges[i]) & (stdPredictions < bin_edges[i+1])

        if np.any(mask):
            avg_errors.append(np.mean(abs_errors[mask]))
        else:
            avg_errors.append(np.nan)

    plt.figure(figsize=(6,6))

    plt.plot(bin_centers, avg_errors, marker="o")

    plt.xlabel("Predicted Uncertainty (Std, MW)")
    plt.ylabel("Average Absolute Error (MW)")
    plt.title("Binned Error vs Predicted Uncertainty")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "binned_error_vs_uncertainty.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name

def _PlotStandardizedResiduals(meanPredictions, stdPredictions, targets, plot_dir):

    residuals = (targets - meanPredictions) / stdPredictions

    plt.figure(figsize=(6,6))

    plt.hist(residuals, bins=50, density=True)

    x = np.linspace(-4,4,200)
    normal = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

    plt.plot(x, normal)

    plt.xlabel("Standardized Residual")
    plt.ylabel("Density")
    plt.title("Standardized Residual Distribution")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "standardized_residuals.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name

def _PlotUncertaintyHistogram(stdPredictions, plot_dir):

    plt.figure(figsize=(6,6))

    plt.hist(stdPredictions, bins=40)

    plt.xlabel("Predicted Uncertainty (Std, MW)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predictive Uncertainty")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "uncertainty_histogram.png"

    plt.savefig(save_path)
    plt.close()

    return save_path.name
