from EvaluateModel import (
    predictions_vs_actual_plot,
    residuals_plot,
    first_week_prediction_plot,
    first_week_prediction_mc_plot
)

def main():
    """Run evaluation and generate plots for the trained LSTM model."""

    print("\nGenerating plots...")
    residuals_plot(show_plots=True)
    predictions_vs_actual_plot(show_plots=True)
    # We only have the training and validation losses when training the model, so we don't test TrainingValidationPlot()
    first_week_prediction_plot(show_plots=True)
    first_week_prediction_mc_plot(show_plots=True)

if __name__ == '__main__':
    main()