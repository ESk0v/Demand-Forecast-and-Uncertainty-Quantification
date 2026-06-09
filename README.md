# Demand Forecasting and Uncertainty Quantification

This repository contains code, experiments, datasets, plots, and trained model artifacts for forecasting district heating demand and quantifying forecast uncertainty. The main modelling approach is based on LSTM sequence-to-sequence forecasting with prediction intervals, interval scoring, empirical coverage analysis, and reliability-style evaluation.

The project has evolved through several generations of modelling code. The most current work is concentrated in `LSTMMain.py`, `LSTMModel.py`, and `LSTMTraining.py`, while older pipeline implementations and baseline approaches are kept for comparison and reference.

## Project Structure

| Folder | Purpose |
| --- | --- |
| `Experiments/` | Experiment workspace containing standalone experiment folders, plots, evaluation scripts, and model comparison utilities. These experiments mirror much of the current work and include scripts for coverage, Winkler score, model comparison, and best-week plotting. |
| `FinalLSTM/` | Main current LSTM experimentation area. Contains experiment folders, model variants, trained checkpoints, generated plots, logs, and comparison scripts used for the latest uncertainty-quantification work. |
| `OldModel/` | Archived older version of the full modelling pipeline. It includes dataset creation, LSTM training, hyperparameter tuning, ensemble code, saved models, and older run scripts. Kept mainly for historical reference. |
| `PicturesToSave/` | Collected figures and plots intended for saving or presentation. These are output artifacts rather than source code. |

## Root Files

| File | Purpose |
| --- | --- |
| `Main.py` | Root-level pipeline controller. Supports modes such as dataset creation, hyperparameter tuning, LSTM training, plotting, and ensemble runs. |
| `LSTMMain.py`, `LSTMModel.py`, `LSTMTraining.py` | Root-level LSTM training entrypoint, model definition, and training utilities used by the pipeline controller. |
| `run.sh` | Shell script for running the project pipeline from the command line. |
