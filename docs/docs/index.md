# Term Deposit Classification Documentation

This project implements a binary classification workflow to predict whether a
client subscribes to a term deposit.

## What This Project Includes

- Data preparation from raw CSV to model-ready train/test data.
- Feature analysis and reusable feature-engineering transformers.
- Model training with class-imbalance handling and hyperparameter search.
- Evaluation using threshold-based and probability-based metrics.
- An interactive Streamlit app (`streamlit_app.py`) for exploring the data and
  trying live predictions without running the pipeline yourself.

## Pipeline Overview

1. Run `deposit_classification/dataset.py` to split and clean data.
2. Run `deposit_classification/plots.py` to generate EDA plots and print analysis tables.
3. Run `deposit_classification/features.py` for feature diagnostics and feature-selection plots.
4. Run `deposit_classification/modeling/train.py` to train and save the model.
5. Run `deposit_classification/modeling/predict.py` to evaluate on the test set.

`models/model_pipeline.joblib` is already trained and committed to the repo,
so steps 1-5 are only needed if you want to reproduce or retrain the model —
see [Getting Started](getting-started.md) for how to jump straight to the demo
app instead.

## Main Artifacts

- Processed data in `data/processed/`
- Model pipeline in `models/model_pipeline.joblib`
- Visual outputs in `reports/figures/` (including EDA plots from `plots.py`)
- Analysis-table outputs printed in the terminal from `plots.py` and `features.py`

