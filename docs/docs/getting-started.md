# Getting Started

## Prerequisites

- Python 3.12
- A virtual environment tool (`venv` is sufficient)

## Installation

Create and activate your virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Try The Demo App First

`models/model_pipeline.joblib` is already trained and committed, so you can
launch the interactive demo without running the pipeline at all:

```bash
streamlit run streamlit_app.py
```

This works from the repository root or its parent directory. See the
[project README](https://github.com/cameroncochrane/Term-Deposit-Classification#streamlit-demo-app)
for a tour of what each tab shows.

## Data

Place the source dataset at:

`data/raw/term-deposit-marketing-2020.csv`

This is only needed to reproduce the pipeline below, or to unlock the demo
app's data-exploration and live model-performance tabs — the CSV is
gitignored, so it isn't included in a fresh clone.

## Run The Pipeline

Execute the project in this order from the repository root:

```bash
python deposit_classification/dataset.py
python deposit_classification/plots.py
python deposit_classification/features.py
python deposit_classification/modeling/train.py
python deposit_classification/modeling/predict.py
```

## Expected Outputs

- Train/test pickles in `data/interim/` and `data/processed/`
- EDA plots in `reports/figures/` from `deposit_classification/plots.py`
- Analysis tables printed to terminal from `deposit_classification/plots.py` and
	`deposit_classification/features.py`
- Trained model in `models/model_pipeline.joblib`
- Generated feature plots in `reports/figures/`

## Build Documentation

```bash
mkdocs build -f docs/mkdocs.yml
mkdocs serve -f docs/mkdocs.yml
```
