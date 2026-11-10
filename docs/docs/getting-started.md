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

The demo's Try a Prediction tab calls the already-deployed prediction API, so you can launch
it without training anything (it does need an internet connection):

```bash
streamlit run streamlit_app.py
```

This works from the repository root or its parent directory. See the
[project README](https://github.com/cameroncochrane/Term-Deposit-Classification#streamlit-demo-app)
for a tour of what each tab shows.

## Data

Place the source dataset at:

`data/raw/term-deposit-marketing-2020.csv`

This is needed to reproduce the pipeline below, and by the demo app's
data-exploration and model-performance tabs (they show a message if it's missing).

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
- Trained model in `models/model_pipeline_2.joblib` (local output; the promoted model served by
  the live API is `app/models/model_pipeline_2.joblib`)
- Generated feature plots in `reports/figures/`

## Build Documentation

```bash
mkdocs build -f docs/mkdocs.yml
mkdocs serve -f docs/mkdocs.yml
```
