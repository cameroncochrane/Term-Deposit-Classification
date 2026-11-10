# Streamlit showcase for the Term Deposit Classification project.

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from joblib import load
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Resolve paths relative to this file so the app works no matter which
# directory it's launched from.
PROJ_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJ_ROOT / "data" / "raw" / "term-deposit-marketing-2020.csv"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
GITHUB_URL = "https://github.com/cameroncochrane/Term-Deposit-Classification"

# The deployed prediction API (see app/README.md). Override with the TERM_DEPOSIT_API_URL
# environment variable, e.g. to point at a local `uvicorn api.main:app` while developing.
DEFAULT_API_URL = "https://aalr9ksiel.execute-api.eu-west-2.amazonaws.com"
API_URL = os.environ.get("TERM_DEPOSIT_API_URL", DEFAULT_API_URL).rstrip("/")
# Generous because a Lambda that has gone cold can take a few seconds to answer.
API_TIMEOUT_SECONDS = 30

# The repo's copy of the model the API serves. Only used by the Model Performance tab: scoring
# thousands of test rows through the live API would be slow and would flood the prediction log
# that the drift monitor reads. The Try a Prediction tab never touches this file.
DEPLOYED_MODEL_PATH = PROJ_ROOT / "app" / "models" / "model_pipeline_2.joblib"

# Valid inputs for the API's /predict endpoint. Mirrors app/api/schemas.py (PredictionInput) -
# the model only ever sees these five fields. Note "unknown" jobs are rejected by the API.
JOB_VALUES = [
    "admin", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed",
]
YES_NO_VALUES = ["no", "yes"]
MONTH_VALUES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


st.set_page_config(
    page_title="Term Deposit Classification",
    page_icon="\U0001F4B0",
    layout="wide",
)


class ApiError(Exception):
    """A failure talking to the prediction API, with a message that's safe to show the user."""


def _call_api(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, f"{API_URL}{path}", timeout=API_TIMEOUT_SECONDS, **kwargs)
    except requests.Timeout:
        raise ApiError(
            f"No response from {API_URL} within {API_TIMEOUT_SECONDS}s. "
            "A cold-started Lambda can be slow - try again."
        ) from None
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach {API_URL}: {exc}") from exc


def check_api_health() -> float:
    """Return the /health round-trip time in seconds, or raise ApiError."""
    response = _call_api("GET", "/health")
    if response.status_code != 200 or response.json().get("status") != "ok":
        raise ApiError(f"/health returned HTTP {response.status_code}: {response.text[:200]}")
    return response.elapsed.total_seconds()


def predict_via_api(payload: dict) -> tuple[dict, float]:
    """POST `payload` to /predict. Returns (response JSON, round-trip seconds) or raises ApiError."""
    response = _call_api("POST", "/predict", json=payload)
    if response.status_code == 422:
        raise ApiError(f"The API rejected the input: {response.json().get('detail')}")
    if response.status_code != 200:
        raise ApiError(f"/predict returned HTTP {response.status_code}: {response.text[:200]}")
    return response.json(), response.elapsed.total_seconds()


@st.cache_resource(show_spinner="Loading the deployed model...")
def load_deployed_model():
    # The deployed pipeline was pickled against api.transformers (see app/api/service.py), so
    # that module has to be importable before joblib can unpickle it.
    app_dir = str(PROJ_ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import api.transformers  # noqa: F401

    return load(DEPLOYED_MODEL_PATH)


@st.cache_data(show_spinner="Loading raw dataset...")
def load_raw_data():
    if not RAW_DATA_PATH.exists():
        return None
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner="Reconstructing held-out test split...")
def build_eval_split(raw_df: pd.DataFrame):
    """Mirror the split used at training time (test_size=0.3, random_state=13,
    stratified on y), then drop rows with an unknown job/education, matching
    the row-filtering `deposit_classification/dataset.py` applies before
    training. Contact imputation is skipped here since the trained pipeline
    discards the `contact` column entirely, so it can't affect predictions.
    """
    X = raw_df.drop(columns="y")
    y = raw_df["y"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=13, stratify=y
    )

    known_mask = (X_test["job"] != "unknown") & (X_test["education"] != "unknown")
    X_test = X_test.loc[known_mask].reset_index(drop=True)
    y_test = y_test.loc[known_mask].reset_index(drop=True)
    y_test = y_test.map({"no": 0, "yes": 1}).astype("int8")

    return X_test, y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Matthews Corrcoef": matthews_corrcoef(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Log Loss": log_loss(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


raw_df = load_raw_data()

st.title("Term Deposit Classification")
st.caption(
    "Predicting whether a bank customer subscribes to a term deposit, based on "
    f"the outcome of a marketing campaign. [View the repo on GitHub]({GITHUB_URL})."
)

tab_overview, tab_explore, tab_predict, tab_performance = st.tabs(
    ["Overview", "Explore the Data", "Try a Prediction", "Model Performance"]
)

with tab_overview:
    st.header("Project Overview")
    st.markdown(
        """
This project builds an end-to-end machine learning pipeline for binary
classification of term-deposit subscriptions, using a bank telemarketing
dataset of ~40,000 customer contacts with a **~7.2% positive class rate**.

**Pipeline stages** (see `deposit_classification/`):

1. **Ingestion & cleaning** (`dataset.py`) — train/test split, dropping rows
   with missing job/education, and mode-based imputation of `contact`.
2. **Feature analysis** (`features.py`) — chi-squared, Cramer's V, and mutual
   information tests to rank categorical features; Pearson correlation for
   numeric features.
3. **Feature engineering** — a `FeatureDropper` removes low-signal columns
   (`age`, `balance`, `campaign`, `default`, `loan`, `contact`, `education`,
   `marital`), leaving just `duration`, `day`, `month`, `housing` and `job`,
   and a `CyclicalDateEncoder` converts `day`/`month` into sine/cosine pairs.
4. **Modeling** (`modeling/train.py`) — an imbalanced-learn pipeline
   (impute → scale/encode → `RandomOverSampler` → `XGBClassifier`), tuned via
   `RandomizedSearchCV` (540 iterations, 5-fold stratified CV, optimizing F1).
5. **Evaluation** (`modeling/predict.py`) — accuracy, precision/recall, F1,
   MCC, ROC AUC, and log loss on the held-out test set.

**Deployed system** (see `app/`):

`Streamlit` → `API Gateway` → `Lambda (FastAPI + model)` → `S3 prediction log`
→ daily `Evidently` drift check → retrain → pull request → CI deploy on merge.

The **Try a Prediction** tab calls the live API rather than loading a model
locally, so it always reflects whichever model is currently in production.

Use the tabs above to explore the raw data, try a live prediction, or review
the model's held-out performance.
        """
    )
    st.info(
        "**Caveat:** `duration` (call length) is the strongest predictor available, "
        "but it's only known *after* a call ends — it can't be used to decide who to "
        "call in advance. It's kept here for demonstration purposes."
    )

with tab_explore:
    st.header("Explore the Data")
    if raw_df is None:
        st.warning(
            f"Raw dataset not found at `{RAW_DATA_PATH.relative_to(PROJ_ROOT)}`. "
            "Add the CSV there to enable this view."
        )
    else:
        st.subheader("Sample rows")
        st.dataframe(raw_df.head(20), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class balance")
            st.bar_chart(raw_df["y"].value_counts())
        with col2:
            st.subheader("Subscription rate by feature")
            cat_cols = ["job", "marital", "education", "housing", "loan", "contact", "month"]
            selected_col = st.selectbox("Group by", cat_cols)
            rate = (
                raw_df.assign(subscribed=(raw_df["y"] == "yes").astype(int))
                .groupby(selected_col)["subscribed"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(rate)

        st.subheader("EDA figures")
        figure_captions = {
            "categorical_value_counts.png": "Value counts for categorical features.",
            "pearson_matrix.png": "Pearson correlation between numeric features and the target.",
            "numeric_columns_boxplots.png": "Distribution of numeric features.",
            "numeric_vs_target_boxplots.png": "Numeric features split by subscription outcome.",
        }
        cols = st.columns(2)
        for i, (filename, caption) in enumerate(figure_captions.items()):
            figure_path = FIGURES_DIR / filename
            if figure_path.exists():
                cols[i % 2].image(str(figure_path), caption=caption, width="stretch")

with tab_predict:
    st.header("Try a Prediction")
    st.write(
        "Enter the details of a contact and the app sends them to the **live prediction API**, "
        "which scores them with whichever model is currently in production."
    )
    st.caption(
        f"Endpoint: `{API_URL}`. The model only uses these five fields — the other columns in "
        "the dataset were dropped during feature selection. Predictions made here are logged like "
        "any other live request, so they feed the drift monitor."
    )

    if st.button("Check API status"):
        with st.spinner("Contacting the API..."):
            try:
                latency = check_api_health()
                st.success(f"API is healthy ({latency:.2f}s round trip).")
            except ApiError as exc:
                st.error(str(exc))

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            job = st.selectbox("Job", JOB_VALUES)
            housing = st.radio("Has housing loan?", YES_NO_VALUES, horizontal=True)
        with col2:
            month = st.selectbox("Last contact month", MONTH_VALUES, index=4)
            day = st.slider("Last contact day of month", 1, 31, 15)
        with col3:
            duration = st.number_input(
                "Last contact duration (seconds)", min_value=0, value=180, step=10
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "duration": int(duration),
            "day": int(day),
            "month": month,
            "housing": housing,
            "job": job,
        }

        with st.spinner("Calling the prediction API..."):
            try:
                result, latency = predict_via_api(payload)
            except ApiError as exc:
                st.error(str(exc))
                result = None

        if result is not None:
            probability = float(result["probability"])

            result_col, gauge_col = st.columns(2)
            with result_col:
                if result["prediction"] == 1:
                    st.success(f"Predicted outcome: **subscribes** (probability {probability:.1%})")
                else:
                    st.error(
                        f"Predicted outcome: **does not subscribe** (probability {probability:.1%})"
                    )
            with gauge_col:
                st.metric("Predicted subscription probability", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))

            with st.expander(f"Request and response ({latency:.2f}s round trip)"):
                st.write(f"`POST {API_URL}/predict`")
                st.json({"request": payload, "response": result})

with tab_performance:
    st.header("Model Performance (Held-Out Test Set)")
    if raw_df is None:
        st.warning(
            f"Raw dataset not found at `{RAW_DATA_PATH.relative_to(PROJ_ROOT)}`. "
            "Add the CSV there to enable this view."
        )
    elif not DEPLOYED_MODEL_PATH.exists():
        st.warning(
            f"Deployed model not found at `{DEPLOYED_MODEL_PATH.relative_to(PROJ_ROOT)}`."
        )
    else:
        model = load_deployed_model()
        X_test, y_test = build_eval_split(raw_df)
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test)

        st.caption(
            "Scored locally with the repo's copy of the deployed model "
            f"(`{DEPLOYED_MODEL_PATH.relative_to(PROJ_ROOT)}`) rather than through the live API - "
            "that would mean thousands of requests and a flooded prediction log. "
            "Pull `main` after merging a retrain PR to keep this in step with production."
        )

        metric_cols = st.columns(4)
        for i, (name, value) in enumerate(metrics.items()):
            metric_cols[i % 4].metric(name, f"{value:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        with col2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            st.pyplot(fig)

        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred, target_names=["no", "yes"]))

        st.caption(
            f"Evaluated on {len(y_test):,} held-out rows "
            "(70/30 split, random_state=13, stratified on the target, "
            "rows with unknown job/education excluded to match training)."
        )
