# Streamlit showcase for the Term Deposit Classification project.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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
MODEL_PATH = PROJ_ROOT / "models" / "model_pipeline.joblib"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
GITHUB_URL = "https://github.com/cameroncochrane/Term-Deposit-Classification"

RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration", "campaign",
]

# Domain values for the Bank Marketing dataset schema, hardcoded so the
# prediction form works even without the raw CSV present.
JOB_VALUES = [
    "admin", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL_VALUES = ["divorced", "married", "single"]
EDUCATION_VALUES = ["primary", "secondary", "tertiary", "unknown"]
YES_NO_VALUES = ["no", "yes"]
CONTACT_VALUES = ["cellular", "telephone", "unknown"]
MONTH_VALUES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


st.set_page_config(
    page_title="Term Deposit Classification",
    page_icon="\U0001F4B0",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return load(MODEL_PATH)


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


model = load_model()
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
   `marital`), and a `CyclicalDateEncoder` converts `day`/`month` into
   sine/cosine pairs.
4. **Modeling** (`modeling/train.py`) — an imbalanced-learn pipeline
   (impute → scale/encode → `RandomOverSampler` → `XGBClassifier`), tuned via
   `RandomizedSearchCV` (540 iterations, 5-fold stratified CV, optimizing F1).
5. **Evaluation** (`modeling/predict.py`) — accuracy, precision/recall, F1,
   MCC, ROC AUC, and log loss on the held-out test set.

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
        st.dataframe(raw_df.head(20), use_container_width=True)

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
                cols[i % 2].image(str(figure_path), caption=caption, use_container_width=True)

with tab_predict:
    st.header("Try a Prediction")
    st.write("Fill in a customer profile and campaign contact details to get a live prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 95, 40)
            job = st.selectbox("Job", JOB_VALUES)
            marital = st.selectbox("Marital status", MARITAL_VALUES)
            education = st.selectbox("Education", EDUCATION_VALUES)
        with col2:
            default = st.radio("Has credit in default?", YES_NO_VALUES, horizontal=True)
            balance = st.number_input("Average yearly balance (EUR)", value=500, step=100)
            housing = st.radio("Has housing loan?", YES_NO_VALUES, horizontal=True)
            loan = st.radio("Has personal loan?", YES_NO_VALUES, horizontal=True)
        with col3:
            contact = st.selectbox("Contact communication type", CONTACT_VALUES)
            day = st.slider("Last contact day of month", 1, 31, 15)
            month = st.selectbox("Last contact month", MONTH_VALUES, index=4)
            duration = st.number_input("Last contact duration (seconds)", min_value=0, value=180)
            campaign = st.number_input("Number of contacts this campaign", min_value=1, value=2)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_row = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign,
        }])[RAW_COLUMNS]

        probability = float(model.predict_proba(input_row)[0, 1])
        prediction = "yes" if probability >= 0.5 else "no"

        result_col, gauge_col = st.columns(2)
        with result_col:
            if prediction == "yes":
                st.success(f"Predicted outcome: **subscribes** (probability {probability:.1%})")
            else:
                st.error(f"Predicted outcome: **does not subscribe** (probability {probability:.1%})")
        with gauge_col:
            st.metric("Predicted subscription probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

with tab_performance:
    st.header("Model Performance (Held-Out Test Set)")
    if raw_df is None:
        st.warning(
            f"Raw dataset not found at `{RAW_DATA_PATH.relative_to(PROJ_ROOT)}`. "
            "Add the CSV there to enable this view."
        )
    else:
        X_test, y_test = build_eval_split(raw_df)
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test)

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
