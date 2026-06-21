from pathlib import Path
from joblib import dump, load
import pickle

import numpy as np

from loguru import logger
from tqdm import tqdm
import typer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from deposit_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_test.pkl", # X_test
    labels_path: Path = PROCESSED_DATA_DIR / "y_test.pkl", # y_test
    model_path: Path = MODELS_DIR / "model_pipeline.joblib",
    predictions_path: Path = PROCESSED_DATA_DIR / "y_pred.pkl",
    probabilities_path: Path = PROCESSED_DATA_DIR / "y_proba.pkl"
):
    def evaluate_binary_classifier(
            y_test,
            y_pred,
            y_proba=None,
            positive_label=1,
            verbose=True):
        """
        Evaluate a binary classification model.

        Parameters
        ----------
        y_test : array-like
            True labels.

        y_pred : array-like
            Predicted class labels.

        y_proba : array-like, optional
            Predicted probabilities for the positive class.
            Required for ROC-AUC and Log Loss.

        positive_label : int or str, default=1
            Label considered as the positive class.

        verbose : bool, default=True
            Whether to print results.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing evaluation metrics.
        """

        # Convert to numpy arrays
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Core metrics
        metrics_dict = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "Matthews Corrcoef": matthews_corrcoef(y_test, y_pred)
        }

        # Probability-based metrics
        if y_proba is not None:
            y_proba = np.array(y_proba)

            metrics_dict["ROC AUC"] = roc_auc_score(y_test, y_proba)
            metrics_dict["Log Loss"] = log_loss(y_test, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        clf_report = classification_report(y_test, y_pred)

        if verbose:
            print("\n===== Binary Classification Evaluation =====\n")

            for metric, value in metrics_dict.items():
                print(f"{metric}: {value:.4f}")

            print("\nConfusion Matrix:")
            print(cm)

            print("\nClassification Report:")
            print(clf_report)

        return {
            "metrics": metrics_dict,
            "confusion_matrix": cm,
            "classification_report": clf_report
            }
    
    with open(features_path, "rb") as f:
        X_test = pickle.load(f)

    with open(labels_path, "rb") as f:
        y_test = pickle.load(f)

    # Test score:
    best_xgb_ros = load(model_path)
    y_pred = best_xgb_ros.predict(X_test)
    y_proba = best_xgb_ros.predict_proba(X_test)[:, 1]

    best_xgb_ros_evaluate = evaluate_binary_classifier(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )


if __name__ == "__main__":
    app()
