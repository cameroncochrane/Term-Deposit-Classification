from pathlib import Path
import sys
import pickle
from joblib import dump, load

from loguru import logger
from tqdm import tqdm
import typer
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import randint, loguniform, uniform

import json

from deposit_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LATEST_RUN_INFO_PATH,
)

from deposit_classification.features import FeatureDropper, CyclicalDateEncoder

app = typer.Typer()


@app.command()
def main(
        features_path: Path = PROCESSED_DATA_DIR / "X_train.pkl", # X_train
        labels_path: Path = PROCESSED_DATA_DIR / "y_train.pkl", # y-train
        model_path: Path = MODELS_DIR / "model_pipeline_2.joblib"
    ):
    print("train")

    with open(features_path, "rb") as f:
        X_train = pickle.load(f)

    with open(labels_path, "rb") as f:
        y_train = pickle.load(f)

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # Depends highly on class imbalance
        tree_method="hist",
        random_state=13,
        n_jobs=-1
    )

    features_to_remove_afs = [
        "age",
        "balance",
        "campaign",
        "default",
        "loan",
        "contact",
        "education",
        "marital"
    ]

    numeric_features = [
        "duration",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos"
    ]

    housing_features = [
        "housing"
    ]

    one_hot_features = [
        "job"
    ]

    numeric_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "robust_scaler",
                RobustScaler()
            )
        ]
    )

    housing_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=[["no", "yes"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ]
    )

    one_hot_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Combine the above components
    column_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "housing",
                housing_pipeline,
                housing_features
            ),
            (
                "categorical",
                one_hot_pipeline,
                one_hot_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

        # Define the overall model pipeline:

    xgb_model_pipeline_afs = ImbPipeline(
        steps=[
            (
                "feature_selection",
                FeatureDropper(
                    columns=features_to_remove_afs
                )
            ),
            (
                "cyclical_encoding",
                CyclicalDateEncoder(
                    day_column="day",
                    month_column="month",
                    drop_original=True
                )
            ),
            (
                "preprocessor",
                column_preprocessor  # All encoding + scaling (apart from cyclical encoding which is done above)
            ),
            (
                "sampler",
                RandomOverSampler(
                    sampling_strategy="auto",
                    random_state=13
                )
            ),
            (
                "model",  # This is the part to change between trying different models but with the same processing pipeline.
                xgb_model
            )
        ]
    )

    # Application:

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=13
    )

    param_distributions = {
        "sampler__sampling_strategy": [
            0.25,
            0.40,
            0.60,
            0.80
        ],
        "model__n_estimators": randint(
            100,
            1000
        ),
        "model__learning_rate": loguniform(
            0.1,
            0.12
        ),
        "model__max_depth": randint(
            2,
            6
        ),
        "model__min_child_weight": randint(
            3,
            21
        ),
        "model__gamma": uniform(
            0,
            1.5
        ),
        "model__subsample": uniform(
            0.60,
            0.35
        ),
        "model__colsample_bytree": uniform(
            0.60,
            0.35
        ),
        "model__reg_alpha": loguniform(
            1e-4,
            2
        ),
        "model__reg_lambda": loguniform(
            1,
            30
        )
    }

    search_xgb = RandomizedSearchCV(
        estimator=xgb_model_pipeline_afs,
        param_distributions=param_distributions,
        n_iter=540,
        scoring="f1",
        cv=cv,
        refit=True,
        return_train_score=True,
        random_state=13,
        n_jobs=-1,
        verbose=2,
        error_score="raise"
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        search_xgb.fit(
            X_train,
            y_train
        )

        # Training score
        print("Best 5-fold CV F1-score:", search_xgb.best_score_)
        print("Best parameters:")
        print(search_xgb.best_params_)

        best_xgb_ros = search_xgb.best_estimator_
        dump(best_xgb_ros, model_path) # Save

        mlflow.log_params(search_xgb.best_params_)
        mlflow.log_metric("cv_f1", search_xgb.best_score_)
        # MLflow 3.x defaults to skops serialization, which audits pickled classes
        # against an allowlist and rejects our custom transformers (and imblearn/
        # xgboost's own classes) as "untrusted". Plain pickle is what we already
        # trust elsewhere in this pipeline (the joblib dump above), so use it here too.
        logged_model = mlflow.sklearn.log_model(
            best_xgb_ros,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

        # predict.py logs test-set metrics into this same run, and register_model.py
        # registers this exact model, so the two later scripts' separate CLI
        # invocations need to agree on the run id and the (MLflow 3.x "Logged Model")
        # model_uri log_model actually produced.
        LATEST_RUN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        LATEST_RUN_INFO_PATH.write_text(
            json.dumps({"run_id": run.info.run_id, "model_uri": logged_model.model_uri})
        )
        print("MLflow run id:", run.info.run_id)
        print("MLflow model uri:", logged_model.model_uri)

if __name__ == "__main__":
    app()
