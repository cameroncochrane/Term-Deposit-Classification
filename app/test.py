from joblib import load

PIPELINE_PATH = "models/model_pipeline.joblib"
pipeline = load(PIPELINE_PATH)

# Pipeline summary:
print(pipeline)

# Input features summary:
print(pipeline.feature_names_in_)
print(pipeline.n_features_in_)