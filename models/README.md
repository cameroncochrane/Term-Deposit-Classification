This directory holds two unrelated things that happen to share a folder:

- **`streamlit_demo/model_pipeline.joblib`** — a frozen model, committed to git, used only by
  `streamlit_app.py`. Not touched by retraining; update it manually if you ever want the demo to
  reflect a newer model.
- **Everything else here** (`model_pipeline_2.joblib`, `latest_run_info.json`) — working output
  from the local training pipeline (`deposit_classification/modeling/`), gitignored, overwritten
  on every run. This is `deposit_classification/config.py`'s `MODELS_DIR` - the default place
  `train.py` and friends read/write, not a deliberately curated artifact store.

**The model actually deployed and serving predictions is [app/models/model_pipeline_2.joblib](../app/models/model_pipeline_2.joblib)**,
a separate file, written by `export_production_model.py` from whatever the MLflow registry has
tagged `production`. See [deposit_classification/modeling/README.md](../deposit_classification/modeling/README.md).
