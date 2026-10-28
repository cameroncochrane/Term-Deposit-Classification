## Model training, evaluation, and MLflow experiment tracking / registry.

## Pipeline

```bash
python deposit_classification/modeling/train.py             # trains, logs params/cv score/model to MLflow
python deposit_classification/modeling/predict.py           # evaluates on the test set, logs test metrics into the same run
python deposit_classification/modeling/register_model.py    # registers that run's model, tags it validation_status=staging
python deposit_classification/modeling/promote_model.py     # promotes staging -> production if it doesn't regress
python deposit_classification/modeling/export_production_model.py  # writes the production model to app/models/
```

`train.py` writes `models/latest_run_info.json` (`{run_id, model_uri}`) so the later steps know which
run/model to act on without you passing IDs around by hand. Each script also accepts the relevant
ID explicitly (`--version`, etc.) if you want to act on something other than "the most recent run."

## Project-specific reference

| Setting | Value |
|---|---|
| MLflow version | `3.17.2` (pinned in `requirements.txt`) |
| Tracking/registry backend store | `sqlite:///mlflow.db` (project root) |
| Artifact store | `mlruns/` (project root) |
| Experiment name | `term-deposit-classification` |
| Registered model name | `term-deposit-classifier` |
| Bridge file | `models/latest_run_info.json` (`{run_id, model_uri}`) |
| Production alias | `models:/term-deposit-classifier@production` |

All of the above are defined once in `deposit_classification/config.py` - every script imports
them from there rather than hardcoding strings.

## Tracking backend

- Backend store (experiments, runs, params, metrics, **and the Model Registry** - registry
  operations require a database-backed store, not plain file tracking): local sqlite at
  `mlflow.db` (project root). No server, no AWS cost.
- Artifact store (the logged model files): local `mlruns/` (project root).
- Both are gitignored - this is single-operator tooling for now, same as Terraform's local state.
  View runs and the registry with:

  ```bash
  source .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
  ```

  `mlflow` only exists inside `.venv` - running it from a shell that hasn't activated the venv
  gives `command not found`; use `.venv/bin/mlflow ...` instead if you don't want to activate.
  Also registered in `.claude/launch.json` as `mlflow` for tooling that launches dev servers by
  name from there. Then open `http://localhost:5001`.

## Registry and promotion

Registered model name: `term-deposit-classifier`. Every newly registered version is tagged
`validation_status=staging`. `promote_model.py` moves a version to the `production` alias only if
it does **not regress** on either metric relative to whatever currently holds that alias:

- `test_f1_score`
- `test_recall` - included alongside F1 because a false negative (missing a would-be subscriber)
  is the costlier mistake for this business problem, and F1 alone can mask a recall drop.

If there's no current production version yet (first run), promotion happens unconditionally. A
failed promotion exits non-zero with the specific metric(s) that regressed, printed - usable as a
CI gate later rather than something a human has to read off the MLflow UI.

## Getting a promoted model into the live stack

`export_production_model.py` loads whatever is tagged `production`, re-points the two custom
transformer classes' pickled module references from `deposit_classification.features` to
`api.transformers` (same fix as the one made directly to `app/models/model_pipeline_2.joblib` in
phase 2 - the Lambda container only ships `app/`, so the pickle can't reference anything outside
it), and overwrites `app/models/model_pipeline_2.joblib`.

That's as far as this goes automatically. Getting the *new* model actually running in AWS still
means rebuilding and redeploying the container - see [app/README.md](../../app/README.md#updating-the-api).
Wiring "a new production model" straight through to a redeployed container is CI/CD (phase 5) and
drift-triggered retraining (phase 6) territory, not this script's job.

## MLflow 3.x gotchas hit building this

Both are already handled in this project's code - noted here so nobody "fixes" them back by accident,
and so updating the pinned `mlflow` version doesn't silently reintroduce them.

- **`mlflow.sklearn.log_model` defaults to `skops` serialization**, which audits the pickled
  object tree against an allowlist of trusted types and raises `UntrustedTypesFoundException` for
  anything outside it - which includes our own `FeatureDropper`/`CyclicalDateEncoder`, plus
  imblearn's and xgboost's own classes. `train.py` passes
  `serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE` explicitly to opt back into
  plain pickle, the same trust model the rest of this pipeline already uses (the parallel
  `joblib.dump` a few lines above it, `export_production_model.py`, etc.).
- **`log_model(..., name=...)` no longer stores the model as a run artifact at that path.** MLflow
  3.x's "Logged Model" entity means `runs:/<run_id>/model` does *not* reliably resolve anymore.
  `train.py` captures the `model_uri` that `log_model` actually returns and writes it into
  `latest_run_info.json` - `register_model.py` reads that URI rather than reconstructing one.
