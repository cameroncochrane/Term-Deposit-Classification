Drift monitoring for the deployed model, using Evidently.

- `drift_report.py` - pure comparison logic: load the reference dataset, pull recent production
  predictions from S3, run Evidently, return a `DriftResult` verdict. No CLI, no side effects -
  this is what `app/tests/test_drift.py` exercises directly with synthetic data.
- `alerting.py` - opens or updates a GitHub Issue (label `model-drift`) when drift is detected.
  Comments on an existing open issue rather than creating duplicates on every scheduled run.
- `monitoring_job.py` - the actual CLI entrypoint the scheduled workflow runs: load data, check
  drift, write an HTML report artifact, alert if needed, exit non-zero on drift so the workflow
  can branch into the retrain trigger.
- `reference_data.csv` - the training distribution's serving-relevant columns (`duration`, `day`,
  `month`, `housing`, `job` - the only fields the API ever sees, since `FeatureDropper` removes
  the rest before the model does). Regenerate with
  `python deposit_classification/modeling/export_reference_data.py` after a meaningfully
  different retrain.

## Where "current" data comes from

`app/api/prediction_log.py` logs every live `/predict` request's input (plus the prediction and
a timestamp) to S3 as `predictions/<date>/<uuid>.json`, written by the Lambda's execution role.
It's a no-op locally (no `PREDICTIONS_BUCKET` set) and never raises - a logging failure must
never break an actual prediction response.

## Thresholds

- `DEFAULT_DRIFT_SHARE_THRESHOLD = 0.4` - dataset drift is flagged when Evidently's own test
  (`DriftedColumnsCount`) says at least 40% of the 5 serving columns have drifted. Evidently
  auto-selects the underlying stat test per column (p-value based for small samples, distance
  based for large ones) - these two use *opposite* comparison directions against their
  threshold, so the verdict is read from Evidently's own `TestStatus`, not re-derived from the
  raw metric value.
- `MIN_SAMPLE_SIZE = 30` - below this many logged predictions, the check is skipped entirely
  rather than run on a handful of noisy rows.

## Schedule and retraining

A scheduled GitHub Actions workflow runs `monitoring_job.py` periodically. On drift, it opens the
alert issue and triggers the full `train.py → predict.py → register_model.py → promote_model.py →
export_production_model.py` chain (see
[deposit_classification/modeling/README.md](../../deposit_classification/modeling/README.md)).
If the retrained model is promoted (doesn't regress on F1/recall), the workflow opens a PR with
the updated `app/models/model_pipeline_2.joblib` rather than committing straight to `main` - an
automated process silently replacing the production model felt like the one place worth a human
checkpoint, even though everything else in this pipeline auto-deploys on merge.

**Honest limitation:** retraining here still uses the same static training CSV, not genuinely new
labeled data - there's no live data-ingestion pipeline. This demonstrates the automation (detect
drift → retrain → gate on regression → redeploy) working correctly end-to-end, but it can't
actually correct for real-world drift without new ground truth, which is out of scope for a
project built on a static dataset.
