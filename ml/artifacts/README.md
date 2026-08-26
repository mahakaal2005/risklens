# ml/artifacts/

Generated model and evaluation artifacts from `ml/train_baseline_model.py` and `ml/evaluate_model.py`. **Not committed to version control** — these are binary/regenerable build outputs, not source. If this repository is placed under git, add to `.gitignore`:

```
ml/artifacts/*.joblib
ml/artifacts/*_metadata.json
ml/artifacts/latest_evaluation_report.json
ml/artifacts/evaluation_report_*.json
```

## Files

- `logistic_regression_v0.1.0.joblib` (from `train_baseline_model.py`) — the fitted scikit-learn `Pipeline` (preprocessing `ColumnTransformer` + `LogisticRegression`), fit on the training split only.
- `logistic_regression_v0.1.0_metadata.json` (from `train_baseline_model.py`) — seed, generation timestamp, feature column list, training/validation row counts, split date ranges, selected operating threshold and how it was chosen, and the synthetic-data-only statement.
- `latest_evaluation_report.json` (from `evaluate_model.py`, Milestone 7) — the persisted, aggregate-only held-out evaluation report that `GET /metrics` reads. **Always overwritten** by the latest `python3 -m ml.evaluate_model` run — see `ml/evaluation_report.py` for the schema and `docs/MILESTONE_7_METRICS.md` for the full artifact lifecycle. Contains no per-row data, no merchant IDs, and no latent-state values — aggregate metrics only.
- `evaluation_report_<timestamp>.json` (from `evaluate_model.py`) — an additional timestamped snapshot of the same report, written alongside `latest_evaluation_report.json` on every run, for historical comparison. Not read by the API.

## Regenerating

```bash
python3 ml/generate_synthetic_data.py
python3 -m ml.train_baseline_model
python3 -m ml.evaluate_model      # also writes latest_evaluation_report.json
```

## Important

This model is trained on synthetic data for demonstration and decision-support purposes only. It is not a real-world chargeback prediction model, and its artifacts must never be described as validated against real merchant data.
