# external_benchmark/

A standalone, separate transaction-level fraud-classification experiment on
the public IEEE-CIS Fraud Detection dataset. **Not part of RiskLens's
core product** — see `docs/EXTERNAL_BENCHMARK.md` and
`docs/EXTERNAL_BENCHMARK_DESIGN.md` for the full data boundary and
non-claims.

## Manual preflight (required before running)

This project does not download, verify, or redistribute this dataset.
Before running this benchmark, you must independently download the
official IEEE-CIS Fraud Detection `train_transaction.csv` file from
Kaggle, review its source terms/license/access conditions yourself, and
place it at exactly:

```
data/external/ieee_cis/train_transaction.csv
```

`train_identity.csv` is **deferred** — not loaded, joined, inspected, or
required in v1, even if present.

## How to run

```bash
python3 -m external_benchmark.ieee_cis_evaluate
```

If the file is missing, this fails with a clear message stating the
expected path, the manual-download requirement, and that this benchmark
is separate from RiskLens's core model — it will not fabricate a result.

## Files

- `ieee_cis_loader.py` — loads the local CSV only; never downloads, never
  touches `train_identity.csv`.
- `ieee_cis_validate.py` — schema/type validation; also guards that no
  identity-table or ID-like column is ever used as a feature.
- `ieee_cis_features.py` — the compact v1 feature set (see
  `docs/EXTERNAL_BENCHMARK.md`).
- `ieee_cis_train.py` — chronological split (70/15/15), sklearn Pipeline,
  Logistic Regression baseline only (no gradient boosting/XGBoost/random
  forest/neural nets/SHAP/hyperparameter search in v1), F2-based threshold
  selection on validation only.
- `ieee_cis_evaluate.py` — held-out-test evaluation and the aggregate-only
  report writer (`artifacts/latest_ieee_cis_report.json`).

## Deferred / out of scope for v1

- `train_identity.csv` and any identity-joined feature.
- `HistGradientBoostingClassifier`, XGBoost, random forest, neural nets,
  SHAP, hyperparameter search — documented as optional future benchmark
  work only after this Logistic Regression pipeline is proven.
