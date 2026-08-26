# External Benchmark — IEEE-CIS Fraud Detection

Status: Implemented (v1: Logistic Regression only, `train_transaction.csv`
only). **Standalone, separate from ClearRisk Recover's core product.**

## Non-claims (read first)

> This external benchmark uses a public anonymized transaction-level
> fraud dataset. It does not validate ClearRisk Recover's synthetic
> merchant-week refund/chargeback-loss model, merchant evidence workflow,
> or any Razorpay/UPI/payment-gateway integration.
>
> This benchmark is not India-specific, not Razorpay data, not UPI data,
> and not proof of production fraud performance.

## Manual download preflight (required)

This project does not download, verify, or redistribute this dataset, and
never requires, stores, prints, or requests Kaggle credentials.

Before running this benchmark, you must independently download the
official IEEE-CIS Fraud Detection `train_transaction.csv` file, review
the source terms/license/access conditions yourself, and place it in:

```
data/external/ieee_cis/train_transaction.csv
```

`train_identity.csv` is **explicitly deferred** for v1 — it is never
loaded, joined, inspected, or required, even if you happen to have it
locally. See "Deferred scope" below.

If the file is missing, `external_benchmark.ieee_cis_loader.load_train_transaction()`
raises `DatasetNotFoundError` with a message that states the expected
path, the manual-download requirement, and that this benchmark is
separate from ClearRisk's core model — no fabricated result is ever
produced.

## Task separation

```text
ClearRisk core (UNCHANGED by this benchmark):
  Synthetic merchant-week refund/chargeback-loss early-warning,
  human review, evidence response, audit trail.
  demo_data/synthetic_merchant_week_data.csv -> ml/ -> app/ -> dashboard/

                    (no shared code, data, model, or DB)

External benchmark (this document):
  Public anonymized transaction-level fraud-classification experiment.
  data/external/ieee_cis/train_transaction.csv -> external_benchmark/
    -> external_benchmark/artifacts/latest_ieee_cis_report.json
```

Never merged: datasets, labels, models, or reported metrics. External
benchmark metrics are never displayed as ClearRisk merchant-loss
performance, and no FastAPI route or Streamlit page reads this module's
output.

## Input schema

Required columns in `train_transaction.csv`:

- `TransactionDT` — numeric, non-negative, seconds relative to an
  unspecified reference point (documented by the competition as relative
  time, not a real calendar timestamp).
- `TransactionAmt` — numeric, non-negative.
- `isFraud` — binary (0 or 1).

Optional columns used only if present: `ProductCD`.

## Validation

`external_benchmark/ieee_cis_validate.py` checks, before anything else
runs: the file exists (else the manual-download message above), the CSV
parses, all required columns are present, `TransactionDT`/`TransactionAmt`
are numeric and non-negative, and `isFraud` contains only 0/1. It also
guards that no identity-table (`id_*`, `DeviceType`, `DeviceInfo`) or
ID-like (`TransactionID`) column is ever selected as a feature, even
defensively, even though `train_identity.csv` itself is never loaded.

No raw rows are logged to console except a safe shape/schema-only preview
(`ieee_cis_loader.preview()`), and only when explicitly called — never
automatically.

## Feature list (v1, compact and explainable)

Numeric:
- `transaction_amt` — `TransactionAmt` as-is.
- `transaction_amt_has_cents` — whether the amount has a non-zero cents
  component.
- `transaction_relative_day` — `TransactionDT // 86400`, a relative day
  index (not a real calendar day).
- `transaction_hour_of_day_approx` — `(TransactionDT // 3600) % 24`, an
  approximate hour-of-day bucket, explicitly labeled approximate since
  `TransactionDT`'s zero point is not a documented real-world timestamp.

Categorical (only if present):
- `product_cd` — `ProductCD`, one-hot encoded.

Excluded from v1: the `V1`-`V339` anonymized engineered-feature block, the
full `id_01`-`id_38` identity block, and any raw identifier column. The
selected feature list is printed and saved to
`external_benchmark/artifacts/selected_feature_list.json` on every run.

## Chronological split

Ordered by `TransactionDT` (no random split, no shuffling before split):

- Earliest 70% → training.
- Next 15% → validation (threshold selection only).
- Latest 15% → held-out test.

Row counts and `TransactionDT` min/max per split are recorded in the
saved report's `split` field.

## Model and threshold approach

A single scikit-learn `Pipeline`: median imputation for numeric features,
most-frequent imputation + one-hot encoding for the optional categorical
feature, then `LogisticRegression(class_weight="balanced", random_state=42)`
— the documented class-imbalance handling for v1. No secondary model
(gradient boosting, XGBoost, random forest, neural network) is
implemented in v1; see "Deferred scope" below.

Threshold is selected on the validation split only, over a grid from 0.05
to 0.95 in steps of 0.05, maximizing F2 (recall-weighted, since missed
fraud is typically costlier than a false alarm). If every threshold on
the grid produces zero F2 signal, a documented fallback of 0.5 is used
and flagged (`fallback_used: true`) rather than silently picked. The full
grid and the selected threshold's precision/recall/F2 are saved.

## Metrics (held-out test, at the selected threshold)

Class prevalence, precision, recall, F1, F2, PR-AUC, ROC-AUC (marked
secondary), false-positive rate, false-negative rate, and the full
confusion matrix.

## Artifact location

```
external_benchmark/artifacts/latest_ieee_cis_report.json
external_benchmark/artifacts/selected_feature_list.json
```

Both contain aggregate metrics and configuration only — no raw
transaction rows, transaction IDs, merchant IDs, identity fields, or
record-level predictions.

## Runtime / memory guidance

`train_transaction.csv` is a large public competition file (commonly
several hundred MB uncompressed — exact size not verified here; check
after your own download). The loader uses a plain `pandas.read_csv` call
in v1; if this proves impractical on your machine, that is a known,
documented limitation, not silently worked around with hidden
downsampling. Logistic Regression is expected to run in at most a few
minutes on the v1 feature subset.

## Deleting local external data after testing

```bash
rm -rf data/external/ieee_cis
```

This does not affect already-generated files under
`external_benchmark/artifacts/`.

## Deferred scope (not in v1)

- `train_identity.csv` and any identity-joined feature — deferred to
  reduce complexity and avoid unnecessary privacy/interpretation risk, per
  the approved design decision. May be revisited in a future, separately
  approved pass.
- `HistGradientBoostingClassifier`, XGBoost, random forest, neural
  networks, SHAP, hyperparameter-search infrastructure — documented as
  optional future benchmark work only after this Logistic Regression
  pipeline is proven and only if time remains. Not part of v1 or this
  submission's pitch/demo.
- Exact Kaggle dataset URL, license, and access-condition verification —
  this is a human manual preflight item. Neither this document nor any
  code in `external_benchmark/` claims to have verified Kaggle's current
  terms; the user must check them directly before downloading.

## How to run once the file is manually placed

```bash
python3 -m external_benchmark.ieee_cis_evaluate
```

This prints the selected feature list, the threshold-selection grid, and
the full report to the console, then saves
`external_benchmark/artifacts/latest_ieee_cis_report.json`.
