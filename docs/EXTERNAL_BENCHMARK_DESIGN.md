# External Benchmark Design — IEEE-CIS Fraud Detection

**Status: approved and implemented (v1). See `docs/EXTERNAL_BENCHMARK.md` for the
as-built documentation.** This document is retained as the design record; the
final approved decisions are noted inline below where they narrow or resolve an
earlier open question.

**Approved final decisions:**

1. `train_identity.csv` is **deferred** — not loaded, joined, inspected, or
   exposed in v1, anywhere. See Section 9 (now resolved).
2. Model scope is **Logistic Regression only** in v1 — no
   `HistGradientBoostingClassifier`, XGBoost, random forest, neural networks,
   SHAP, or hyperparameter search. Gradient boosting is documented as optional
   future work only after the Logistic Regression pipeline is proven and only
   if time remains.
3. Kaggle dataset URL/license/access-term verification is a **human manual
   preflight item** — this project does not claim to have verified it, does
   not download automatically, and never requires/stores/prints Kaggle
   credentials.

This document is required, per the requester's instruction, before writing any code
for `external_benchmark/`. It exists to make the scope and the non-claims explicit
and reviewable up front — nothing below has been built yet.

## 0. What this is and is not

This is a **separate, standalone technical benchmark module** that trains and
evaluates a transaction-level fraud classifier on a public, external, anonymized
Kaggle dataset (IEEE-CIS Fraud Detection). It exists to demonstrate that the team
can also build and evaluate a classic tabular fraud-modeling pipeline (feature
engineering, chronological split, baseline + optional gradient-boosting model,
precision/recall/PR-AUC/ROC-AUC/F1/F2/confusion-matrix/FPR reporting) on a dataset
that is *not* synthetic and *not* authored by this project.

It is **not**:

- A Razorpay dataset, an Indian payment dataset, or merchant chargeback data.
- A validation of ClearRisk Recover's merchant-week refund/chargeback model, its
  rules engine, or its 5-state latent-state label design.
- A source of any claim that ClearRisk Recover has been tested against real-world
  fraud data. IEEE-CIS is per-transaction, card-network fraud data from an
  unrelated domain (US e-commerce/card transactions circa the competition's
  release), not merchant-week refund/chargeback behavior.
- Wired into `ml/`, `app/`, `rules/`, or `dashboard/` in any way. It does not call
  `ml.features`, `ml.rules_engine`, `app.services`, or any FastAPI route, and no
  FastAPI route or Streamlit page will read from it.

## 1. Task separation diagram

```text
ClearRisk Recover core product (UNCHANGED by this task)
  demo_data/synthetic_merchant_week_data.csv
    -> ml/features.py -> rules/risk_rules.yaml -> ml/train_baseline_model.py
       -> ml/evaluate_model.py -> app/ (FastAPI) -> dashboard/ (Streamlit)
  Scope: merchant-week refund/chargeback loss risk. Synthetic data only.

                                   |
                    (no import, no shared code, no shared DB)
                                   |
                                   v

External technical benchmark (THIS task, fully separate)
  data/external/ieee_cis/train_transaction.csv   (user-supplied, gitignored, never committed)
    -> external_benchmark/ieee_cis_loader.py      (read + basic parsing)
    -> external_benchmark/ieee_cis_validate.py     (schema/column checks, prevalence report)
    -> external_benchmark/ieee_cis_features.py     (compact documented feature subset)
    -> external_benchmark/ieee_cis_train.py        (chronological split, LogisticRegression [+ optional HGB])
    -> external_benchmark/ieee_cis_evaluate.py     (metrics -> external_benchmark/artifacts/*.json)
  Scope: transaction-level fraud classification benchmark. External anonymized data only.
```

The only thing the two sides share is this repository and the general engineering
conventions (fail-safe validation, no fabricated metrics, synthetic-data-style
honesty labeling adapted to "external-data" labeling here). No code, config, model
artifact, or database row is shared.

## 2. Data boundary

- **Source**: IEEE-CIS Fraud Detection, a Kaggle competition dataset. **Unverified
  by me** — the requester must confirm the exact current Kaggle URL, license terms,
  and whether redistribution/local use for this purpose is permitted under Kaggle's
  competition rules before any file is downloaded. This document does not link to
  or assume access to the dataset; I have not fetched Kaggle's terms in this
  session.
- **Location**: `data/external/ieee_cis/train_transaction.csv` (required),
  `train_identity.csv` (optional), placed there manually by the user. This project
  will not download it automatically and will not embed or reference any Kaggle
  API token/credential anywhere in code, config, or docs.
- **`.gitignore`**: `data/external/` added so the dataset is never committed. (Note:
  this repository currently has no `.gitignore` file at all and is not a git
  repository yet — either state is fine for this addition; the pattern will be
  ready whenever git tracking starts.)
- **No PII/identity use by default**: `train_identity.csv` join is optional and, if
  used at all, only for anonymized categorical device/browser-type fields already
  present in the public dataset — never any field that could re-identify a person.
  Default runs use `train_transaction.csv` alone.
- **No external transmission**: nothing in `external_benchmark/` makes a network
  call. All processing is local, offline, over a user-supplied local file.
- **No UI/API exposure**: no transaction record, row, or identifier from this
  dataset is ever rendered in the Streamlit dashboard or returned by any FastAPI
  endpoint. The benchmark's only outputs are aggregate metrics and reports.

## 3. Feature plan (compact, documented subset — not the full ~430-column schema)

Starting subset, chosen because they are well-documented in the competition's own
public data description and require no identity-table join:

| Feature | Source column(s) | Notes |
|---|---|---|
| `transaction_amt` | `TransactionAmt` | Used as-is; log-transform considered if skew is extreme. |
| `transaction_amt_decimal_part` | `TransactionAmt` | Whether the amount has a non-zero cents component — a commonly documented signal in public IEEE-CIS writeups. |
| `product_cd` | `ProductCD` | One-hot/categorical. |
| `card_type_bucket` | `card1`-`card6` | Only non-identifying bucketed/one-hot categorical use — no raw card token treated as a real card number (these are already anonymized competition fields, not real PANs, but are still handled only as opaque categorical IDs, never displayed). |
| `email_domain_bucket` | `P_emaildomain`, `R_emaildomain` | Bucketed into a small set of common domains vs. "other" vs. missing — no raw email ever stored or shown. |
| `transaction_hour_of_day` | derived from `TransactionDT` | `TransactionDT` is documented as seconds from a reference point, not a wall-clock timestamp; hour-of-day is taken modulo a day-length assumption and labeled as approximate. |
| `count_null_features` | count of nulls across a fixed column list | A commonly documented simple signal; avoids depending on the full `V1`-`V339` block for the first pass. |

Explicitly **excluded from the first pass**: the `V1`-`V339` anonymized
engineered-feature block and the full `id_01`-`id_38` identity block — both add
real modeling value in public leaderboards but are out of scope for a "compact,
documented" first version. Adding them would be a follow-up, not part of this
approval.

## 4. Split plan

- Chronological split using `TransactionDT` (documented as a relative time
  reference, not a real calendar date) — earliest ~60% train, next ~20%
  validation (threshold selection), latest ~20% held-out test. Mirrors this
  project's own "time-based, never random" split discipline (`ARCHITECTURE.md`,
  `MODEL_CARD.md`), applied here for the same reason: prevents leakage from
  future transactions into training.
- No merchant/customer grouping is applied unless a later pass adds `card1`-based
  grouping to check for identity leakage across the split boundary — flagged here
  as a known follow-up, not attempted in v1.

## 5. Evaluation plan

- Primary: precision, recall, PR-AUC (appropriate given `isFraud`'s known heavy
  class imbalance).
- Secondary: ROC-AUC, F1, F2 (recall-weighted, since missed fraud is typically
  costlier than a false alarm in this literature), confusion matrix at the
  selected operating threshold, false-positive rate.
- Threshold selection method: chosen on the validation split only (never on
  test), using the same "maximize F2 subject to a precision floor" style already
  used in `ml/evaluate_model.py`, adapted with its own threshold value — not
  shared with ClearRisk's merchant-week threshold.
- Class prevalence (`isFraud` positive rate) reported explicitly for train/val/test
  splits, since a shift in prevalence across the chronological split is itself a
  data-quality finding worth surfacing, not hiding.
- Two models: `LogisticRegression` (required baseline, mirrors this project's
  "transparent rules/interpretable model first" principle) and, only if the
  baseline runs successfully and runtime is practical on this machine,
  `HistGradientBoostingClassifier` as an optional secondary comparison — never a
  replacement for reporting the baseline.
- Output: a single offline JSON report,
  `external_benchmark/artifacts/ieee_cis_benchmark_report.json` (or timestamped
  variants, mirroring `ml/artifacts/latest_evaluation_report.json`'s pattern) —
  written to a path entirely separate from `ml/artifacts/`, never merged with or
  overwriting ClearRisk's own evaluation report.

## 6. Non-claims (will be repeated verbatim in `docs/EXTERNAL_BENCHMARK.md` and `external_benchmark/README.md`)

1. This dataset is external, anonymized, transaction-level data from an unrelated
   payments/e-commerce context — not Razorpay data, not Indian payment data, and
   not merchant chargeback data.
2. It does not contain merchant refund/chargeback/evidence/support-ticket fields,
   so it cannot exercise or validate ClearRisk Recover's actual product scope
   (merchant-week refund/chargeback loss risk).
3. Results from this benchmark validate only that the team can build a correct,
   honestly-evaluated transaction-fraud modeling pipeline — they say nothing about
   ClearRisk Recover's own held-out-test performance, which is reported separately
   in `MODEL_CARD.md`/`ml/artifacts/latest_evaluation_report.json`.
4. The exact dataset source URL, current license/competition-rules access
   conditions, and permitted-use terms must be verified by the user directly on
   Kaggle before download — this document does not assert those terms and no
   code in this module will assume, hardcode, or bypass them.

## 7. Runtime plan

- `train_transaction.csv` is commonly several hundred MB to ~700MB uncompressed in
  the public competition (unverified exact figure — user should check after
  download); the loader will use `pandas.read_csv` with an explicit `dtype` map
  and, if needed, chunked reading, rather than assuming it fits trivially in
  memory. No downsampling is silently applied — if a sample is used for a fast
  dev loop, it will be an explicit, logged, documented flag, not a hidden default.
- `LogisticRegression` is expected to run in at most a few minutes on this
  feature subset. `HistGradientBoostingClassifier` is attempted only after the
  baseline succeeds, with a wall-clock timeout/early-exit noted in the script if
  it proves impractical locally — if so, the report states "gradient boosting:
  not run (impractical locally)" rather than a fabricated result.

## 8. Test plan

- `tests/test_ieee_cis_validate.py`: schema validation against a small, hand-built
  fixture DataFrame (never the real Kaggle file) — required columns present
  (`TransactionDT`, `TransactionAmt`, `isFraud`, `ProductCD`, etc.), rejects
  missing columns, rejects out-of-range `isFraud` values, reports class
  prevalence correctly on a known fixture.
- `tests/test_ieee_cis_features.py`: the compact feature subset (Section 3)
  produces expected values on a small hand-built fixture — e.g. a
  `TransactionAmt` of `100.50` yields a non-zero
  `transaction_amt_decimal_part`; missing `P_emaildomain` buckets to a documented
  "missing" category, not a crash.
- No test reads or requires the real Kaggle CSV to exist — every test uses small
  synthetic/hand-built fixtures matching the *documented public schema*, so the
  suite runs the same whether or not the user has downloaded the real dataset
  locally.
- `external_benchmark/ieee_cis_loader.py`, `_train.py`, and `_evaluate.py` are
  exercised only via a manual `if __name__ == "__main__"` run against the real
  local file (documented in `external_benchmark/README.md`), not via the
  automated `pytest` suite — consistent with how this project already treats
  "requires the real generated dataset" scripts.

## 9. Open questions — resolved by approval

- **Kaggle dataset URL/license/access terms**: resolved as a permanent human
  manual preflight item, not something this project verifies or claims. Stated
  in `external_benchmark/README.md`, `data/external/README.md`, and
  `docs/EXTERNAL_BENCHMARK.md`.
- **`train_identity.csv`**: resolved as deferred for v1 — never loaded, joined,
  inspected, or exposed, even if present locally. `external_benchmark/ieee_cis_validate.py`
  additionally guards against any identity-table-prefixed column being used as
  a feature, defensively.
- **`HistGradientBoostingClassifier`**: resolved as deferred — v1 implements
  Logistic Regression only. Documented as optional future benchmark work in
  `docs/EXTERNAL_BENCHMARK.md`'s "Deferred scope" section, only after the
  Logistic Regression pipeline is proven and only if time remains.
