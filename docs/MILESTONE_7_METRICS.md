# Milestone 7 — Persisted Evaluation Report and Metrics API Completion

Status: Implemented and run against real held-out data (seed=42). **Local synthetic-data demonstration only.**

---

## 1. Artifact lifecycle

```text
python3 -m ml.evaluate_model
        │
        ▼
ml.evaluate_model.evaluate()          -- computes rules-only / logistic-regression / combined-policy
        │                                 metrics on the held-out test split (unchanged from Milestone 3)
        ▼
ml.evaluation_report.build_report()   -- assembles the aggregate-only report dict from that result
        │                                 plus demo_data/synthetic_data_metadata.json and the
        │                                 trained model's metadata JSON
        ▼
ml.evaluation_report.save_report()    -- writes ml/artifacts/latest_evaluation_report.json
        │                                 (always overwritten) and a timestamped copy
        ▼
GET /metrics (app/api/routes/metrics.py)
        │
        ▼
ml.evaluation_report.load_report() + validate_report()   -- read + validate only, never recompute
        │
        ▼
HTTP 200 response (always 200 -- missing/invalid artifact is a valid state, not a server error)
```

The artifact is produced entirely offline, by the CLI command `python3 -m ml.evaluate_model`. The API process never imports or calls `ml.train_baseline_model.train()` or `ml.evaluate_model.evaluate()` — verified by `tests/test_api_metrics.py::test_metrics_never_trains_or_evaluates_on_request`, which monkeypatches both functions to raise if called and confirms the endpoint still responds correctly.

## 2. Report schema explanation

See `ml/evaluation_report.py` for the authoritative schema (`build_report()`, `REQUIRED_TOP_LEVEL_KEYS`, `REQUIRED_METHOD_METRIC_KEYS`). Ten top-level sections:

- `report_version`, `generated_at`, `data_mode`, `synthetic_data_notice` — report metadata.
- `dataset` — seed, generator version, row/merchant counts, date range (from `demo_data/synthetic_data_metadata.json`).
- `split` — train/validation/test week counts, row counts, and date ranges (from `ml.split_data`).
- `model` — name, version, selected threshold, threshold-selection method, feature count, and `excluded_fields` (the one place `label_high_loss_next_30d` and `latent_state_for_demo_only` legitimately appear — as documented column *names*, not values; see Section 4).
- `methods.rules_only` / `methods.logistic_regression` / `methods.combined_policy` — precision, recall, F1, F2, PR-AUC, ROC-AUC (secondary), false-positive/negative rate, confusion matrix, predicted-review-case count, and the two synthetic-latent-state aggregate counts (seasonal-sale false positives, early-hidden-risk false negatives).
- `near_perfect_score_investigation` — gate status, the three conditions checked, and a notes string (never per-row detail).
- `limitations` — fixed, non-empty list, always present.

## 3. How the API reads it

`app/api/routes/metrics.py::get_metrics_route()`:

1. `ml.evaluation_report.load_report(DEFAULT_REPORT_PATH)` — raises `ReportNotFoundError` if the file doesn't exist, or lets `json.JSONDecodeError` propagate if the file is corrupt.
2. If either exception occurs, return `status: "not_available"` with `error_code: "METRICS_NOT_AVAILABLE"` (missing) or `"METRICS_ARTIFACT_INVALID"` (corrupt) — always **HTTP 200**, never a 500, and never the raw parse error or file path in the response body.
3. If the file loads, `ml.evaluation_report.validate_report(report)` re-checks the schema (required sections, `data_mode`, rate bounds, non-negative confusion-matrix counts, prohibited-string scan). Any issue also returns `status: "not_available"` / `"METRICS_ARTIFACT_INVALID"` — an invalid report is never returned to a caller, even if it parses as JSON.
4. Only a report that both loads and validates is mapped into `MetricsResponse` and returned as `status: "available"`.

## 4. Safe exposure boundaries

- The report is **aggregate-only** by construction — it is built from `evaluate_result["test_results"]`, which already contains only summary statistics (precision, recall, confusion-matrix counts, etc.), never a per-row dataframe. There is no code path from a raw merchant-week row into the report.
- `validate_report()` additionally scans the serialized report for: the five literal latent-state names (must never appear, anywhere), the five prohibited enforcement words, and a `merchant_demo_\d+`-shaped pattern (would indicate an accidental per-merchant ID leak). None of these are ever expected to appear, and their presence fails validation outright.
- `label_high_loss_next_30d` and `latent_state_for_demo_only` are a deliberate, narrow exception: the report schema requires listing them as strings inside `model.excluded_fields`, to document that they were excluded from the feature set — this is a column *name* used as documentation, not a per-row *value*. `validate_report()` checks this distinction explicitly: it re-serializes the report with `model.excluded_fields` removed and confirms neither string appears anywhere else.
- No raw model coefficients are included (only the derived precision/recall/etc. metrics and `feature_count`, an integer).
- `support_ticket_rate` (the Milestone 4 diagnostic-only feature) does not appear anywhere in the report — the report has no per-feature breakdown at all, only per-method aggregate metrics, so there is no mechanism by which a diagnostic-only feature name could be implied as a trusted signal.

## 5. Difference between this metrics artifact and live model monitoring

This is a **static, offline snapshot** — a JSON file written once per `python3 -m ml.evaluate_model` run and read verbatim by the API. It is not:

- **Live monitoring** — there is no continuous evaluation, no streaming metric computation, no dashboard that updates as new merchant-weeks arrive.
- **Drift detection** — the report does not compare the current model against a moving population; it reports one held-out test split's results as of the last offline run.
- **An alerting system** — nothing pages anyone if metrics degrade; a human must re-run the evaluation command and read the new report.

Producing genuine live monitoring (e.g. a scheduled re-evaluation job, a metrics time series, drift alerts) is out of scope for this MVP and would belong to a later phase, if ever required — consistent with `CLAUDE.md`'s "no background jobs, no automatic retraining" boundary for Phase 1.

## 6. Known limitations

- Synthetic data only — see the report's own `synthetic_data_notice` and `limitations` fields, always present in the response.
- Static offline report, not live monitoring (Section 5).
- No real payment gateway, merchant, settlement, enforcement, or financial-decision data is used anywhere in this pipeline.
- The report reflects whichever `python3 -m ml.evaluate_model` run last completed — if the underlying dataset, model, or rules change without re-running that command, `GET /metrics` will silently continue to serve the stale prior report rather than detecting staleness (no dataset/model version cross-check against the currently-loaded API state exists in this milestone).
