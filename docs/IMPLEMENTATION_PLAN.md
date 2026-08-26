# Implementation Plan — ClearRisk Recover

Status: Design proposal. This plan sequences Phase 1 into small, independently testable milestones. **No application code has been written against this plan yet** — per the work plan, code starts only after this document is reviewed. Each milestone should be built, tested, and reported on (Implemented / Mocked / Not implemented, per `CLAUDE.md`'s required output style) before moving to the next.

Stale artifacts from the pre-rescope four-scenario design (`rules/risk_rules.yaml`, `ml/data_validation.py`, `ml/__init__.py`) will be fully rewritten, not patched, starting at Milestone 1/4.

---

## Milestone 1 — Synthetic merchant-week generator (5-state latent simulation)

**Goal:** Produce reproducible, seeded merchant-week records matching `DATA_DICTIONARY.md`'s schema, generated from the 5-state hidden latent-merchant simulation (not from rule thresholds).

**Files to create/change:**
- `ml/generate_synthetic_data.py` (full rewrite — currently does not exist for this scope)
- `demo_data/merchant_weeks.csv` (generated output, not hand-written)

**Acceptance criteria:**
- Running with `SYNTHETIC_DATA_SEED=42` produces the same output on repeat runs (byte-identical or hash-identical CSV).
- Every column in `DATA_DICTIONARY.md`'s merchant-week table is present with correct types.
- Merchant state persists across consecutive weeks (not resampled independently every week) — verified by checking autocorrelation of state assignment.
- Overall and per-state `label_high_loss_next_30d` positive rates fall within the documented ranges in `MODEL_CARD.md` (e.g. Stable ~1-2%, Seasonal ~8-12%, Operational-failure ~30-40%, High-risk ~70-85%, Early-hidden-risk ~45-60%), within a reasonable statistical tolerance.
- The trailing ~4-5 weeks of history (incomplete label horizon) are excluded or null-labeled, never defaulted to 0.
- `label_high_loss_next_30d` is demonstrably **not** a deterministic function of `refund_rate_30d`/`chargeback_rate_30d` alone (verified by Milestone 1's own test: find at least one high-refund/high-chargeback row labeled 0, and at least one low-refund/low-chargeback row labeled 1).

**Tests (`tests/test_generate_synthetic_data.py`):**
- Same seed → identical output (reproducibility).
- All required columns present, correct dtypes, no prohibited field names.
- Per-state label-rate check against documented ranges (with tolerance).
- Non-circularity check: label is not recoverable via a simple threshold rule on refund/chargeback fields alone (fit a trivial threshold classifier on those two fields only and confirm it does *not* achieve near-perfect precision/recall).
- Trailing-week label-horizon exclusion check.

**Expected demo output:** `demo_data/merchant_weeks.csv` (~150-300 merchants × 52-104 weeks), plus a printed console summary: row count, date range, label distribution overall and per latent state, state-transition persistence rate.

---

## Milestone 2 — Data validation

**Goal:** Reject malformed merchant-week records with a clear validation error; scan for prohibited fields.

**Files to create/change:**
- `ml/data_validation.py` (full rewrite — replaces the old `MerchantRecord`/`TransactionRecord` schemas with a single `MerchantWeekRecord` schema)

**Acceptance criteria:**
- Pydantic model with `extra="forbid"`, correct enums (`merchant_category`, `top_dispute_reason_category`, `previous_review_outcome`), and range constraints (rates in [0,1], non-negative counts, `merchant_age_days >= 0`).
- `assert_no_prohibited_fields` reused/adapted from the existing prohibited-field-name list.
- The full Milestone 1 output validates cleanly with zero errors.
- A hand-built malformed record (e.g. `refund_rate_30d = 1.5`, or a prohibited field name) raises `DataValidationError` with a message identifying the offending field.

**Tests (`tests/test_data_validation.py`):**
- Valid record passes.
- Out-of-range rate field raises.
- Missing required field raises.
- Extra/unexpected field raises (`extra="forbid"`).
- Prohibited field name raises before Pydantic even runs.

**Expected demo output:** running validation over `demo_data/merchant_weeks.csv` reports "N/N records valid" or a precise list of failures.

---

## Milestone 3 — Feature engineering

**Goal:** Turn a raw merchant-week row (plus its own merchant's recent history, if needed) into the flat feature dict the rules engine and model both consume.

**Files to create/change:**
- `ml/features.py` (new — the "feature builder" component named in `ARCHITECTURE.md`)

**Acceptance criteria:**
- Given a validated merchant-week record, returns a dict with all fields needed by the rule catalogue in `RISK_POLICY.md` (refund/chargeback rate + change, evidence coverage, support metrics, previous outcome) — no additional joins needed since the merchant-week schema is already flat and self-contained.
- Function is pure (no I/O), unit-testable on a hand-built fixture.

**Tests (`tests/test_features.py`):**
- Hand-built input row → expected feature dict, exact match.

**Expected demo output:** printed feature dict for one sample merchant-week row.

---

## Milestone 4 — Rules engine (refund/chargeback only)

**Goal:** Implement the 5-rule catalogue from `RISK_POLICY.md` (`R_REFUND_SPIKE`, `R_CHARGEBACK_SPIKE`, `R_EVIDENCE_COVERAGE_DROP`, `R_SUPPORT_DEGRADATION`, `R_REPEAT_REVIEW`) against merchant-week features.

**Files to create/change:**
- `rules/risk_rules.yaml` (full rewrite — replaces the old R1/R2/R3/R4 transaction-level rules)
- `ml/rules_engine.py` (new)

**Acceptance criteria:**
- Numeric thresholds are derived from Milestone 1's actual generated data distribution (e.g. "materially higher" for refund-rate change is set using the observed gap between the Stable and Operational-failure state distributions), not guessed before data exists — this closes the "pending sign-off" note in `RISK_POLICY.md`.
- Each rule can be independently triggered and independently not-triggered via constructed fixtures.
- Triggered-rule output includes `rule_id`, `severity`, `reason_category`, and `safe_explanation` with the actual before/after values interpolated in (e.g. "increased from 0.4% to 2.2%"), per FR-6.
- No rule references any field outside the merchant-week schema (no transaction/device/customer fields).

**Tests (`tests/test_rules_engine.py`):**
- One trigger-case and one non-trigger-case per rule (10 tests minimum).
- One combined-trigger case (multiple rules fire on the same row) verifying all expected rules appear and none extra.
- `R_REPEAT_REVIEW` specifically tested against a `previous_review_outcome = confirmed_risk` fixture combined with another triggering rule.

**Expected demo output:** printed rule-trigger results for a handful of sample merchant-weeks spanning all 5 latent states.

---

## Milestone 5 — Rules-only policy and recommendation (baseline #1)

**Goal:** Combine triggered rules into a `risk_score`/`risk_tier`/recommendation using the final 5-value enum, with no ML involved yet — this is the "rules-only baseline" required to be reported first in `MODEL_CARD.md`'s evaluation order.

**Files to create/change:**
- `ml/policy.py` (new)

**Acceptance criteria:**
- Recommendation is always one of exactly: `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE` — never anything else, and never an enforcement action.
- Deterministic given the same triggered rules and score (no hidden randomness).
- Computing rules-only precision/recall/confusion matrix against Milestone 1's labels on a small held-out slice produces a plausible (imperfect) result.

**Tests (`tests/test_policy.py`):**
- Recommendation ladder produces the expected value for each condition row in `RISK_POLICY.md`'s recommendation-policy table.
- Confirms the enum never includes `STEP_UP_VERIFICATION_RECOMMENDED` or any enforcement action, by exhaustively checking the allowed-value set.

**Expected demo output:** printed rules-only baseline metrics (precision, recall, confusion matrix) on a small held-out sample — the first number that will later be compared against the ML-augmented policy.

---

## Milestone 6 — Time-based split

**Goal:** Split merchant-weeks into train/validation/held-out-test by `week_start`, excluding label-horizon-incomplete trailing weeks.

**Files to create/change:**
- `ml/split.py` (new)

**Acceptance criteria:**
- Split is global by `week_start` (earliest ~60% / next ~20% / latest ~20%), not per-merchant shuffling.
- No merchant-week appears in more than one split.
- Trailing weeks without a complete 30-day forward label window are excluded from all three splits.
- Split boundary dates are printed/logged for later dashboard display (per PRD FR-10's "training/validation/test time periods are displayed").

**Tests (`tests/test_split.py`):**
- Split proportions are approximately correct.
- No `week_start` value appears in two splits.
- A synthetically-constructed trailing incomplete week is confirmed excluded.

**Expected demo output:** printed date ranges and row counts for each of the three splits.

---

## Milestone 7 — Logistic Regression baseline (baseline #2)

**Goal:** Train an interpretable Logistic Regression model on `label_high_loss_next_30d`, select an operating threshold on validation data, and version the artifact.

**Files to create/change:**
- `ml/train.py` (rewrite/implement)
- `ml/artifacts/` — versioned model file + feature list (generated, e.g. `model_v0.1.0.joblib`, `feature_list_v0.1.0.json`)

**Acceptance criteria:**
- Trains only on the training split from Milestone 6.
- Threshold is chosen using the validation split only — the held-out test split is never touched during model choice or threshold tuning.
- Model artifact and the exact feature list are saved together with a version string.
- If the artifact is missing/unreadable at scoring time, the system falls back to rules-only mode (Milestone 5's policy) — this fallback path must be demonstrated, not just asserted.

**Tests (`tests/test_train.py`):**
- Smoke test: training on a small fixture dataset completes without error and produces well-formed probabilities in [0,1].
- Missing-artifact fallback test: scoring with an intentionally deleted/corrupted artifact path falls back to rules-only and logs the fallback.

**Expected demo output:** printed validation-split metrics used to justify the chosen operating threshold.

---

## Milestone 8 — Held-out evaluation and near-perfect-score investigation

**Goal:** Compute the full required metric set on the held-out test split only, compare rules-only vs. rules+ML, and run the near-perfect-score investigation checklist automatically when triggered.

**Files to create/change:**
- `ml/evaluate.py` (new)

**Acceptance criteria:**
- Reports: precision, recall, PR-AUC, false-positive rate, confusion matrix, precision/recall at the selected operating threshold, rules-only vs. rules+ML comparison, metrics segmented by latent state and `merchant_category` where sample size allows.
- Surfaces at least 3 concrete false-positive examples and at least 2 concrete false-negative examples, pulled from the actual held-out set (not fabricated).
- If PR-AUC >= 0.98, precision >= 0.98 and recall >= 0.98, or zero false positives/negatives occur, the result is marked `"Under investigation"` and the checklist from `MODEL_CARD.md` (label leakage, time leakage, split integrity, entity duplication, latent-state overlap, intentional-case presence, rules-vs-model comparison) is run and its outcome recorded — this must not be silently skipped even if the number looks good.
- Displays the required synthetic-data limitation banner text verbatim.

**Tests (`tests/test_evaluate.py`):**
- Metric computation verified against a hand-built confusion-matrix fixture with known precision/recall/PR-AUC.
- Near-perfect trigger correctly fires on an intentionally all-correct fixture and correctly does *not* fire on a normal fixture.

**Expected demo output:** a saved evaluation report (JSON and/or Markdown) containing every required metric, the example rows, the train/val/test date ranges, and — if applicable — the investigation section with its outcome.

---

## Milestone 9 — Explanation generator

**Goal:** Convert triggered rules and top model factors into safe, plain-language explanation text with concrete before/after values, in an analyst variant and a merchant-safe variant.

**Files to create/change:**
- `ml/explain.py` (new)

**Acceptance criteria:**
- Every explanation for a rule-triggered case includes at least one concrete numeric before/after pair (matches the FR-6 example format in `PRD.md`).
- Merchant-safe variant never includes exact thresholds, raw coefficients, or internal severity weights.
- Explanation text is deterministic given the same triggered rules and feature values (for testability).

**Tests (`tests/test_explain.py`):**
- Given a fixture with `R_CHARGEBACK_SPIKE` and `R_REFUND_SPIKE` triggered, output contains both trend sentences with the correct before/after numbers.
- Merchant-safe variant is checked against a denylist of forbidden substrings (exact threshold numbers, "coefficient", "weight").

**Expected demo output:** printed example explanation matching the format in `PRD.md` FR-6.

---

## Milestone 10 — SQLite persistence + FastAPI scoring/case endpoints

**Goal:** Persist merchant-weeks, risk assessments, cases, evidence, reviewer decisions, and audit events; expose scoring and case-management via FastAPI.

**Files to create/change:**
- `app/db/` — SQLAlchemy models and session setup
- `app/models/` — ORM entity definitions (`MerchantWeek`, `RiskAssessment`, `ReviewCase`, `EvidenceSubmission`, `ReviewerDecision`, `AuditEvent`, `ModelRun`)
- `app/schemas/` — Pydantic request/response schemas
- `app/api/` — scoring endpoint, case-queue/detail endpoints, reviewer-action endpoint, appeal-submission endpoint
- `app/main.py` — FastAPI app wiring

**Acceptance criteria:**
- Scoring a merchant-week persists a `RiskAssessment` and, for medium/high tier, a `ReviewCase`, plus an `AuditEvent` for every key step (per FR-9's event list).
- Reviewer-action endpoint requires a note and rejects the request otherwise.
- Appeal-submission endpoint accepts free text + fake evidence filenames only — no real file upload.
- Database-unavailable and invalid-schema failure modes match `ARCHITECTURE.md` Section 7 (safe error, no false claim of a saved decision).

**Tests (`tests/test_api_scoring.py`, `tests/test_api_cases.py`):**
- FastAPI `TestClient`/`httpx` tests covering: score → case created → reviewer action → appeal → resolution, and the full audit trail is retrievable afterward.
- Validation-error path for a malformed scoring request.
- Reviewer-action-without-note rejection.

**Expected demo output:** an example request/response walkthrough (e.g. via `httpx` or `curl`) showing a merchant-week scored, a case created, and its audit trail retrieved.

---

## Milestone 11 — Streamlit dashboard

**Goal:** Build the 6 pages defined in `docs/RESCOPE_REVIEW.md` Section 6: merchant-week feed, risk detail, review case, merchant appeal, audit timeline, risk-manager metrics.

**Files to create/change:**
- `dashboard/streamlit_app.py` (+ page modules as needed)

**Acceptance criteria:**
- Each page renders against the FastAPI backend/SQLite data without error.
- The synthetic-data limitation banner is always visible on the metrics page.
- The rules-only vs. rules+ML comparison and the false-positive/false-negative examples from Milestone 8 are visibly displayed.

**Tests:** Streamlit UIs are not practically unit-tested; this milestone's verification is a manual local run per `CLAUDE.md`'s build-verification rule ("run it locally where possible, report what was actually executed") — screenshots or a short walkthrough substituting for automated tests, explicitly labeled as such.

**Expected demo output:** a working local Streamlit session matching `docs/DEMO_SCRIPT.md`'s walkthrough steps 2-7.

---

## Milestone 12 — End-to-end test and demo run

**Goal:** Exercise the complete Flow A / Flow B / Flow C (from `PRD.md` Section 9) against a temporary SQLite database, confirming the MVP definition of done.

**Files to create/change:**
- `tests/test_end_to_end.py`

**Acceptance criteria:**
- Full merchant-week → rules+model score → case → reviewer action → merchant appeal → resolution → audit-timeline flow passes against a clean temp database.
- Dashboard metrics page reflects a real (not mocked) held-out evaluation run.
- Every item in `CLAUDE.md`'s "Definition of done for MVP" checklist is demonstrably true.

**Expected demo output:** this milestone is the actual demo — the point at which `docs/DEMO_SCRIPT.md` can be performed live and matches an existing, working system.

---

## Recommended first implementation files

In order, the exact files to create next (Milestone 1 and 2 — data must exist and validate before anything else can be built or tested against it):

1. `ml/generate_synthetic_data.py`
2. `ml/data_validation.py` (rewrite, replacing the current stale transaction+merchant version)
3. `tests/test_generate_synthetic_data.py`
4. `tests/test_data_validation.py`

`ml/__init__.py` already exists and needs no change. `rules/risk_rules.yaml` is Milestone 4's concern, not Milestone 1's — its thresholds depend on having real generated data to derive them from, per Milestone 4's acceptance criteria.
