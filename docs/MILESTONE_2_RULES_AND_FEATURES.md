# Milestone 2 — Feature Engineering and Transparent Rules Engine

Status: Design proposal / implemented for Milestone 2 scope only. This document describes `ml/features.py`, `rules/risk_rules.yaml`, and `ml/rules_engine.py`. No Logistic Regression training pipeline, FastAPI, SQLite, Streamlit, case workflow, merchant appeal, dashboard, SHAP, or external integration exists yet — those are later milestones.

**Rules identify review signals. They do not establish fraud, justify a fund hold, or make an account termination decision.**

---

## 1. Feature catalogue

All features are computed by `ml/features.py::compute_features()` (single row) or `compute_feature_frame()` (vectorized). Both defensively exclude `label_high_loss_next_30d` and `latent_state_for_demo_only` even if present on the input.

| Feature | Formula / source | Meaning | Range / type | Missing-value behavior | Used by |
|---|---|---|---|---|---|
| `refund_rate_change` | `refund_rate_30d - refund_rate_previous_30d` | Absolute change in refund rate vs. the merchant's own prior 30-day window | ~-1.0 to 1.0, usually small | `0.0` if either input rate is missing | both |
| `chargeback_rate_change` | `chargeback_rate_30d - chargeback_rate_previous_30d` | Absolute change in chargeback rate vs. prior window | ~-1.0 to 1.0, usually small | `0.0` if either input rate is missing | both |
| `transaction_volume_change` | `(volume_30d - volume_previous_30d) / volume_previous_30d` | Relative change in payment volume | ~-1.0 to arbitrarily large | `0.0` if prior volume is zero/missing (avoids divide-by-zero) | ML |
| `refund_to_chargeback_ratio` | `refund_rate_30d / chargeback_rate_30d`, capped at 50 | How refund-driven vs. chargeback-driven the loss pattern is; high values suggest legitimate-return behavior rather than chargeback loss | 0-50, or missing | **Missing (`None`)** if `chargeback_rate_30d < 0.002` — not numerically stable that close to zero, so it is left unset rather than reported as a misleading extreme value ("where safe", per the Milestone 2 requirement) | ML |
| `delivery_evidence_gap` | `1 - delivery_evidence_coverage` | Fraction of evidence missing | 0.0-1.0 | Omitted if raw field missing | both |
| `support_resolution_hours_normalized` | `min(hours, 100) / 100` | Support resolution time scaled ~0-1 for a linear model | 0.0-1.0 | Omitted if raw field missing | ML |
| `support_resolution_risk_band` | `low` if <24h, `medium` if <48h, else `high` | Categorical band for analyst-facing readability | `low` / `medium` / `high` | Omitted if raw field missing | rules (documentation only; rules compare against raw hours directly) |
| `support_ticket_rate` | Passthrough | Support tickets per transaction, trailing 30 days | 0.0-~0.2 in this dataset | Passed through as-is | both |
| `merchant_age_days` | Passthrough | Days since onboarding as of `week_start` | Non-negative integer | Passed through as-is | ML |
| `merchant_age_band` | `new` if <90 days, `growing` if <365, else `established` | Categorical age band for analyst readability | `new` / `growing` / `established` | Omitted if raw field missing | rules |
| `transaction_count_30d` | Passthrough | Trailing-30-day transaction count, scale context for rate features | Non-negative integer | Passed through as-is | both |
| `merchant_category` | Passthrough | Descriptive segmentation only — see note below | One of 6 categories | Passed through as-is; one-hot encoding happens at Milestone 7 training time, not here | ML |
| `previous_review_outcome` | Passthrough | Outcome of the merchant's most recent PRIOR review, if any | One of 5 outcomes | Passed through as-is | both |

**Note on `merchant_category`:** included as a legitimate control feature (baseline refund/return norms genuinely differ by category, e.g. apparel vs. digital_services) — not a profile-mismatch signal. Per the 2026-08-22 rescope decision, declared-vs-observed category mismatch was dropped from Phase 1 entirely; no "observed_category" field exists anywhere in this codebase to compare against.

**Note on `previous_review_outcome` time-safety:** it reflects a review that concluded *before* the current prediction week, so it cannot leak this week's own outcome. Known simplification (documented in the Milestone 1 data-quality report): the synthetic generator currently samples this field independently per week from state-conditioned weights rather than tracking an actual persistent review history across weeks. It is time-safe but not yet a fully realistic time series — acceptable for Milestone 2, worth revisiting when the case-review workflow (a later milestone) exists.

**Fields deliberately excluded** from the feature set (not features, not rule inputs): `merchant_id`, `week_start` (identifiers/time keys), `transaction_volume_30d`/`transaction_volume_previous_30d`/`refund_count_30d`/`chargeback_count_30d`/`refund_rate_previous_30d`/`chargeback_rate_previous_30d` (superseded by the derived rate/change fields), `top_dispute_reason_category` (not in the frozen candidate list), and of course `label_high_loss_next_30d` and `latent_state_for_demo_only`.

---

## 2. Rule catalogue

Full rule definitions live in `rules/risk_rules.yaml`; this is the narrative summary. All thresholds below are read from that YAML file at runtime — nothing is hard-coded in `ml/rules_engine.py`.

### `REFUND_RATE_SPIKE` (severity: medium)
Triggers when `refund_rate_change >= 0.02` (absolute) OR current rate `>= 1.5x` the prior rate. Recommends `REQUEST_EVIDENCE`. On its own, this is treated as an early, ambiguous signal (could be a legitimate return spike), not an automatic escalation.

### `CHARGEBACK_RATE_SPIKE` (severity: high)
Triggers when `chargeback_rate_change >= 0.01` (absolute) OR current rate `>= 1.5x` the prior rate. Recommends `MANUAL_REVIEW_REQUIRED`. High severity because a chargeback is a confirmed cardholder dispute with direct financial/network consequences, unlike a merchant-initiated refund.

### `EVIDENCE_COVERAGE_GAP` (severity: medium)
Triggers only when `delivery_evidence_gap >= 0.35` **and** a supporting refund (`>=0.015`) or chargeback (`>=0.005`) change is also present. Never triggers on evidence coverage alone — per the Milestone 2 requirement, since many legitimate merchants simply keep lighter records.

### `SUPPORT_OPERATIONAL_STRESS` (severity: medium)
Triggers only when `support_ticket_rate >= 0.03` **and** `average_support_resolution_time_hours >= 40`, **and** a supporting refund (`>=0.01`) or chargeback (`>=0.003`) change is present.

### `COMBINED_LOSS_SIGNAL` (severity: high)
Triggers when either `(REFUND_RATE_SPIKE and CHARGEBACK_RATE_SPIKE)` or `(CHARGEBACK_RATE_SPIKE and EVIDENCE_COVERAGE_GAP)` co-occur. Represents a stronger, harder-to-explain-away pattern than any single rule alone.

Every rule returns the schema specified in Milestone 2: `rule_id`, `rule_version`, `triggered`, `severity`, `score_contribution`, `analyst_explanation`, `merchant_safe_explanation`, `suggested_evidence`, `recommended_action`, `input_snapshot` (containing only the specific fields that rule reads — never the label or latent state).

---

## 3. Preliminary score / risk-band mapping

- **Score:** sum of `severity_weights` (`low=10, medium=30, high=50`) for every triggered rule, capped at 100.
- **Risk tier:** `low` if score `<30`, `medium` if `<70`, else `high` (`risk_tier_thresholds` in the YAML).
- **Recommendation** (rule-identity-driven, evaluated top-down — see `recommendation_policy` in `rules/risk_rules.yaml`):
  1. `COMBINED_LOSS_SIGNAL` triggered **and** `previous_review_outcome == confirmed_risk` → `ESCALATE_TO_COMPLIANCE`
  2. Any high-severity rule triggered (`CHARGEBACK_RATE_SPIKE` and/or `COMBINED_LOSS_SIGNAL`) → `MANUAL_REVIEW_REQUIRED`
  3. Any medium-severity rule triggered → `REQUEST_EVIDENCE`
  4. Any low-severity rule triggered (none currently defined) → `ALLOW_WITH_MONITORING`
  5. Otherwise → `APPROVE`

**Why rule-identity-driven, not purely score-driven:** three co-occurring medium-severity rules (as in the operational-fulfilment-failure fixture) can push the numeric score to "high" tier (90) without representing a chargeback-confirmed or multi-signal-combined pattern. Recommending `REQUEST_EVIDENCE` in that case — not `MANUAL_REVIEW_REQUIRED` — better matches the "least harmful recommendation that fits the risk evidence" policy principle. The numeric score/tier is still reported (useful for dashboards and later comparison against the ML model), but it does not solely drive the recommendation.

This mapping is preliminary and may be refined once the Milestone 7 Logistic Regression model exists and its probability is combined with rule severity.

---

## 4. Thresholds, and why they are illustrative

Every numeric threshold in `rules/risk_rules.yaml` (e.g. `absolute_increase_min: 0.02` for refund spikes, `evidence_gap_min: 0.35`) was chosen to make the 5 required Milestone 2 fixtures behave as specified — they are **not** derived from any real merchant population, and they are **not** claimed to be optimal even for this synthetic dataset. They are synthetic-data-only, demonstration thresholds, consistent with `SECURITY.md`'s requirement that UI/documentation state "Synthetic data / demonstration only," and with `CLAUDE.md`'s truthfulness rule against exposing fraud thresholds as if they were operationally validated. Changing any threshold is a one-line YAML edit — see `rules/README.md`.

---

## 5. Fixture walkthroughs

All five fixtures live in `tests/test_rules_engine.py::FIXTURES`. Actual output (from `python3 -m ml.rules_engine` — see Section "Actual output" in the deliverable report below) for each:

| Fixture | Triggered rules | Score | Tier | Recommendation |
|---|---|---|---|---|
| `stable_merchant` | none | 0 | low | `APPROVE` |
| `seasonal_sale_false_positive_candidate` | `REFUND_RATE_SPIKE` | 30 | medium | `REQUEST_EVIDENCE` |
| `operational_fulfilment_failure_case` | `REFUND_RATE_SPIKE`, `EVIDENCE_COVERAGE_GAP`, `SUPPORT_OPERATIONAL_STRESS` | 90 | high | `REQUEST_EVIDENCE` |
| `high_risk_merchant_case` | `REFUND_RATE_SPIKE`, `CHARGEBACK_RATE_SPIKE`, `EVIDENCE_COVERAGE_GAP`, `SUPPORT_OPERATIONAL_STRESS`, `COMBINED_LOSS_SIGNAL` | 100 | high | `MANUAL_REVIEW_REQUIRED` |
| `early_hidden_risk_case` | none | 0 | low | `APPROVE` |

**Why each is correct:**
- **Stable merchant:** every input sits near baseline (refund/chargeback rates flat, high evidence coverage, low support load) — no rule has reason to fire.
- **Seasonal-sale false-positive candidate:** refund rate jumps from 2.0% to 6.0% (triggers `REFUND_RATE_SPIKE`), but chargeback rate barely moves and evidence coverage stays high (0.88) — so `EVIDENCE_COVERAGE_GAP` correctly does *not* fire even though a supporting refund signal exists, because the gap itself (0.12) is below the 0.35 threshold. Result: a reviewable evidence request, not an automatic high-risk conclusion.
- **Operational fulfilment failure:** refund spike (2.5%→5.0%) plus a real evidence gap (coverage 0.60, gap 0.40 ≥ 0.35, with a supporting refund signal) plus support stress (ticket rate 4.5%, resolution 55h, supporting signal present) — three independent medium concerns, no chargeback spike, so `COMBINED_LOSS_SIGNAL` does not fire. Numeric tier is "high" (90) but the recommendation correctly stays `REQUEST_EVIDENCE` per Section 3's rule-identity-driven policy.
- **High-risk merchant:** both refund (2.0%→7.0%) and chargeback (1.0%→3.0%) spike, evidence coverage is very low (0.30), support load is high — `COMBINED_LOSS_SIGNAL` fires via both required pairs. `previous_review_outcome` is `none` (not `confirmed_risk`), so the recommendation lands on `MANUAL_REVIEW_REQUIRED`, not `ESCALATE_TO_COMPLIANCE`.
- **Early hidden-risk merchant:** every input is mild (refund 1.8%→2.5%, chargeback 0.5%→0.6%, evidence coverage 0.80, support load modest) — below every threshold, so nothing triggers. This is the fixture that proves the rules are not perfect: in the full synthetic dataset, this latent state carries a genuinely elevated future-loss probability (~45-60%, per `MODEL_CARD.md`) despite looking unremarkable this week. The rules engine cannot see that; only the Milestone 7 model, trained on the label, has a chance of catching it.

---

## 6. Analyst-safe vs. merchant-safe explanation example

For the `high_risk_merchant_case` fixture, `CHARGEBACK_RATE_SPIKE`:

- **Analyst explanation:** "Chargeback rate increased from 1.00% to 3.00% (2.00% change)."
- **Merchant-safe explanation:** "Your recent chargeback rate is higher than your usual pattern for this account."

The analyst version includes the concrete before/after numbers (per `PRD.md` FR-6); the merchant-safe version states only a general reason category, per `RISK_POLICY.md`'s merchant-safe communication standard — no exact threshold values, no raw scores, no other merchants' information.

---

## 7. What rules cannot determine

- Whether a flagged pattern reflects actual fraud, an honest operational failure, or a legitimate business event (e.g. a sale) — a rule trigger is a signal for a human reviewer, not a verdict.
- Anything about a merchant-week where every observed signal looks normal but the underlying (hidden, unobservable) risk is elevated — the `early_hidden_risk_case` fixture demonstrates this directly. Only a model trained on outcomes (Milestone 7) has a chance of catching this, and even then imperfectly (per `MODEL_CARD.md`'s honesty requirements).
- Intent, real-world financial impact, or anything requiring information outside this merchant-week's own recent history (no cross-merchant, cross-network, or gateway-side signals exist in this MVP).

## 8. Statement

**Rules identify review signals. They do not establish fraud, justify a fund hold, or make an account termination decision.**
