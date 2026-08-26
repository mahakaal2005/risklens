# Risk Policy — ClearRisk Recover Demonstration Policy

Status: Design proposal, rescoped 2026-08-22 to a single flagship scenario (merchant refund/chargeback loss risk) at merchant-week granularity. See `docs/RESCOPE_REVIEW.md`.

## Purpose

This document defines illustrative, synthetic-data-only rules for ClearRisk Recover's rules engine, and the recommendation policy that combines rule triggers with the ML model's `label_high_loss_next_30d` probability.

## Policy principles

1. Use the least harmful recommendation that fits the risk evidence.
2. Do not automatically impose financial restrictions.
3. Require human review before high-impact recommendations are acted on outside the MVP boundary.
4. Provide a safe explanation and evidence path.
5. Track reviewer outcomes to measure false positives and false negatives.

## Rule catalogue

All rules operate on a single merchant-week record. There are no transaction-level, device-level, or customer-dispute rules in Phase 1.

**Updated (Milestone 2, 2026-08-23):** the rule catalogue below reflects what is actually implemented in `rules/risk_rules.yaml` and `ml/rules_engine.py`. It supersedes the earlier illustrative `R_*` rule IDs drafted at rescope time, before real synthetic data or a feature-engineering module existed. Thresholds are no longer "pending sign-off" — they are frozen in `rules/risk_rules.yaml` and were chosen deliberately so that all 5 required Milestone 2 fixture scenarios (stable, seasonal-sale, operational-fulfilment-failure, high-risk, early-hidden-risk) produce the expected outcome; see `docs/MILESTONE_2_RULES_AND_FEATURES.md` for the full fixture walkthroughs and why the exact numbers remain illustrative, synthetic-data-only values rather than production thresholds.

| Rule ID | Scenario | Condition (see `rules/risk_rules.yaml` for exact thresholds) | Severity | Safe explanation |
|---|---|---|---|---|
| `REFUND_RATE_SPIKE` | Refund-rate spike | Refund rate increased materially versus the merchant's own prior 30-day period | Medium | "Your recent refund rate is higher than your usual pattern for this account." |
| `CHARGEBACK_RATE_SPIKE` | Chargeback-rate spike | Chargeback rate increased materially versus the merchant's own prior 30-day period | High | "Your recent chargeback rate is higher than your usual pattern for this account." |
| `EVIDENCE_COVERAGE_GAP` | Weak fulfilment evidence, in context | Delivery/fulfilment evidence coverage is low, and only counted alongside a refund or chargeback increase (never on its own) | Medium | "We were unable to confirm delivery/fulfilment evidence for a notable share of your recent disputed orders." |
| `SUPPORT_OPERATIONAL_STRESS` | Support-quality degradation, in context | Support ticket rate and resolution time are both elevated, and only counted alongside a refund or chargeback increase | Medium | "Your recent support ticket volume and response times are outside your usual pattern for this account." |
| `COMBINED_LOSS_SIGNAL` | Multiple co-occurring signals | Refund spike + chargeback spike, or chargeback spike + evidence gap | High | "Several account signals were flagged together this week, and a manual review is required before any further action." |

## Recommendation policy

**Updated (Milestone 2, 2026-08-23):** the recommendation is driven by which specific rules/severities triggered, not purely by the accumulated numeric score — this matters because several co-occurring *medium*-severity rules (e.g. the operational-fulfilment-failure fixture) can push the numeric score into "high" tier without representing the same concern level as an actual high-severity trigger, so the policy below deliberately checks rule identity first.

| Condition | Recommendation |
|---|---|
| `COMBINED_LOSS_SIGNAL` triggered AND `previous_review_outcome` is `confirmed_risk` (repeat offender) | `ESCALATE_TO_COMPLIANCE` |
| Any high-severity rule triggered (`CHARGEBACK_RATE_SPIKE` and/or `COMBINED_LOSS_SIGNAL`) | `MANUAL_REVIEW_REQUIRED` |
| Any medium-severity rule triggered, no high-severity trigger | `REQUEST_EVIDENCE` |
| Only a low-severity rule triggered (none currently defined) | `ALLOW_WITH_MONITORING` |
| No rule triggered | `APPROVE` |

This is a preliminary mapping — it may be refined once the Milestone 7 model exists and its probability is combined with rule severity. The recommendation enum permanently excludes `STEP_UP_VERIFICATION_RECOMMENDED` (approved 2026-08-22) — it was tied to the removed device/dispute scenario. The final Phase 1 enum is exactly: `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`.

## Reviewer actions

- `CLEAR_CASE`
- `MARK_FALSE_POSITIVE`
- `REQUEST_EVIDENCE`
- `MARK_OPERATIONAL_ISSUE`
- `ESCALATE_TO_COMPLIANCE`
- `MARK_INCONCLUSIVE`

## Merchant-safe communication standard

Merchant view must provide:

- Case status
- General reason category
- Evidence requested
- Review timeframe in the demo
- Appeal mechanism

Merchant view must not provide:

- Exact threshold values
- Raw model weights
- Internal fraud/loss patterns
- Other merchants' information
- Law-enforcement-sensitive detail
