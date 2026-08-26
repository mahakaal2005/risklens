# Milestone 4 — Explainability and Reviewer Case-Packet Generation

Status: Implemented and run against real held-out data (seed=42). No LLM call and no agent is used anywhere in this milestone — every explanation sentence comes from a fixed template filled in with actual model/rule values.

**Case packets are an in-memory/JSON demonstration artifact only. No real case is created in any database, and no action is taken against any merchant.**

---

## 1. Explanation design

Two independent explanation sources are combined into one packet:

1. **Rule explanations** — each triggered rule (`ml/rules_engine.py`) already carries a template-filled `analyst_explanation` (with concrete before/after values) and a separate `merchant_safe_explanation`, built in Milestone 2.
2. **Model factor explanations** — `ml/explain_cases.py::compute_top_factors()` reads the fitted Logistic Regression's own coefficients and each row's transformed feature values directly from the trained pipeline; nothing here is a separate/approximate importance measure.

`ml/case_packet.py::build_case_packet()` merges both into the schema documented in `docs/CASE_PACKET_SCHEMA.md`.

## 2. Top-factor methodology

For a single merchant-week's transformed feature vector `x` and the model's coefficient vector `w`, each feature's contribution to the logit is `w_i * x_i`. `compute_top_factors()`:

1. Runs the row through the fitted `ColumnTransformer` to get the exact transformed values the model actually saw (standardized numeric values, one-hot categorical indicators).
2. Multiplies elementwise by the fitted `LogisticRegression` coefficients.
3. Drops zero-contribution entries (inactive one-hot categories contribute exactly 0).
4. Sorts by `|contribution|` descending and keeps the top 5.
5. Converts each into a template sentence:
   - Numeric: `"{Higher/Lower} than usual {feature label} contributed to {the elevated/a lower} score."` — "higher/lower than usual" is derived from the standardized value's sign (above/below the training mean), and "elevated/lower" from the contribution's sign. This is a small, honest refinement of the Milestone 4 example sentence ("Higher refund-rate change contributed to the elevated score.") — "than usual" is added because "higher" is meaningless without a reference point, and the reference point actually used is the training-data mean, not an arbitrary claim.
   - Categorical: `"{Merchant category / Previous review outcome} '{value}' contributed to a {higher/lower} score."`

No causal language is used anywhere (never "caused," "confirmed," or "proves") — only "contributed to."

**Observed, honest finding:** some coefficients are directionally counter-intuitive relative to naive human expectation — e.g. in the trained model, `support_ticket_rate` has a *negative* coefficient (higher ticket rate slightly lowers the predicted score), likely reflecting correlations with other features in this synthetic dataset (e.g. established merchants with more support volume overall).

### Explainability-quality policy (added post-Milestone-4 review)

> A model feature may be shown as a natural-language positive or negative risk factor only when its observed direction is plausible, stable across validation/test or sensitivity checks, and does not contradict the documented risk-policy interpretation. Otherwise it is diagnostic-only and excluded from ranked explanations. The model may continue to use it, but its direction must be documented as unvalidated.

**Implementation:** `ml/explain_cases.py::DIAGNOSTIC_ONLY_FEATURES = {"support_ticket_rate"}`. `compute_top_factors()` skips any feature in this set before ranking — it never appears in `top_model_factors`, and therefore never in analyst or merchant-safe explanation text. The feature is **not** removed from the model itself: it remains in `ML_FEATURE_COLUMNS`, is still fit and used for scoring, and its counter-intuitive coefficient is still documented in `MODEL_CARD.md` as an unvalidated-direction limitation. This is a presentation-layer exclusion, not a model change — the reasoning is that `support_ticket_rate`'s negative coefficient contradicts `RISK_POLICY.md`'s `SUPPORT_OPERATIONAL_STRESS` rule (which treats rising support load as a risk signal, not a risk-reducing one), and it has not passed a sensitivity/stability check, so it fails the policy above and stays diagnostic-only until it does. Verified by `tests/test_explanation_safety.py`.

## 3. Rules versus model explanation

| | Rule explanations | Model factor explanations |
|---|---|---|
| Source | Fixed template + actual field values (Milestone 2) | Fitted coefficients × transformed values (this milestone) |
| Always available? | Yes, whenever `ml/rules_engine.py` runs | No — requires a loaded model artifact; falls back to degraded mode otherwise |
| Granularity | Rule-level (5 possible triggers) | Feature-level (11 features, up to 20 after one-hot encoding) |
| Shown to merchant? | Yes (via `merchant_safe_explanation`, rule text only) | No — top model factors are analyst-only |

## 4. Degraded mode

If no model artifact/pipeline is supplied to `build_case_packet()` (`pipeline=None`) or no probability is available (`ml_probability=None`), the packet sets `assessment.degraded_mode = true`, `analyst_explanation.top_model_factors = []`, and `analyst_explanation.uncertainty_statement` switches to an explicit degraded-mode sentence stating the assessment used the rules engine only. The recommendation and evidence checklist still work correctly in this mode, since they depend only on rule triggers, not the model. Verified by `tests/test_case_packet.py::test_degraded_mode_without_model_artifact`.

## 5. Privacy and safe-language controls

- `build_case_packet()` strips `label_high_loss_next_30d` and `latent_state_for_demo_only` from the input record before doing anything else, and `_assert_packet_is_safe()` re-checks the fully serialized packet JSON afterward and raises if either ever appears.
- `merchant_safe_explanation` is scanned for a forbidden-term list (`fraud confirmed`, `ban`, `freeze`, `terminate`, `latent`, `probability`) and raises if any appear.
- `merchant_id` is included only as an identification field — never interpolated into analyst or merchant-safe narrative sentences.
- Evidence checklists are only ever populated from actually-triggered rules — a stable merchant-week gets an empty checklist, not a generic one.

## 6. Demo-case walkthroughs

Five packets generated from real held-out test data (`python3 -m ml.generate_demo_cases`), selected using `latent_state_for_demo_only`/`label_high_loss_next_30d` internally (never placed in packet text):

1. **Stable merchant** (`merchant_demo_XXXX`, low intensity, `APPROVE`) — no rules triggered, near-zero model probability, empty evidence checklist.
2. **Seasonal-sale false-positive candidate** (Medium intensity, `REQUEST_EVIDENCE`) — only `REFUND_RATE_SPIKE` triggered; merchant-safe explanation states only "Your recent refund rate is higher than your usual pattern" — non-accusatory, no chargeback/evidence-gap language; evidence checklist correctly omits delivery-proof items (refund-only pattern).
3. **Operational fulfilment problem** (High intensity, `REQUEST_EVIDENCE` — not `MANUAL_REVIEW_REQUIRED`, preserving the Milestone 2/3 rule that high signal intensity does not automatically mean manual review) — `EVIDENCE_COVERAGE_GAP` + `SUPPORT_OPERATIONAL_STRESS` triggered; evidence checklist focuses on fulfilment/delivery proof and support records.
4. **High-risk combined-loss case** (High intensity, `MANUAL_REVIEW_REQUIRED`) — all 5 rules triggered including `COMBINED_LOSS_SIGNAL`; full evidence checklist; explanation covers refund, chargeback, and evidence-gap trends together.
5. **Early-hidden-risk case** (Medium intensity, `ALLOW_WITH_MONITORING`) — zero rules triggered, model probability 0.081 (just under the 0.10 threshold), yet this merchant-week's true simulated outcome was a later high-loss state. The uncertainty statement is present verbatim, demonstrating a low/medium score is not a guarantee.

Full JSON for all five: `demo_data/demo_case_packets.json`.

### One full analyst explanation example (high-risk combined-loss case)

> Triggered rule(s): REFUND_RATE_SPIKE, CHARGEBACK_RATE_SPIKE, EVIDENCE_COVERAGE_GAP, SUPPORT_OPERATIONAL_STRESS, COMBINED_LOSS_SIGNAL. Higher than usual delivery-evidence gap contributed to the elevated score. Recommended action: MANUAL_REVIEW_REQUIRED. This reflects rule and model signals only, not a confirmed outcome.
>
> **Top model factors:** delivery_evidence_gap (+6.49), support_ticket_rate (−1.95), previous_review_outcome_confirmed_risk (−0.61), chargeback_rate_change (−0.55), refund_rate_change (+0.33).
>
> **Uncertainty statement:** "This assessment reflects synthetic-data model and rule signals as of the prediction date only; it does not guarantee a future outcome and is not a confirmed finding of fraud."

### One full merchant-safe explanation example (same case)

> **Reason category:** combined account review
> **Review signal statement:** "This is a review signal only. It is not a final determination of fraud, and no action has been taken against your account."
> **Reasons:** refund rate higher than usual; chargeback rate higher than usual; delivery/fulfilment evidence could not be confirmed for a notable share of disputed orders; support ticket volume and response times outside the usual pattern; multiple signals flagged together requiring manual review.
> **Suggested evidence:** refund/cancellation records; explanation for recent product/pricing/listing changes; chargeback/dispute reason breakdown; proof of delivery/service fulfilment; fulfilment/delivery proof for disputed orders; customer-support response records; plan to address recurring operational issues.
> **Appeal placeholder:** "If you believe this review signal does not reflect your business accurately, you may submit a written explanation and supporting evidence for a reviewer to consider. (Simulated appeal flow — not yet implemented in this milestone.)"

## 7. Known limitations

- Explanations are entirely template/rules-based — no natural-language generation model or LLM is involved, and none should be added without revisiting this design.
- All data is synthetic; no real merchant communication or payment action occurs.
- No case, evidence submission, or audit event is persisted anywhere — `audit_preview_events` are structured previews only, computed fresh every time a packet is built.
- Category one-hot factor sentences (`merchant_category`, `previous_review_outcome`) are simple translations, not deeply "explained" — they state which category was active and its direction of effect, not why that category matters causally.
- The top-factor list reflects the model's actual learned coefficients, including any counter-intuitive ones (see Section 2) — this is a faithfulness choice, not a bug, but it means the explanation can occasionally surprise a reader expecting naive intuition to hold.
