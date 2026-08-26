# Product Requirements Document
# ClearRisk Recover — Explainable Merchant Refund/Chargeback Loss-Risk Review MVP

Status: Design proposal, rescoped 2026-08-22 from an earlier four-scenario "AI Risk Manager" concept. See `docs/RESCOPE_REVIEW.md` for the full rationale.

## 1. Product summary

ClearRisk Recover is a local decision-support prototype for merchant refund/chargeback loss-risk review. It predicts, explains, and routes for human review — it does not decide or enforce.

The product does **not** process payments, connect to Razorpay or any gateway, or take real enforcement action.

## 2. Problem statement

Payment gateways and merchant-risk teams must monitor merchants for worsening refund and chargeback patterns before losses compound. This creates two risks:

1. **Loss risk:** weak or late monitoring lets a merchant's refund/chargeback losses grow before anyone reviews the account.
2. **False-positive harm:** legitimate merchants (e.g. running a seasonal sale with a temporarily higher, but valid, return rate) are incorrectly flagged and disrupted.

The MVP focuses on catching the first problem without ignoring the second: it adds explainability, a human-review step, and honest false-positive/false-negative measurement to a single, bounded loss class — merchant refund/chargeback loss risk.

## 3. Product vision

> Help payment-risk teams catch worsening merchant refund/chargeback patterns early, explain why a merchant was flagged, keep a human reviewer in control, and give the merchant a fair path to respond.

## 4. Target users

| User | Goal | Primary product need |
|---|---|---|
| Payment-risk analyst (primary) | Resolve merchant-week alerts quickly and correctly | Risk score, trend evidence, reasons, evidence checklist |
| Merchant support/compliance reviewer | Communicate a safe explanation and gather evidence | Safe reason category, evidence request text |
| Merchant | Understand a review and respond productively | Safe reason category, appeal mechanism |
| Risk manager | Improve the system over time | False-positive/negative rate, precision/recall, rules-only vs. rules+ML comparison |

## 5. Jobs to be done

### Payment-risk analyst

- When a merchant-week is flagged, I want to see the trend reasons (with concrete before/after numbers) and supporting evidence so I can resolve the case correctly.

### Merchant

- When my account is under review, I want to understand the high-level reason and have a way to respond with evidence.

### Risk manager

- When the risk engine generates alerts, I want to measure confirmed risk versus false positives and false negatives, and compare the rules-only baseline against rules+ML.

## 6. MVP scope

### In scope

- Local synthetic merchant-week dataset generation, built from a 5-state latent merchant simulation (not derived directly from the rules engine's thresholds).
- One merchant-week risk score and risk tier (low/medium/high).
- One bounded risk scenario: merchant refund/chargeback loss risk, 30-day forward horizon.
- Rules engine plus Logistic Regression baseline, reported separately and combined.
- Explainable reasons with concrete before/after trend values, and triggered rules.
- Recommendation-only workflow.
- Review queue, reviewer notes, and resolution state.
- Simulated merchant appeal/evidence submission (text + fake evidence filenames).
- Append-only application audit timeline.
- Time-based held-out evaluation and key metrics, including concrete false-positive and false-negative examples.
- Streamlit dashboard and FastAPI API.

### Out of scope

- Live payment processing, settlement, account restrictions, or account termination.
- Real gateway APIs/data, live webhook integration, and real user authentication.
- Real KYC/AML checks, sanctions checks, or regulator reporting.
- Production deployment/security certification.
- Full fraud-graph or deep-learning system.
- Per-transaction fraud scenarios (unusual amount, new device + disputes, account takeover) and merchant profile mismatch as a standalone scenario — moved to Future work only, not part of Phase 1.
- Gateway connector of any kind.

## 7. Functional requirements

### FR-1: Synthetic data

The system shall generate synthetic, non-identifying **merchant-week** records with the fields defined in `DATA_DICTIONARY.md`.

Acceptance criteria:

- Data includes `merchant_id`, `week_start`, transaction/refund/chargeback aggregates, delivery-evidence coverage, support metrics, previous review outcome, and `label_high_loss_next_30d`.
- Data dictionary explains each field, clearly identifies it as synthetic, and documents the exact latent-state label definition.
- No realistic payment credentials or direct personal identifiers exist in the data.
- The label is generated from a hidden/latent merchant-state simulation, not as a direct deterministic function of the rules engine's own thresholds.

### FR-2: Risk scenario

The system shall detect one scenario family: merchant refund/chargeback loss risk, predicting `label_high_loss_next_30d` per merchant-week.

Acceptance criteria:

- The scenario has a documented rule definition (`RISK_POLICY.md`) and a documented ML target definition (`MODEL_CARD.md`).
- The scenario can be demonstrated with known synthetic examples covering all four confusion-matrix quadrants: true positive, false positive (seasonal/legitimate high-return), false negative (early hidden-risk), true negative.
- The system can show which rule(s) and which trend values contributed to a flag.

### FR-3: Rules engine

The system shall evaluate transparent business rules over merchant-week aggregates, independently from the ML model.

Acceptance criteria:

- Rules are configurable from a versioned YAML file.
- Rule results include rule ID, severity, reason category, and safe explanation text with concrete trend values.
- Rule evaluation is unit-tested.
- No rule references a single transaction, device token, or customer dispute count (those fields do not exist in the merchant-week entity).

### FR-4: ML risk scoring

The system shall produce a merchant-week `label_high_loss_next_30d` probability using an interpretable model.

Acceptance criteria:

- Rules-only baseline is implemented and reported first.
- Logistic Regression is implemented second, and reported both alone and combined with rules.
- Train/validation/test split is time-based (earliest/middle/latest weeks).
- The latest period is held out and untouched during model choice/threshold tuning.
- Model artifact and feature list are versioned.
- If the model is unavailable, the system falls back to rules-only mode.
- A near-perfect held-out result (PR-AUC >= 0.98, precision >= 0.98 and recall >= 0.98, or zero false positives/negatives on a meaningful test set) is marked "Under investigation" per the checklist in `MODEL_CARD.md`, and is not used in the submission pitch until that checklist passes.

### FR-5: Decision policy

The system shall combine rules and model score into a recommendation.

Allowed recommendations:

- `APPROVE`
- `ALLOW_WITH_MONITORING`
- `REQUEST_EVIDENCE`
- `MANUAL_REVIEW_REQUIRED`
- `ESCALATE_TO_COMPLIANCE`

Acceptance criteria:

- No automatic freeze, hold, ban, termination, payment rejection, or money movement exists.
- Medium/high risk creates a review case.
- Decision policy version is recorded in the audit log.

### FR-6: Explainability

The system shall show a safe, plain-language explanation for each medium/high-risk merchant-week, stating concrete before/after values for the trend fields that drove the flag.

Example:

```text
Recommended action: Manual review required

Reasons:
- Chargeback rate increased from 0.4% to 2.2%.
- Refund rate increased from 1.1% to 5.8%.
- Delivery-evidence coverage decreased.
- Support resolution time increased.

Suggested reviewer step:
Request delivery/fulfilment evidence and review recent dispute reasons.
```

Acceptance criteria:

- Explanation contains triggered rules, concrete trend values, and top model factors.
- Explanation does not expose raw model coefficients or exact evasion-sensitive thresholds to the merchant view.
- Explanation is stored with the decision record.

### FR-7: Analyst review

The system shall support a review queue and case-resolution workflow.

Reviewer actions:

- Clear case
- Mark false positive
- Request evidence
- Mark operational issue
- Escalate to compliance
- Mark inconclusive

Acceptance criteria:

- Reviewer decision requires notes.
- Reviewer action writes an audit event.
- A false-positive resolution becomes a feedback label for future analysis.
- Reviewer view shows: risk score/tier, main reasons, rule triggers, refund/chargeback trend, volume trend, delivery-evidence coverage, support-ticket trend, previous review outcomes, evidence checklist, and audit timeline.

### FR-8: Merchant appeal simulation

The system shall allow a merchant to submit simulated evidence and an appeal note for an open case.

Acceptance criteria:

- Merchant can see a safe reason category and evidence checklist.
- Merchant cannot view secret fraud rules/thresholds.
- Merchant submits free-text explanation plus simulated evidence references (fake filenames/URLs) — no real file upload.
- Appeal submission changes case state and creates an audit event.

### FR-9: Audit log

The system shall maintain an append-only application audit trail.

Audit events include:

- Event creation
- Feature/rule evaluation summary
- Model scoring version
- Explanation generated
- Policy recommendation
- Case creation
- Reviewer actions
- Evidence submission
- Case resolution

Acceptance criteria:

- Every audit record includes timestamp, entity/case ID, actor type, event type, and structured payload.
- Update/delete UI actions are not provided for audit rows.
- Documentation states that this is an application audit log, not a cryptographically immutable ledger.

### FR-10: Evaluation dashboard

The system shall report model and process performance on the held-out test set.

Required metrics:

- Precision
- Recall
- PR-AUC
- False-positive rate
- Confusion matrix
- Precision/recall at the selected operating threshold
- Rules-only vs. rules+ML comparison
- Counts of confirmed risk, false positives, false negatives, inconclusive cases
- Metrics by latent merchant state and merchant category where sample size allows
- At least three concrete false-positive examples and at least two concrete false-negative examples

Acceptance criteria:

- Metrics are calculated on the held-out test set only.
- Training/validation/test time periods are displayed.
- Dashboard always displays: *"Synthetic-data metrics demonstrate the prototype workflow only. They do not prove real-world payment fraud or chargeback performance."*
- A near-perfect result triggers a documented investigation note in the dashboard, not a headline claim.

## 8. Non-functional requirements

| Category | Requirement |
|---|---|
| Simplicity | Must run locally with clear setup instructions |
| Safety | Must not handle real payment credentials or move money |
| Explainability | Every elevated-risk case must have an understandable reason with concrete trend values |
| Traceability | Important decisions must create an audit event |
| Reliability | Invalid input receives a clear validation error; model failures fall back to rules-only mode |
| Reproducibility | Dataset generation, training, and evaluation are scripted and seedable |
| Testability | Core rules, policy, audit logging, and API validation have tests |
| Honesty | Near-perfect synthetic metrics are investigated, not reported as proof of real-world performance |
| Documentation | README, model card, data dictionary, security boundaries, architecture must be included |

## 9. Primary user flows

**Flow A: Merchant-week review**

```text
Synthetic merchant-week arrives
→ rules evaluate it
→ model scores it
→ policy recommends action
→ medium/high risk creates case
→ analyst reviews reasons and trends
→ analyst resolves, requests evidence, or escalates
→ audit log records all steps
```

**Flow B: Merchant appeal**

```text
Open risk case
→ merchant sees safe reason category and evidence request
→ merchant submits appeal/evidence note
→ case returns to analyst queue
→ analyst records final result
→ audit timeline updates
```

**Flow C: Risk-manager evaluation**

```text
Load held-out metrics
→ inspect false positives and false negatives
→ compare rules-only vs. rules-plus-model
→ evaluate threshold effects
→ document recommended policy changes
```

## 10. Success metrics for MVP

Product success is measured by demonstration quality and evaluation honesty, not real-world revenue or fraud reduction.

- 100% of medium/high-risk demo cases display a reason (with concrete trend values) and recommendation.
- 100% of high-risk demo cases require a human-review workflow.
- 100% of reviewer and merchant-appeal actions appear in audit history.
- Model metrics are reproducibly generated on a held-out synthetic test period.
- At least three concrete false-positive examples and two concrete false-negative examples are demonstrable in the dashboard.
- Rules-only fallback works when the model toggle is disabled.
- A near-perfect synthetic result is flagged and investigated, not reported as a headline claim.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data creates unrealistic metrics | State limitations prominently; avoid production-performance claims; treat near-perfect results as a bug signal |
| Label leaks directly from rule thresholds (circular evaluation) | Generate the label from a hidden latent-state simulation, never as a direct function of rule fields |
| Scope creep back toward multiple loss classes | Limit to one flagship scenario in Phase 1; other scenarios documented only under Future work |
| ML adds little beyond rules | Compare against rules-only baseline and keep ML optional |
| Confusing recommendation with enforcement | Use recommendation labels; prohibit automatic freeze/hold/ban/terminate/reject actions |
| Weak data labels | Clearly document the latent-state simulation and label horizon assumptions |

## 12. MVP definition of done

The MVP is ready for demonstration when the full merchant-week-to-case-to-appeal-to-audit flow works locally with synthetic data, the held-out evaluation (including false-positive and false-negative examples) is reproducible, and all required documentation and limitations are present.
