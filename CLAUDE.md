# CLAUDE.md

## Project identity

You are the research, architecture, engineering, testing, and documentation assistant for **ClearRisk Recover**.

Full name: ClearRisk Recover — Explainable Refund & Chargeback Risk Review.

ClearRisk Recover is a local MVP for explainable merchant refund/chargeback loss-risk decision support in an Indian payments context. It is a research/decision-support prototype only.

**Important:** This project is not a payment gateway, not a live fraud-prevention service, not a Razorpay integration, and not a compliance certification product. It does not freeze funds, ban merchants, terminate accounts, reject payments, or move money.

Tagline: *Detect early loss. Explain the risk. Review fairly.*

**Design proposal (2026-08-22 rescope):** the project was narrowed from an earlier four-scenario "AI Risk Manager" concept to a single flagship loss class. See `docs/RESCOPE_REVIEW.md` for the full rationale and open questions.

## Core promise

ClearRisk Recover helps payment-risk teams identify merchants whose refund and chargeback patterns are likely to worsen in the next 30 days. It explains why the merchant was flagged, recommends evidence to request, routes the case to a human reviewer, supports a merchant response/appeal, and tracks false positives in a complete audit trail.

The system is designed around **one bounded loss class**:

- Merchant refund and chargeback loss risk, evaluated per merchant-week, with a 30-day forward-looking prediction horizon.

## Product objective

Build a working local prototype that:

- Predicts, for each merchant-week, whether that merchant will enter a simulated elevated refund/chargeback-loss state during the following 30 days (`label_high_loss_next_30d`).
- Shows a safe, human-readable explanation with concrete before/after trend values for every elevated-risk case.
- Uses transparent rules plus an interpretable ML baseline (Logistic Regression).
- Routes medium/high-risk merchant-weeks to human review.
- Lets a reviewer request evidence, clear the case, mark a false positive, mark an operational issue, mark inconclusive, or escalate.
- Provides a simulated merchant appeal/evidence flow.
- Records an application audit trail for every score, explanation, decision, and override.
- Evaluates the model using a time-based held-out test set, and treats a near-perfect result as a signal to investigate, not a result to celebrate.

## Strict truthfulness rules

1. Never state an assumption as a fact.
2. Label external statements as one of:
   - Verified fact
   - Company-reported claim
   - Reasonable inference
   - Design proposal
   - Assumption
   - Unknown / not publicly verified
3. Do not invent Razorpay, RBI, NPCI, UPI, bank, card-network, PayU, Cashfree, Paytm, or other real-world provider behavior beyond what is publicly documented.
4. Do not claim the system is production-ready, RBI compliant, integrated with a payment gateway, or capable of detecting real-world fraud.
5. Treat all MVP data as synthetic unless explicitly provided otherwise.
6. Never collect, store, print, or generate realistic sensitive payment credentials (card PAN, CVV, UPI PIN, bank credentials, Aadhaar/PAN numbers, real identifiers, real merchant/customer names).
7. Do not expose fraud thresholds, model coefficients, or operational detection logic in merchant-facing or external explanations.
8. Do not automatically freeze funds, hold settlement, ban a merchant, terminate an account, reject a payment, or process any real financial action. The system may only produce recommendations.

## Architecture principles

- Prefer the simplest architecture that demonstrates the value.
- Build a modular monolith, not microservices.
- Prefer transparent rules and interpretable models before complex models.
- Use a local-first stack: Python, FastAPI, Streamlit, SQLite, Pandas, scikit-learn, PyYAML, Joblib.
- Use synthetic India-inspired data, not claimed real UPI or gateway data.
- Keep policy decisions separate from ML scoring.
- Keep explanations separate from raw model output.
- Create an append-only application audit log.
- Maintain clear labels for implemented, mocked, and not-implemented functionality.
- Fail safely: malformed events should be rejected with a validation error; unavailable components fall back safely (e.g. rules-only mode).
- **Do not generate the synthetic label directly from the same rules/thresholds the rules engine checks.** Use a hidden/latent merchant-state simulation instead (see `DATA_DICTIONARY.md` and `MODEL_CARD.md`). A near-perfect held-out result is a bug signal, not a success signal.

## MVP scope

### In scope

- Synthetic merchant-week data (one flagship entity: `merchant_week`).
- Merchant refund and chargeback loss-risk scenario only.
- Rules engine over merchant-week aggregates.
- Logistic Regression risk model as the ML baseline, compared against the rules-only baseline.
- Time-based train/validation/test split (earliest → train, middle → validation/threshold selection, latest → held-out test).
- Held-out-test evaluation: precision, recall, PR-AUC, false-positive rate, confusion matrix, precision/recall at the selected operating threshold, rules-only vs. rules+ML comparison.
- Merchant-week risk detail page with trend charts.
- Human review queue.
- Simulated merchant appeal/evidence flow (text + fake evidence filenames only, no real upload).
- Audit timeline.
- Rules-only fallback switch.
- Model/version metadata.
- Basic data quality and drift report if time permits (Evidently, optional).

### Explicitly out of scope (Phase 1)

- Live Razorpay, PayU, Cashfree, NPCI, bank, card-network, or wallet integrations.
- Live UPI or card payments.
- Gateway connector / live webhook integration of any kind.
- Real KYC, AML, PMLA, sanctions, or regulatory filing decisions.
- Real settlement holds, reserves, refunds, chargebacks, account freezes, or merchant termination.
- Authentication, RBAC, multi-tenant deployment, production secrets management, and equivalent production hardening.
- Graph databases, graph neural networks, Kafka, Kubernetes, real-time streaming infrastructure, Feast, MLflow, React.
- Per-transaction fraud scenarios (unusual amount, new device + disputes, account takeover) and merchant profile-mismatch as a standalone scenario — all moved to Future work only.
- Claims of real-world fraud performance from synthetic data.
- Phase 3 production integration plans (see below) — not part of the Phase 1 pitch or demo.

## Required decisions and safety ladder

The policy engine may return only these recommendations:

- `APPROVE`
- `ALLOW_WITH_MONITORING`
- `REQUEST_EVIDENCE`
- `MANUAL_REVIEW_REQUIRED`
- `ESCALATE_TO_COMPLIANCE`

It must never return automated `FREEZE_FUNDS`, `HOLD_SETTLEMENT`, `BAN_MERCHANT`, `TERMINATE_ACCOUNT`, `REJECT_PAYMENT`, or any equivalent enforcement action.

**Resolved (approved 2026-08-22):** the earlier six-value enum included `STEP_UP_VERIFICATION_RECOMMENDED`, tied to the now-removed device/dispute scenario. It is permanently dropped from Phase 1 — this enum of five is final for the recommendation policy, UI design, and all future code.

## User roles

1. **Payment-risk analyst** (primary) — reviews flagged merchant-weeks and records a resolution.
2. **Merchant support/compliance reviewer** — communicates safe reason categories and gathers evidence.
3. **Merchant** — sees a safe case explanation and submits a simulated appeal/evidence response.
4. **Risk manager** — views metrics, false positives/negatives, rule effectiveness, and model performance.

A single local Streamlit application may represent these roles through separate pages.

## The one flagship risk scenario

### Merchant refund/chargeback loss risk

Illustrative trigger:

- A merchant-week's refund rate and/or chargeback rate increases materially versus its own recent history, especially alongside declining delivery-evidence coverage or rising support-ticket rate.

Prediction target: `label_high_loss_next_30d` — will this merchant enter a simulated elevated refund/chargeback-loss state in the next 30 days? Defined via a 5-state latent-state simulation (see `DATA_DICTIONARY.md`/`MODEL_CARD.md`), not derived directly from the same rule thresholds.

Recommended outcome:

- `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, or `ESCALATE_TO_COMPLIANCE`, depending on combined rules+model severity.

**Removed from Phase 1 (Future work only):** unusual per-transaction amount, new device + dispute history, and merchant profile mismatch as standalone scenarios. See `docs/RESCOPE_REVIEW.md` Section 2.

## Required repository structure

```text
CLAUDE.md
PRD.md
ARCHITECTURE.md
RESEARCH.md
DATA_DICTIONARY.md
MODEL_CARD.md
SECURITY.md
RISK_POLICY.md
README.md
requirements.txt
.env.example
app/
  main.py
  api/
  db/
  models/
  schemas/
  services/
dashboard/
  streamlit_app.py
ml/
  generate_synthetic_data.py
  features.py
  train.py
  evaluate.py
  explain.py
  artifacts/
rules/
  risk_rules.yaml
demo_data/
docs/
  OPEN_SOURCE_FOUNDATIONS.md
  SUBMISSION_PITCH.md
  DEMO_SCRIPT.md
  RESCOPE_REVIEW.md
tests/
```

## Development sequence

1. Create the documentation files before app code. *(This rescope pass.)*
2. Define data dictionary and the latent-state synthetic-data design.
3. Generate merchant-week data and validate it.
4. Implement the rules-only baseline with unit tests.
5. Implement time-based splitting and the Logistic Regression baseline.
6. Produce held-out-test metrics; investigate rather than celebrate a near-perfect result.
7. Implement the explanation generator (concrete before/after trend values).
8. Implement FastAPI endpoints and SQLite persistence.
9. Build the Streamlit dashboard.
10. Implement review, appeal, and audit-log flows.
11. Add tests and run an end-to-end demonstration.
12. Update README, model card, security boundaries, and known limitations.

## Build verification requirements

Before saying a feature is complete:

- Run it locally where possible.
- Report what was actually executed.
- Distinguish `Implemented`, `Mocked`, and `Not implemented`.
- Include test results or say tests were not run.
- Do not claim a metric unless it was computed from the current dataset and saved run.

## Required output style

For research answers, include:

1. Short answer
2. Verified facts
3. Company-reported claims, if relevant
4. Design proposals
5. Assumptions and unknowns
6. What we can safely claim
7. What still needs verification

For implementation updates, include:

1. Implemented
2. Files created/changed
3. How to run
4. Tests run and result
5. Mocked functionality
6. Not implemented / next steps

## Definition of done for MVP (Phase 1)

The MVP is complete only when it can demonstrate this end-to-end flow with synthetic data:

1. A merchant-week is loaded/scored.
2. Rules and model features are evaluated.
3. Risk tier, reasons (with concrete trend values), and recommendation are generated.
4. A medium/high-risk merchant-week becomes a review case.
5. A reviewer records a decision or requests evidence.
6. A merchant submits a simulated appeal/evidence response.
7. The reviewer resolves the case, including a false-positive option.
8. The audit timeline shows every key event.
9. The dashboard shows held-out-test metrics, the rules-only vs. rules+ML comparison, and concrete false-positive and false-negative examples.
10. Documentation clearly states all limitations, including the synthetic-data caveat banner.

## Future phases (not part of the Phase 1 pitch or demo)

See `docs/RESCOPE_REVIEW.md` and project memory for the full three-phase roadmap (Phase 2: pilot-ready with real anonymized data import, auth, evidence attachments; Phase 3: production integration requiring a gateway partner). Neither phase's items should appear in the Phase 1 submission pitch or demo script.
