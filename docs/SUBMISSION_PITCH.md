# Submission Pitch — ClearRisk Recover

**ClearRisk Recover — Explainable Refund & Chargeback Risk Review**
*Detect early loss. Explain the risk. Review fairly.*

Status: **Implemented and verified** — all 8 Phase 1 milestones are built, running, and covered by an automated test suite (474 tests passing at last run). This pitch covers Phase 1 only. Phase 2 and Phase 3 roadmap items are intentionally excluded from this pitch and the demo script — see `docs/RESCOPE_REVIEW.md` and the project's phase roadmap for what comes later — even though Phase 2 has since also been built; that is deliberately out of scope for this specific submission pitch.

## Track

Razorpay Buildathon — Track 2: AI Risk Manager ("Stop the merchant losing money to fraud, returns and chargebacks").

## Pitch

> ClearRisk Recover detects early merchant refund and chargeback spikes, explains the risk factors, guides evidence collection, keeps a human reviewer in control, gives merchants a path to respond, and measures false positives.

## The one loss class

Merchant refund and chargeback loss risk — not a broad multi-scenario fraud platform. One flagship prediction task, evaluated honestly:

> For each merchant-week, predict whether that merchant will enter a simulated elevated refund/chargeback-loss state during the following 30 days.

## Why this is defensible against the judging bar

- **"Honest metrics including false-positive cost"** — the dashboard reports precision, recall, PR-AUC, false-positive rate, and confusion matrix on a genuinely held-out, time-based test period, alongside concrete false-positive and false-negative examples pulled straight from that test set (not cherry-picked from training data). Actual numbers from the current held-out test split (220 synthetic merchants, 11,440 merchant-weeks, threshold selected on validation data only):

  | Method | Precision | Recall | PR-AUC | False-positive rate |
  |---|---|---|---|---|
  | Rules-only | 0.269 | 0.425 | 0.246 | 0.170 |
  | Logistic Regression | 0.528 | 0.863 | 0.653 | 0.113 |
  | Combined policy | 0.375 | 0.880 | 0.653 | 0.216 |

  The combined policy trades precision for recall deliberately (it's the early-warning queue-routing default); Logistic Regression alone has the lowest false-positive rate. Neither is a near-perfect score — which this project treats as a bug signal to investigate, not a result to present, per its own model card.
- **The label is not circular.** `label_high_loss_next_30d` is generated from a 5-state hidden latent-merchant simulation (Stable, Seasonal/legitimate high-return, Operational fulfilment failure, High-risk, Early hidden-risk), not as a direct threshold function of the same fields the rules engine checks. A near-perfect score is explicitly treated as a bug signal to investigate, not a headline number.
- **"Strictly defense-only: anything offense-capable is disqualified."** The policy engine can only ever return a recommendation (`APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`) — never an automated freeze, hold, ban, termination, or payment rejection. Every high-impact action requires a human reviewer.
- **Fair to the merchant, not just accurate.** The flagged merchant sees a safe, plain-language explanation with concrete before/after trend values, an evidence checklist, and a genuine appeal path — not just a score.

## What's built for the demo (Phase 1 only)

All of the below is implemented, running, and tested — not a plan:

- Synthetic merchant-week data generator (5-state latent simulation) — 220 merchants, 52 weeks.
- Rules engine (refund/chargeback-trend rules) and a Logistic Regression baseline, reported separately and combined.
- Explanation layer with concrete trend values.
- Human review queue with required reviewer notes and full audit trail.
- Simulated merchant appeal (text + fake evidence references).
- A 5-page Streamlit dashboard (Overview, Review Queue, Case Detail, Merchant Response, Audit Timeline) reading live from a local FastAPI backend — held-out metrics with a rules-only vs. rules+ML comparison, a clickable case queue, per-case explanation and reviewer actions, the merchant-safe appeal flow, and the full ordered audit trail.

## What this explicitly is not

- Not a real Razorpay integration, live payment gateway, or payment processor.
- Not a system that can freeze funds, ban merchants, terminate accounts, reject payments, or move money.
- Not a claim of RBI compliance or production readiness.
- Not a broad multi-scenario fraud platform — per-transaction amount anomalies, device/dispute account-takeover signals, and merchant profile mismatch are explicitly Future work, not part of this pitch.

## One-line differentiator

The product is not merely a score — it is a fair, explainable, reviewable merchant-risk handling workflow, end to end, with honestly measured false positives.
