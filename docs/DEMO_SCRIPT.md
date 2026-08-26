# Demo Script — ClearRisk Recover

Status: Design proposal. Describes the intended demo walkthrough once Phase 1 application code exists. No code has been implemented yet for this scope — see `docs/RESCOPE_REVIEW.md` Section 8 for what's pending approval before implementation starts. Nothing below should be read as "already working."

Scope: Phase 1 only. Do not demo or mention Phase 2/3 roadmap items (real data import, authentication, gateway integration) during the walkthrough.

## 1. Open with the one-line pitch

"ClearRisk Recover detects early merchant refund and chargeback spikes, explains the risk factors, guides evidence collection, keeps a human reviewer in control, gives merchants a path to respond, and measures false positives."

State up front: this is a local, synthetic-data decision-support prototype. It is not a live gateway integration and cannot freeze funds, ban merchants, or move money.

## 2. Show the merchant-week risk feed

- Open the Streamlit dashboard's merchant-week feed page.
- Point out a merchant-week flagged `MANUAL_REVIEW_REQUIRED` and one flagged `APPROVE`, to show the system isn't flagging everything.

## 3. Open the flagged case's detail page

- Show the risk score/tier and the explanation, calling out that it states concrete before/after values (e.g. "chargeback rate increased from 0.4% to 2.2%"), not just a rule name.
- Show the refund/chargeback trend chart, volume trend, delivery-evidence coverage, and support-ticket trend that back up the explanation.
- Show the triggered rule(s) alongside the model's contribution — rules and model are shown as separate, transparent inputs to the recommendation, not a black box.

## 4. Walk through human review

- As the reviewer, open the case action panel.
- Show the required note field — no action can be submitted without one.
- Pick "Request evidence" and show the resulting case-status change and the new audit-timeline entry.

## 5. Walk through the merchant appeal

- Switch to the merchant view for the same case.
- Show the safe reason category and evidence checklist (no exact thresholds or raw model output visible).
- Submit a simulated appeal: free-text explanation plus fake evidence references (e.g. `invoice_demo_001.pdf`).
- Return to the reviewer view and show the case updated with the merchant's response, then resolve it (e.g. "Mark false positive") — pointing out this demonstrates the false-positive path, not just the confirmed-risk path.

## 6. Show the audit timeline

- Open the full audit timeline for the case and show every step recorded: scoring, case creation, reviewer action, appeal submission, resolution — each with a timestamp and actor type.

## 7. Show the held-out evaluation, and be honest about it

- Open the risk-manager metrics dashboard.
- Show precision, recall, PR-AUC, false-positive rate, and the confusion matrix on the held-out test period, with the train/validation/test date ranges displayed.
- Show the rules-only vs. rules+ML comparison — this is the evidence that the ML model adds value beyond the transparent rules baseline, not just complexity.
- Show at least one concrete false-positive example (a seasonal/legitimate high-return merchant that was flagged but resolved clean) and one concrete false-negative example (an early hidden-risk merchant that looked normal but later needed review).
- Read the on-screen banner aloud: *"Synthetic-data metrics demonstrate the prototype workflow only. They do not prove real-world payment fraud or chargeback performance."*
- If the numbers look unusually perfect, say so — a near-perfect result is a flag to investigate, not a headline claim.

## 8. Close

Reiterate: recommendation-only, human-reviewed, one bounded loss class, honest about what synthetic metrics can and cannot prove. No claim of gateway integration, real-world performance, production readiness, or regulatory compliance.
