# Demo Script — ClearRisk Recover

Status: **Implemented and verified.** All steps below describe the actual running application — walked through live against the real FastAPI backend and Streamlit dashboard, not a plan. Sign-in is required for every dashboard page (Phase 2 authentication); the steps below include it as a practical necessity, not as a feature being showcased.

Scope: Phase 1 only. Do not demo or discuss Phase 2/3 roadmap items (evidence attachments, review SLA, feedback retraining, external data import, real data import, gateway integration) as features during the walkthrough, even though they now exist in the codebase.

## 0. Start the app

```bash
rm -f clearrisk_recover.db
python3 scripts/seed_demo_cases.py
python3 scripts/seed_demo_users.py    # prints demo login credentials once -- copy reviewer_demo and merchant_demo
uvicorn app.main:app --reload          # terminal 1
streamlit run dashboard/streamlit_app.py  # terminal 2
```

Open `http://localhost:8501` and sign in as `reviewer_demo`.

## 1. Open with the one-line pitch

"ClearRisk Recover detects early merchant refund and chargeback spikes, explains the risk factors, guides evidence collection, keeps a human reviewer in control, gives merchants a path to respond, and measures false positives."

State up front: this is a local, synthetic-data decision-support prototype. It is not a live gateway integration and cannot freeze funds, ban merchants, or move money.

## 2. Show the Review Queue

- Open the **Review Queue** page.
- Point out a case flagged `MANUAL_REVIEW_REQUIRED` or `REQUEST_EVIDENCE` alongside the summary metrics showing not every case is escalated — the system isn't flagging everything.
- Click a row to select a case, then **Open case detail**.

## 3. Open the flagged case's detail page

- Show the header (merchant, week, status, risk-signal badge) and the recommended workflow action.
- Open the **Why flagged** tab: show the analyst summary, then the triggered-rule list — each triggered rule shows a **concrete before/after sentence** (e.g. "Refund rate increased from 1.63% to 6.45% (4.82% change)."), not just a rule name. This is the "explains the risk with concrete before/after values" claim, live.
- Note explicitly what's *not* shown yet: a ranked list of every model feature's contribution and trend values for features that didn't trigger a rule — the **Analyst detail** tab states this gap plainly rather than fabricating a number.

## 4. Walk through human review

- Still as `reviewer_demo`, use the **Reviewer actions** section (directly below the case header, not buried in a tab).
- Show the required note field — no action can be submitted without one.
- Pick "Request evidence" and show the resulting case-status change.

## 5. Walk through the merchant response

- Sign out, sign back in as `merchant_demo`.
- Open **Merchant Response**, select the same case (now `EVIDENCE_REQUESTED`).
- Show the safe reason category and evidence checklist (no exact thresholds or raw model output visible).
- Submit a simulated response: free-text explanation plus fake evidence references (e.g. `invoice_demo_001.pdf`).
- Sign out, sign back in as `reviewer_demo`. Show the case updated with the merchant's response (`EVIDENCE_SUBMITTED`), start review, then resolve it (e.g. "Mark false positive") — pointing out this demonstrates the false-positive path, not just the confirmed-risk path.

## 6. Show the Audit Timeline

- Open **Audit Timeline** for the same case and show every step recorded in order: scoring, case creation, reviewer action, merchant response, resolution — each with a timestamp and actor type. Click a row to see that event's safe payload detail.

## 7. Show the held-out evaluation, and be honest about it

- Open the **Overview** page.
- Show the rules-only / Logistic Regression / combined-policy comparison table: precision, recall, PR-AUC, and false-positive rate on the held-out test period.
- Point out the combined policy trades precision for recall (the early-warning default), while Logistic Regression alone has the lowest false-positive rate — this is the "honest metrics including false-positive cost" the judging bar asks for.
- Read the on-screen limitation banner aloud: *"Synthetic-data metrics demonstrate the prototype workflow only. They do not prove real-world payment fraud or chargeback performance."*
- If the numbers look unusually perfect, say so — a near-perfect result is a flag to investigate, not a headline claim. (They aren't: recall ~0.86-0.88, precision 0.37-0.53 depending on method — a real, imperfect tradeoff.)

## 8. Close

Reiterate: recommendation-only, human-reviewed, one bounded loss class, honest about what synthetic metrics can and cannot prove. No claim of gateway integration, real-world performance, production readiness, or regulatory compliance.
