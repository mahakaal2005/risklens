# ClearRisk Recover

### ClearRisk Recover — Explainable Refund & Chargeback Risk Review

**Detect early loss. Explain the risk. Review fairly.**

ClearRisk Recover helps payment-risk teams identify merchants whose refund and
chargeback patterns are likely to worsen in the next 30 days. It explains why the
merchant was flagged, recommends evidence to request, routes the case to a human
reviewer, supports a merchant response/appeal, and tracks false positives in a
complete audit trail.

**Local synthetic-data prototype only.** Not a real Razorpay integration, live
payment gateway, or payment processor. Does not process UPI/cards/wallets/net
banking payments, use real payment or personal data, or freeze funds, hold
settlement, ban merchants, terminate accounts, reject payments, or make
legal/compliance decisions. Not production-ready or compliance-certified.

## Razorpay Buildathon — Track 2: AI Risk Manager

> ClearRisk Recover detects early merchant refund and chargeback spikes, explains the risk factors, guides evidence collection, keeps a human reviewer in control, gives merchants a path to respond, and measures false positives.

**The one loss class** — not a broad multi-scenario fraud platform:

> For each merchant-week, predict whether that merchant will enter a simulated elevated refund/chargeback-loss state during the following 30 days (`label_high_loss_next_30d`).

### Why this is defensible against the judging bar

- **"Honest metrics including false-positive cost."** Real numbers from the current held-out test split (900 synthetic merchants, 93,600 merchant-weeks, threshold selected on validation data only — never the test set):

  | Method | Precision | Recall | PR-AUC | False-positive rate |
  |---|---|---|---|---|
  | Rules-only | 0.310 | 0.454 | 0.288 | 0.172 |
  | Logistic Regression (live-scoring model) | 0.561 | 0.828 | 0.664 | 0.110 |
  | Random Forest (comparison only) | 0.323 | 0.944 | 0.689 | 0.335 |
  | Gradient Boosting (comparison only) | 0.470 | 0.925 | 0.693 | 0.177 |
  | Combined policy | 0.403 | 0.870 | 0.664 | 0.219 |

  Logistic Regression is the model actually used for live case scoring — Random Forest and Gradient Boosting are evaluated purely as comparison baselines (never used to create a case), so the complexity-vs-accuracy tradeoff is shown honestly: Gradient Boosting edges out Logistic Regression on recall/PR-AUC at a higher false-positive cost; Random Forest over-flags substantially. None of the five methods is a near-perfect score — this project treats that as a bug signal to investigate, not a headline number (see `MODEL_CARD.md`'s near-perfect-score gate).
- **The label is not circular.** `label_high_loss_next_30d` is generated from a 5-state hidden latent-merchant simulation (Stable, Seasonal/legitimate high-return, Operational fulfilment failure, High-risk, Early hidden-risk), not as a threshold function of the same fields the rules engine checks.
- **"Strictly defense-only: anything offense-capable is disqualified."** The policy engine can only ever return a recommendation (`APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`) — never an automated freeze, hold, ban, termination, or payment rejection. Every high-impact action requires a human reviewer.
- **Fair to the merchant, not just accurate.** The flagged merchant sees a safe, plain-language explanation with concrete before/after trend values, an evidence checklist, and a genuine appeal path — not just a score.

## What it does

- Predicts, per merchant-week, whether a merchant will enter a simulated elevated
  refund/chargeback-loss state in the next 30 days.
- Explains why a merchant-week was flagged, with concrete before/after trend values.
- Routes elevated-risk merchant-weeks to a human review queue.
- Supports a simulated merchant appeal/evidence flow.
- Records an application audit history.
- Reports held-out-test model metrics (rules-only, Logistic Regression, Random Forest,
  Gradient Boosting, combined policy) and concrete false-positive/false-negative examples.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate synthetic data, train the models, and evaluate them
python3 -m ml.generate_synthetic_data     # writes demo_data/synthetic_merchant_week_data.csv
python3 -m ml.train_baseline_model        # Logistic Regression -- the model used for live case scoring
python3 -m ml.train_tree_models           # Random Forest + Gradient Boosting -- comparison baselines only
python3 -m ml.evaluate_model              # writes ml/artifacts/latest_evaluation_report.json

# 2. Generate the 5 demo case packets and seed the local database
python3 -m ml.generate_demo_cases
rm -f clearrisk_recover.db
python3 scripts/seed_demo_cases.py
python3 scripts/seed_demo_users.py        # prints local-demo login credentials once

# 3. Run the test suite
python3 -m pytest tests/ -v               # 483 tests
```

### Running the API

```bash
uvicorn app.main:app --reload
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). **There is no real payment gateway, and no endpoint capable of a real financial or enforcement action anywhere in this API** (verified by `tests/test_api_cases.py::test_no_route_or_response_contains_prohibited_enforcement_words`, which scans the live OpenAPI schema for `freeze`/`ban`/`terminate`/`hold settlement`/etc.).

`GET /metrics` reads the persisted `ml/artifacts/latest_evaluation_report.json` — it never retrains or re-scores on request. If that file doesn't exist yet, or fails validation, the endpoint still returns **HTTP 200** with `status: "not_available"` and the exact command to generate it, never a 500.

Every endpoint except `GET /health`, `POST /auth/login`, and `POST /auth/logout` requires an `Authorization: Bearer <session_token>` header (local-demo authentication, printed by `seed_demo_users.py`). Three roles: `reviewer` (all review actions), `merchant` (evidence submission, scoped to their own `merchant_id`), `risk_manager` (read-only metrics/cases).

| Endpoint | Role | Purpose |
|---|---|---|
| `GET /health` | none | Service status |
| `POST /auth/login`, `POST /auth/logout` | none | Local-demo session token |
| `GET /auth/me` | any | Current session identity |
| `GET /cases`, `GET /cases/{id}` | any | List/read review cases (merchant-scoped for the `merchant` role) |
| `GET /cases/{id}/audit-events` | any | Ordered audit timeline |
| `POST /cases/{id}/review-actions` | `reviewer` | Clear/escalate/request-evidence/mark-outcome |
| `POST /cases/{id}/evidence` | `merchant` | Simulated appeal response |
| `POST/GET /cases/{id}/evidence/{evidence_id}/attachments` | `merchant`/any | Real file upload/download (pdf/txt/png/jpg, 5MB max, magic-byte validated) |
| `GET /metrics` | any | Persisted held-out evaluation report |

Example:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" -d '{"username":"reviewer_demo","password":"..."}'
curl http://127.0.0.1:8000/cases -H "Authorization: Bearer <session_token>"
```

### Running the dashboard

Start the API first (above), then in a separate terminal:

```bash
streamlit run dashboard/streamlit_app.py
```

(If that fails with `ModuleNotFoundError: No module named 'google.protobuf'`, your `streamlit` executable's shebang points at a different Python than the one Streamlit was installed into — use `python3 -m streamlit run dashboard/streamlit_app.py` instead.)

Local dashboard URL: [http://localhost:8501](http://localhost:8501). Sign in with one of the three local-demo accounts printed by `seed_demo_users.py`; the sidebar then shows only the pages your role can use. Five pages: Overview, Review Queue, Case Detail, Merchant Response, Audit Timeline. The sidebar carries the synthetic-data disclaimer, a quiet backend-status line, and the active case ID. If the API isn't running, the dashboard shows a clear "Backend unavailable" message with the exact start command — it never falls back to mock data.

**Demo walkthrough:** sign in as `reviewer_demo` → **Review Queue** → click a case row → **Open case detail** → note the "Why flagged" tab's concrete before/after values (e.g. "Refund rate increased from 1.63% to 6.45%") → **Request evidence** → sign out, sign in as `merchant_demo` → **Merchant Response** (same case) → submit a simulated response → sign out, sign in as `reviewer_demo` → **Case Detail** → **Start review** → **Mark false positive** (or escalate/mark operational issue/mark inconclusive) → **Audit Timeline** to see every step recorded in order → **Overview** to show the held-out metrics table above, live, plus the "Difficulty by scenario" breakdown (which latent-state scenarios are genuinely hard to catch, disclosed rather than averaged away).

**Synthetic-data boundaries:** the dashboard, like the API underneath it, never processes a real payment, holds settlement, freezes funds, bans a merchant, terminates an account, or makes a final fraud determination — every action is a recommendation or a reviewer/merchant workflow step, on synthetic data only.

### Razorpay-shaped payment-event adapter (local, synthetic fixtures only)

```bash
python3 -m ml.razorpay_adapter    # reads demo_data/razorpay_fixtures/*.json, writes ml/artifacts/razorpay_adapter_report.json
```

Reads local, synthetic JSON fixture files whose event names are modeled on Razorpay's publicly documented webhook events, normalizes them, aggregates them into merchant-week rows, and produces a mapping/data-quality report. **This does not call any Razorpay API and is not a Razorpay integration** — every output is labeled `razorpay_shaped_demo_fixture_not_live_razorpay_data`.

## Stack

Python, FastAPI, Streamlit, SQLite, Pandas, scikit-learn, PyYAML, Joblib.

## Documentation

- `PRD.md` — product requirements
- `ARCHITECTURE.md` — component and data-flow design
- `DATA_DICTIONARY.md` — synthetic-data fields, the latent-state label design, and the v0.2.0 noise-injection details
- `RISK_POLICY.md` — illustrative risk rules and decisions
- `MODEL_CARD.md` — model scope, held-out evaluation, comparison baselines, scenario-difficulty breakdown, and limitations
- `SECURITY.md` — MVP security boundaries
- `RESEARCH.md` — evidence, assumptions, and open questions

## Status

All 8 Phase 1 milestones (synthetic data generation, rules engine + feature
engineering, time-based split + Logistic Regression baseline + held-out
evaluation, deterministic explainability, SQLite-persisted case workflow with
audit log, FastAPI read/workflow layer, persisted `/metrics` evaluation
report, Streamlit dashboard) plus Phase 2 (local-demo authentication with 3
roles, real evidence-attachment upload/download, computed review SLA,
manually-triggered feedback retraining, anonymized merchant-week CSV import)
are implemented, running, and covered by 483 passing tests. Random Forest and
Gradient Boosting comparison baselines, a scaled-up (900 merchants x 104
weeks) and noise-injected synthetic dataset, and a per-scenario difficulty
breakdown were added most recently — see `MODEL_CARD.md`.

Not production-grade auth (no MFA/password reset/rate limiting/external
identity provider), no malware scanning on uploads, no real email/SMS/webhook
notifications, no automatic retraining, no real gateway integration — see
`SECURITY.md` for the full boundary list.
