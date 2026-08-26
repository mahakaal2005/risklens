 # ClearRisk Recover

 ### ClearRisk Recover — Explainable Refund & Chargeback Risk Review

 **Detect early loss. Explain the risk. Review fairly.**

 ClearRisk Recover helps payment-risk teams identify merchants whose refund and
 chargeback patterns are likely to worsen in the next 30 days. It explains why the
 merchant was flagged, recommends evidence to request, routes the case to a human
 reviewer, supports a merchant response/appeal, and tracks false positives in a
 complete audit trail. Rescoped 2026-08-22 to this single flagship use case from an
 earlier four-scenario concept — see `docs/RESCOPE_REVIEW.md`.

 ## What it does

 - Predicts, per merchant-week, whether a merchant will enter a simulated elevated
   refund/chargeback-loss state in the next 30 days.
 - Explains why a merchant-week was flagged, with concrete before/after trend values.
 - Routes elevated-risk merchant-weeks to a human review queue.
 - Supports a simulated merchant appeal/evidence flow.
 - Records an application audit history.
 - Reports held-out-test model metrics, a rules-only vs. rules+ML comparison, and
   concrete false-positive and false-negative examples.

 ## What it does not do

 - Does not connect to Razorpay or any real payment gateway.
 - Does not process UPI/cards/wallets/net banking payments.
 - Does not use real payment data or real personal data.
 - Does not freeze funds, hold settlement, ban merchants, terminate accounts, reject
   payments, or make legal/compliance decisions.
 - Is not production-ready or compliance-certified.

 ## MVP scope: one flagship risk scenario

 Merchant refund and chargeback loss risk, evaluated per merchant-week, 30-day
 forward-looking horizon.

 Per-transaction fraud scenarios (unusual amount, new device + disputes, account
 takeover) and merchant profile mismatch from an earlier scope are not part of this
 MVP — see `docs/RESCOPE_REVIEW.md` for what was removed and why.

 ## How to run (implemented so far: data generation through the FastAPI workflow API)

 ```bash
 pip install -r requirements.txt
 python3 ml/generate_synthetic_data.py      # generates demo_data/synthetic_merchant_week_data.csv
 python3 -m ml.inspect_synthetic_data        # data-quality report -> docs/MILESTONE_1_DATA_QUALITY_REPORT.md
 python3 -m ml.train_baseline_model          # trains + saves the Logistic Regression baseline to ml/artifacts/
 python3 -m ml.evaluate_model                # rules-only vs. ML vs. combined-policy evaluation; also writes ml/artifacts/latest_evaluation_report.json
 python3 -m ml.generate_demo_cases           # builds 5 reviewer-ready case packets -> demo_data/demo_case_packets.json
 rm -f clearrisk_recover.db                  # optional: reset the local demo database
 python3 scripts/seed_demo_cases.py          # persists non-APPROVE demo packets as review cases in SQLite
 python3 scripts/demo_case_workflow.py       # walks two seeded cases through the full reviewer/evidence workflow
 python3 -m pytest tests/ -v                 # full test suite (289 tests as of Milestone 6)
 ```

 ### Running the API (local, synthetic-data demo only)

 ```bash
 uvicorn app.main:app --reload
 ```

 Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). See `docs/API_CONTRACT.md` for every endpoint, request/response example, and error code. **There is no authentication, no real payment gateway, and no endpoint capable of a real financial or enforcement action anywhere in this API** — see `docs/API_CONTRACT.md`'s explicit non-existent-endpoint list.

 `GET /metrics` reads `ml/artifacts/latest_evaluation_report.json` (written by `python3 -m ml.evaluate_model`, Milestone 7) — it never retrains or re-scores on request. If that file doesn't exist yet, or fails validation, the endpoint still returns **HTTP 200** with `status: "not_available"` and the exact command to generate it, never a 500. See `docs/MILESTONE_7_METRICS.md` for the full artifact lifecycle.

 Example:

 ```bash
 curl http://127.0.0.1:8000/health
 curl http://127.0.0.1:8000/cases
 curl http://127.0.0.1:8000/cases/<case_id>
 ```

 ### Running the dashboard (local, synthetic-data demo only)

 Start the API first (above), then in a separate terminal:

 ```bash
 streamlit run dashboard/streamlit_app.py
 ```

 (If that fails with `ModuleNotFoundError: No module named 'google.protobuf'`, your `streamlit` executable's shebang points at a different Python than the one Streamlit was installed into — use `python3 -m streamlit run dashboard/streamlit_app.py` instead.)

 Local dashboard URL: [http://localhost:8501](http://localhost:8501). Five pages: Overview, Review Queue, Case Detail, Merchant Response, Audit Timeline. The sidebar carries the synthetic-data disclaimer, a quiet backend-status line, and the active case ID; each page body opens with a one-line synthetic-data reminder. If the API isn't running, the dashboard shows a clear "Backend unavailable" message with the exact start command — it never falls back to mock data.

 Short demo flow: seed the demo cases (`python3 scripts/seed_demo_cases.py`) → open Review Queue → **click a case row** → **Open case detail** → **Request evidence** → Merchant Response (already on that case) → submit a simulated response → Case Detail → **Start review** → **Mark false positive** (or **Mark operational issue** / **Mark inconclusive** / **Escalate case**) → Audit Timeline to see every step recorded in order. Full walkthroughs (seasonal-sale, operational-issue, high-risk combined-loss) are in `docs/UI_DEMO_GUIDE.md`.

 **Synthetic-data boundaries:** the dashboard, like the API underneath it, never processes a real payment, holds settlement, freezes funds, bans a merchant, terminates an account, or makes a final fraud determination — every action is a recommendation or a reviewer/merchant workflow step, on synthetic data only.

 Authentication, real file uploads, and external integrations are not implemented — see `docs/IMPLEMENTATION_PLAN.md` for the remaining milestones.

 ### Razorpay-shaped payment-event adapter (local, synthetic fixtures only)

```bash
python3 -m ml.razorpay_adapter    # reads demo_data/razorpay_fixtures/*.json, writes ml/artifacts/razorpay_adapter_report.json
```

Reads local, synthetic JSON fixture files whose event names are modeled on Razorpay's publicly documented webhook events, normalizes them, aggregates them into merchant-week rows, and produces a mapping/data-quality report. **This does not call any Razorpay API and is not a Razorpay integration** — every output is labeled `razorpay_shaped_demo_fixture_not_live_razorpay_data`. See `docs/RAZORPAY_ADAPTER.md` for the verified event-name sources, the generic event schema, and the merchant-profile fields (support-ticket rate, delivery-evidence coverage, etc.) it honestly cannot produce from payment events alone.

## Intended stack

 - Python
 - FastAPI
 - Streamlit
 - SQLite
 - Pandas
 - scikit-learn
 - PyYAML
 - Joblib
 - Optional SHAP / Evidently

 ## Documentation

 - `PRD.md` — product requirements
 - `ARCHITECTURE.md` — component and data-flow design
 - `DATA_DICTIONARY.md` — synthetic-data fields and the latent-state label design
 - `RISK_POLICY.md` — illustrative risk rules and decisions
 - `MODEL_CARD.md` — baseline model scope and limitations
 - `SECURITY.md` — MVP security boundaries
 - `RESEARCH.md` — evidence, assumptions, and open questions
 - `docs/RESCOPE_REVIEW.md` — the 2026-08-22 rescope decision and open questions
 - `docs/SUBMISSION_PITCH.md` — hackathon submission pitch
 - `docs/DEMO_SCRIPT.md` — demo walkthrough script
 - `docs/IMPLEMENTATION_PLAN.md` — the full milestone breakdown
 - `docs/MILESTONE_1_DATA_QUALITY_REPORT.md`, `docs/MILESTONE_2_RULES_AND_FEATURES.md`, `docs/MILESTONE_3_MODEL_EVALUATION.md`, `docs/MILESTONE_4_EXPLAINABILITY.md`, `docs/MILESTONE_5_CASE_WORKFLOW.md`, `docs/MILESTONE_6_API.md` — per-milestone implementation reports with actual run output
 - `docs/CASE_PACKET_SCHEMA.md` — every case-packet field, labeled analyst-only/merchant-safe/internal-metadata/audit-preview
 - `docs/AUDIT_EVENT_SCHEMA.md` — every persisted audit event type, actor, payload, and state-transition effect
 - `docs/API_CONTRACT.md` — every API endpoint, request/response example, error code, and enum
 - `docs/MILESTONE_7_METRICS.md` — the persisted evaluation-report lifecycle and `/metrics` safe-exposure boundaries
 - `docs/MILESTONE_8_DASHBOARD.md` — dashboard page architecture, data flow, and safety controls
 - `docs/UI_DEMO_GUIDE.md` — exact startup commands and full demo walkthroughs for the Streamlit dashboard
 - `docs/RAZORPAY_ADAPTER.md` — the local Razorpay-shaped payment-event adapter: verified event names, generic event schema, merchant-week aggregation, and the fields it honestly cannot produce

 ## Status

 Milestones 1-8 implemented and approved: synthetic merchant-week data generation
 (Milestone 1), the feature-engineering and transparent rules engine (Milestone 2),
 the time-based split / Logistic Regression baseline / held-out evaluation (Milestone 3),
 deterministic explainability and reviewer case-packet generation (Milestone 4), SQLite
 persistence with a full case-review workflow, simulated merchant evidence, and an
 append-only application audit log (Milestone 5), a validated FastAPI read/workflow
 API layer over that same service layer (Milestone 6), a persisted offline evaluation
 report served by `GET /metrics` (Milestone 7), and a local Streamlit dashboard over the
 same API (Milestone 8). See `docs/IMPLEMENTATION_PLAN.md` for what remains
 (authentication, real integrations, and anything beyond this local Phase 1 prototype).
