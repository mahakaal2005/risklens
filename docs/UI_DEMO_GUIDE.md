# UI Demo Guide — ClearRisk Recover Dashboard

Local synthetic-data demonstration only. Everything below runs entirely on your machine — no internet connection is required once Python dependencies are installed.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Exact startup commands

Run these in order, from the repository root.

```bash
# 1. Generate synthetic data, train the model, and evaluate it
python3 ml/generate_synthetic_data.py
python3 -m ml.train_baseline_model
python3 -m ml.evaluate_model          # also writes ml/artifacts/latest_evaluation_report.json

# 2. Generate the 5 demo case packets
python3 -m ml.generate_demo_cases

# 3. Seed the local demo database (delete first for a fully fresh run)
rm -f clearrisk_recover.db
python3 scripts/seed_demo_cases.py
```

```bash
# 4. Start the FastAPI backend (separate terminal, leave running)
uvicorn app.main:app --reload
```

```bash
# 5. Start the Streamlit dashboard (separate terminal)
streamlit run dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

**If `streamlit run ...` fails with `ModuleNotFoundError: No module named 'google.protobuf'`:** your `streamlit` executable's shebang line points at a different Python interpreter than the one Streamlit's dependencies were installed into. Use `python3 -m streamlit run dashboard/streamlit_app.py` instead — this reliably uses the same `python3` your virtual environment/pyenv is configured to.

## Expected initial cases

After step 3, `scripts/seed_demo_cases.py` prints something like:

```text
stable_merchant: case_id=None status=NOT_CREATED (APPROVE)
seasonal_sale_false_positive_candidate: case_id=case_preview_xxxxxxxx status=OPEN
operational_fulfilment_problem: case_id=case_preview_xxxxxxxx status=OPEN
high_risk_combined_loss_case: case_id=case_preview_xxxxxxxx status=OPEN
early_hidden_risk_case: case_id=case_preview_xxxxxxxx status=OPEN

4 of 5 demo packets became persisted review cases.
```

The `stable_merchant` packet's recommendation is `APPROVE`, so — correctly — no case is created for it. The other 4 appear in the Review Queue, all `OPEN`.

## Five-page walkthrough

The sidebar carries navigation, the persistent safety statement, a quiet backend-status line, and the currently active case ID. Page bodies start with a one-line synthetic-data reminder.

1. **Overview** — product hero text, then (once `ml.evaluate_model` has run) a single comparison table of rules-only / Logistic Regression / combined-policy precision, recall, PR-AUC, and false-positive rate, the selected operating threshold, and the honest synthetic-data limitation. "How to read this comparison" and "Scope and safety boundaries" are expanders.
2. **Review Queue** — four status metrics, a **Filters** expander, and the case table. **Click a table row** to select a case, then click **Open case detail**.
3. **Case Detail** — a compact header strip (merchant, week, status, risk-signal badge, recommendation), then **Reviewer actions** immediately below it, then four tabs: *Why flagged*, *Evidence checklist*, *What the merchant sees*, *Analyst detail* (model probability, rules-only score, versions, and the known API gaps).
4. **Merchant Response** — defaults to the currently active case; the form only accepts a submission once the case is in `EVIDENCE_REQUESTED` status (request evidence from Case Detail first).
5. **Audit Timeline** — pick a case, see every event in order in one table, and click a row to see that event's safe payload detail.

## Seasonal-sale false-positive workflow

1. Review Queue → click that case's row → **Open case detail**. (Or go straight to Case Detail and pick it from the Case ID selector.) Use the `seasonal_sale_false_positive_candidate` case (`recommendation = REQUEST_EVIDENCE`).
2. Reviewer actions → **Request evidence**, enter a note, submit. Status becomes `EVIDENCE_REQUESTED`.
3. Merchant Response → same case → enter an explanation (e.g. "We ran a seasonal sale; refunds are within policy.") and evidence references (e.g. `invoice_demo_001.pdf, refund_policy_demo_url`) → **Submit response**. Status becomes `EVIDENCE_SUBMITTED`.
4. Case Detail → reviewer actions → **Start review**. Status becomes `UNDER_REVIEW`.
5. Case Detail → reviewer actions → **Mark false positive**, enter a note, submit. Status becomes `RESOLVED`, final outcome `FALSE_POSITIVE`.
6. Audit Timeline → same case → confirm all 9 events appear in order in the table, ending in `CASE_MARKED_FALSE_POSITIVE`. Click any row to inspect that event's payload.

## Operational-issue workflow

Same shape, on the `operational_fulfilment_problem` case:

1. Request evidence ("Please share fulfilment/delivery proof and support records.").
2. Merchant Response → submit (e.g. `delivery_proof_demo_001.pdf, support_log_summary_demo`).
3. Start review.
4. **Mark operational issue** → `RESOLVED` / `OPERATIONAL_ISSUE`.

## High-risk combined-loss walkthrough

The `high_risk_combined_loss_case` packet already has `recommendation = MANUAL_REVIEW_REQUIRED` and multiple triggered rules (including `COMBINED_LOSS_SIGNAL`) visible in Case Detail's "Why this case was flagged" section. From `OPEN`, a reviewer can go directly to **Mark false positive** / **Mark operational issue** / **Mark inconclusive**, or **Escalate case**, without necessarily requesting evidence first — demonstrating that high-signal cases don't require the evidence step before resolution.

## Reset instructions

To start over from a completely clean state:

```bash
rm -f clearrisk_recover.db
python3 scripts/seed_demo_cases.py
```

Restart the FastAPI process afterward (`uvicorn app.main:app --reload`) — a running FastAPI process may hold an open connection to the old database file; restarting it ensures it picks up the freshly seeded one.

To regenerate the underlying data/model/report from scratch, repeat step 1 of the startup commands before reseeding.

## Troubleshooting

**"Backend unavailable — start with: `uvicorn app.main:app --reload`"** appears on every dashboard page: the FastAPI process isn't running, isn't on port 8000, or crashed. Check the terminal running `uvicorn` for errors. If you changed the port, set `CLEARRISK_API_BASE_URL` (e.g. `export CLEARRISK_API_BASE_URL=http://127.0.0.1:8001`) before starting Streamlit.

**Overview page shows "Evaluation metrics are not available yet."**: `ml/artifacts/latest_evaluation_report.json` doesn't exist. Run `python3 -m ml.evaluate_model` (after `train_baseline_model.py` has already produced a model artifact) and refresh the page.

**Review Queue / Case Detail / Audit Timeline show "No cases exist yet."**: the database is empty or was reset. Run `python3 scripts/seed_demo_cases.py` and restart `uvicorn`.

**Streamlit shows stale case data after an action**: most actions call `st.rerun()` automatically; if a page still looks stale, use your browser's refresh or re-select the case from the dropdown.
