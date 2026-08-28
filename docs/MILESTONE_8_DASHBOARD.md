# Milestone 8 — Local Streamlit Dashboard

Status: Implemented and verified end to end (all 5 pages, all 3 named demo workflows) against the real FastAPI backend and real SQLite database. **Local synthetic-data demonstration only.**

---

## 1. Page architecture

```text
dashboard/streamlit_app.py         -- sidebar nav (5 pages), disclaimer + connection status on every page
  ├── components/metrics.py        -- Overview
  ├── components/case_list.py      -- Review Queue
  ├── components/case_detail.py    -- Case Detail (embeds reviewer_actions.py)
  ├── components/evidence_form.py  -- Merchant Response
  └── components/audit_timeline.py -- Audit Timeline

components/common.py               -- shared disclaimer, connection status, intensity badges, formatting
components/reviewer_actions.py     -- state-dependent reviewer action buttons, embedded in Case Detail
```

Every page function takes the single shared `ClearRiskAPIClient` instance and returns nothing — all data comes from the API client, all mutation happens through it too.

## 2. Data flow

```text
Streamlit page function
   → dashboard/api_client.py (ClearRiskAPIClient)
       → HTTP GET/POST to http://127.0.0.1:8000 (or $CLEARRISK_API_BASE_URL)
           → FastAPI routes (app/api/routes/*.py)
               → existing service layer (app/services/*.py)
                   → SQLite (clearrisk_recover.db) / ml/artifacts/*
```

No dashboard file imports `app.db`, `app.services`, or `ml.*` directly — verified by `tests/test_dashboard_safety.py::test_dashboard_does_not_import_sqlite_or_sqlalchemy_directly`. All model/rules/state-machine/persistence/audit logic stays exactly where Milestones 2–7 put it; the dashboard only ever calls the existing FastAPI endpoints.

## 3. API-client design (`dashboard/api_client.py`)

- Base URL from `CLEARRISK_API_BASE_URL`, default `http://127.0.0.1:8000`.
- Single `httpx.request(...)` call site (`_request()`), used by every method — all HTTP concerns (timeout, connection errors, malformed JSON, HTTP error status) are normalized into one `DashboardAPIError` type with a short, safe `.message`. No raw `httpx` exception or backend stack trace ever reaches a `st.error(...)` call.
- `_require_notice()` enforces that every response expected to carry `synthetic_data_notice` actually has one — a response missing it is treated as unsafe and raises `DashboardAPIError`, not silently accepted.
- Exactly 7 public methods: `health()`, `get_metrics()`, `list_cases()`, `get_case()`, `get_audit_events()`, `submit_review_action()`, `submit_evidence()` — matching the Milestone 8 spec's required method list one-to-one.
- Write methods (`submit_review_action`, `submit_evidence`) construct their JSON body from named parameters only — no caller can smuggle an extra field through, verified by `tests/test_dashboard_api_client.py::test_review_action_sends_only_allowed_fields` / `test_evidence_submission_sends_only_allowed_fields`.

## 4. Local/offline operation

- Default base URL is `127.0.0.1` — never a public hostname.
- No `requests`/`httpx` call anywhere in `dashboard/` targets any host other than the configured base URL.
- No external image, font, CDN asset, or JavaScript library is loaded — `dashboard/assets/README.md` documents this is intentional; Streamlit's own built-in widgets are the entire UI.
- Verified by `tests/test_dashboard_safety.py::test_no_external_image_cdn_font_or_llm_urls`, which regex-scans all dashboard source for any `http(s)://` URL and asserts every one starts with `http://127.0.0.1` or `http://localhost`.

## 5. Safety controls

- **Global disclaimer** (`components/common.py::GLOBAL_DISCLAIMER`) rendered via `st.sidebar.warning(...)`, so the full statement is on screen for the entire session on every page without consuming the top of each page body. Each page body additionally opens with a one-line `PAGE_NOTICE` caption ("Synthetic demo data · review recommendations only · not a final fraud decision.").
- **Connection status** (`render_connection_status()`) calls `GET /health`. Healthy is quiet — a single sidebar caption (`🟢 Backend connected · synthetic-only`). Broken is loud — a sidebar `🔴` plus a full-width `st.error` in the page body with the exact start command. It never falls back to mock data. A working backend is the expected case and does not warrant a full-width alert on all five pages.
- **No forbidden action language anywhere** in dashboard source except inside the disclaimer's own negation sentence (which must state, correctly, that the tool does *not* do these things) — enforced by `tests/test_dashboard_safety.py::test_no_prohibited_action_labels_in_source`.
- **No forbidden data fields** (`label_high_loss_next_30d`, `latent_state_for_demo_only`, `support_ticket_rate`) appear anywhere in dashboard source.
- **Merchant-safe preview** (Case Detail, *What the merchant sees* tab) renders only `case["merchant_safe_explanation"]` — never `model_probability`, `rules_only_score`, `policy_explanation`, or triggered-rule internals. Those live in a separate *Analyst detail* tab, which is explicitly captioned "Analyst-only. Never shown to the merchant."
- **Risk intensity** always shown with both a color AND a text label (`intensity_badge()`) — never color alone.

## 6. State-dependent reviewer actions (`components/reviewer_actions.py`)

`ACTIONS_BY_STATUS` maps each case status to only the actions valid from that status, mirroring (but not re-implementing) the state machine already enforced server-side in `app/services/case_service.py`:

| Status | Offered actions |
|---|---|
| `OPEN` | Request evidence, Clear case, Mark false positive, Mark operational issue, Mark inconclusive, Escalate case |
| `EVIDENCE_REQUESTED` | Escalate case only, plus an informational "Waiting for merchant evidence." notice |
| `EVIDENCE_SUBMITTED` | Start review, Escalate case |
| `UNDER_REVIEW` | Clear case, Mark false positive, Mark operational issue, Mark inconclusive, Escalate case |
| `RESOLVED` | No buttons — final outcome, reviewer note, and resolved timestamp only |
| `ESCALATED` | No buttons — escalation status only, no automatic action |

The dashboard's action list is a *display* convenience, not the source of truth — an out-of-band or stale request still gets safely rejected by the backend's own state machine (409), and the dashboard renders that error via `render_error()` rather than crashing.

## 7. Error / empty states

- **Backend unreachable:** every page shows the connection-status banner; page bodies that depend on the API (Review Queue, Case Detail, Merchant Response, Audit Timeline) show `render_error()` inline rather than a raw exception.
- **No cases seeded yet:** Review Queue shows "No cases match the current filters."; Case Detail, Merchant Response, and Audit Timeline show "No cases exist yet. Seed demo cases first (see docs/UI_DEMO_GUIDE.md)."
- **Metrics unavailable:** Overview shows a non-alarming `st.info(...)` with the exact command to run (`python3 -m ml.evaluate_model`), never a fabricated number.
- **Evidence submitted in the wrong state:** Merchant Response shows "Evidence can be submitted only after a reviewer requests it." instead of a raw 409.
- **Invalid evidence reference:** the backend's 422 message (already safe, non-technical) is shown via `render_error()`.

## 8. Known data gaps (not fabricated, flagged instead)

Two Milestone 8 "nice to have" items are not currently exposed by the API and are shown as explicit "not available" notices rather than invented:

- **Trend view — partially fixed.** Per-triggered-rule concrete before/after values (e.g. "Refund rate increased from 1.63% to 6.45%") are now persisted (`ReviewCase.triggered_rule_explanations_json`) and shown in the "Why flagged" tab — this closed a real gap against `CLAUDE.md`'s Core Promise, found while auditing the project. Still not available: current-vs-prior trend values for features that did *not* trigger a rule, and a full refund/chargeback/volume/evidence/support-resolution chart. Stated plainly rather than fabricated.
- **Top model factors** — a ranked list of every feature's contribution (not just triggered-rule ones) is still not persisted, so not shown, with a note rather than a fabricated ranking.

Both notes now live as a single caption at the bottom of the *Analyst detail* tab rather than occupying two full mid-page sections, so the absence is still disclosed without dominating the reviewer's screen.

One related field **was** added this milestone, with explicit approval first: `model_probability` and `rules_only_score` existed in the database since Milestone 5 but were never mapped into `CaseDetailResponse` (a Milestone 6 gap). This was reported and approved before `app/schemas/api_responses.py` and `app/api/routes/cases.py` were touched — see the deliverable report for the exact defect and fix.

## 9. Screenshot instructions (no browser screenshots are attached to this repo)

To capture your own screenshots for a submission deck:

1. `uvicorn app.main:app --reload` (terminal 1), `streamlit run dashboard/streamlit_app.py` (terminal 2, or `python3 -m streamlit run ...` if your `streamlit` executable's shebang points at a different Python than the one with Streamlit's dependencies installed — see `docs/UI_DEMO_GUIDE.md` troubleshooting).
2. Open `http://localhost:8501`.
3. Overview → screenshot the hero + model-comparison cards.
4. Review Queue → screenshot the summary cards + filtered table.
5. Case Detail → screenshot the case summary, "why flagged", evidence checklist, and reviewer actions sections separately.
6. Merchant Response → screenshot the form on an `EVIDENCE_REQUESTED` case.
7. Audit Timeline → screenshot the full ordered event list with one expander open.

## 10. Known limitations

- Local SQLite only; no authentication; synthetic demo data only.
- No real file uploads, document storage, or URL fetching anywhere.
- No real merchant communication or payment-gateway integration.
- Append-only application audit log is not cryptographically immutable (stated explicitly on the Audit Timeline page).
- No payment, settlement, hold, ban, termination, or enforcement action exists anywhere in the UI or the API it calls.
- No real-time model monitoring — `/metrics` reads a static, offline-generated report (Milestone 7).
- Per-triggered-rule concrete trend values are shown (Section 8 above); a full ranked top-model-factor display and non-triggered-rule trend values remain "not available" placeholders pending a future milestone's decision on whether to persist that data.
