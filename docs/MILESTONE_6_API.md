# Milestone 6 — FastAPI Read and Workflow API Layer

Status: Implemented and smoke-tested against real seeded demo data. **Local synthetic-data demonstration only** — no authentication, no external gateway, no real file upload, no messaging, no background jobs, no automatic retraining, and no real payment/settlement/enforcement action exists anywhere in this API.

---

## 1. Architecture diagram

```text
┌─────────────────────────┐
│ Streamlit UI (future,    │   Not implemented yet -- see docs/IMPLEMENTATION_PLAN.md
│ Milestone 7+)            │
└───────────┬──────────────┘
            │ HTTP (JSON)
            ▼
┌─────────────────────────────────────────────────────────┐
│ FastAPI app (app/main.py)                                │
│  ┌───────────────┐ ┌───────────────┐ ┌────────────────┐ │
│  │ health router  │ │ cases router  │ │ evidence router│ │
│  └───────────────┘ └───────┬───────┘ └────────┬────────┘ │
│  ┌────────────────┐        │                   │         │
│  │ metrics router │        │                   │         │
│  └────────┬───────┘        │                   │         │
└───────────┼─────────────────┼───────────────────┼─────────┘
            │ reads only       │ delegates          │ delegates
            ▼                  ▼                    ▼
  ml/artifacts/          app/services/        app/services/
  evaluation_report.json case_service.py      evidence_service.py
                                │                    │
                                └─────────┬──────────┘
                                          ▼
                              app/services/audit_service.py
                                          │
                                          ▼
                              app/db/repositories.py
                                          │
                                          ▼
                                    SQLite (local file)
```

## 2. Routing / service-layer separation

No route handler in `app/api/routes/*.py` calls `app/db/repositories.py` to *mutate* state directly. Every state-changing endpoint (`POST /cases/{id}/review-actions`, `POST /cases/{id}/evidence`) calls the existing Milestone 5 service functions (`case_service.apply_reviewer_action`, `case_service.start_review`, `evidence_service.submit_evidence`), which enforce the state machine and write audit events. Routes only:

1. Validate the request shape (via Pydantic) and basic presence checks (non-empty note/actor ID).
2. Call the service function.
3. Catch the service layer's typed exceptions (`CaseNotFoundError`, `InvalidTransitionError`, `ValueError`) and translate them into the documented HTTP status codes and error schema.
4. Map the returned ORM object to a safe Pydantic response schema.

Read-only endpoints (`GET /cases`, `GET /cases/{id}`, `GET /cases/{id}/audit-events`) call `app/db/repositories.py` directly, since listing/reading is pure data access with no workflow logic — the one addition made in this milestone, `repositories.list_cases()`, is a plain filtered/paginated `SELECT`, not business logic.

## 3. Database-session handling

`app/api/dependencies.py::get_db()` is a FastAPI generator dependency yielding one SQLAlchemy `Session` per request, committing on success and rolling back on any exception, always closing in a `finally` block. The underlying engine/session factory is created **lazily** (`_get_session_factory()`, on first real use) rather than at module import time — this matters because tests override `get_db` entirely via `app.dependency_overrides`, and eager initialization at import time would have created/touched the developer's real `clearrisk_recover.db` file merely by importing `app.main`, before any test fixture had a chance to override the dependency. Verified by `tests/test_api_cases.py`'s fixture using a `tmp_path`-based temporary SQLite file, and independently confirmed by checking the real database file's mtime is unchanged after a full API test run.

**Threading note:** FastAPI's `TestClient` (and production ASGI servers) run synchronous route handlers in a worker thread pool. A `sqlite:///:memory:` engine creates a *new, empty* in-memory database per connection unless a shared/static pool is configured — since the existing `create_db_engine()` (Milestone 5) does not set `poolclass=StaticPool`, API tests use a temporary **file-based** SQLite database (`tmp_path / "test.db"`) instead of `:memory:`, which has no such cross-thread isolation problem. This is a documented test-design choice, not a change to `app/db/database.py`.

## 4. Workflow endpoint examples

See `docs/API_CONTRACT.md` for full request/response bodies. Summary flow matching `scripts/demo_case_workflow.py`'s seasonal-sale case, now via HTTP:

1. `GET /cases?recommendation=REQUEST_EVIDENCE` → find the seasonal-sale case.
2. `POST /cases/{id}/review-actions` with `{"action": "REQUEST_EVIDENCE", ...}` → `EVIDENCE_REQUESTED`.
3. `POST /cases/{id}/evidence` with merchant explanation + references → `EVIDENCE_SUBMITTED`.
4. `POST /cases/{id}/review-actions` with `{"action": "START_REVIEW", ...}` → `UNDER_REVIEW`.
5. `POST /cases/{id}/review-actions` with `{"action": "MARK_FALSE_POSITIVE", ...}` → `RESOLVED` / `FALSE_POSITIVE`.
6. `GET /cases/{id}/audit-events` → full 9-event ordered timeline.

## 5. Test strategy

- **Isolated database per test:** every API test file creates its own temporary file-based SQLite database (`tmp_path`) and overrides `get_db` via `app.dependency_overrides`, cleared in fixture teardown. The developer's real `clearrisk_recover.db` is never opened by the test suite (independently verified: file mtime unchanged after a full test run).
- **Seeding via the service layer, not the API:** test fixtures call `case_service.create_case_from_packet()` directly against the temp database (reusing the Milestone 4 demo packets), rather than duplicating packet-construction logic in test code.
- **Safety-negative tests:** every list/detail/audit-timeline response is checked for the literal absence of `label_high_loss_next_30d`, `latent_state_for_demo_only`, `support_ticket_rate`, and the 8 prohibited enforcement words — including a test that scans the live OpenAPI schema itself (`app.openapi()`), not just example responses.
- **State-machine tests:** valid transition, invalid transition (409, confirmed no mutation via before/after comparison), evidence-state gating (409 outside `EVIDENCE_REQUESTED`), and validation failures (422) are each tested explicitly.
- **Metrics artifact tests:** both the "available" and "not_available" code paths are exercised via `monkeypatch` on the module-level artifact path constant, plus an explicit test proving the model-training function is never called by the endpoint.

## 6. Limitations

- No authentication or authorization exists — any client that can reach the local port can call every endpoint.
- No real file upload or URL retrieval exists — evidence is validated strings only.
- `GET /metrics` currently returns `status: "not_available"` in the real (non-test) environment, because no prior milestone has persisted an `ml/artifacts/evaluation_report.json` file yet — `ml/evaluate_model.py` was out of this milestone's file scope, so its output remains stdout-only. This is documented, not hidden: see `docs/API_CONTRACT.md`'s `/metrics` section for the real current response.
- No background jobs, scheduled tasks, or automatic retraining exist.
- No external gateway, email/SMS/WhatsApp, or payment integration exists.
- This API is local-only (default `127.0.0.1:8000`), single-process, and has no production hardening, rate limiting, or TLS configuration.
