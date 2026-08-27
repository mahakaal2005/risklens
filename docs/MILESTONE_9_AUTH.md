# Milestone 9 — Phase 2 Authentication and Basic Roles

Status: Implemented and verified end to end (real HTTP calls against a running FastAPI backend, plus a headless dashboard walkthrough for all three roles). **Local synthetic-data, local-demo authentication only — not production-grade.** See `docs/PHASE_2_AUTH_DESIGN.md` for the approved design and its non-goals.

---

## 1. What this milestone adds

Real login, session-based identity, and three permission tiers, replacing the previous hardcoded actor constants (`DEMO_REVIEWER_ACTOR = "analyst_demo_001"`, `DEMO_MERCHANT_ACTOR = "merchant_demo_001"`) that let any dashboard user attribute any action to any name. The audit trail now records who actually performed an action, derived from an authenticated session — never from client-supplied input.

## 2. Data model

Two new SQLAlchemy tables in `app/db/models.py`, created automatically via the existing `Base.metadata.create_all` path (no formal migration framework in this codebase):

- `users` — username, `password_hash`/`password_salt`, `role` (`reviewer`/`merchant`/`risk_manager`), `actor_id` (recorded in audit events), `merchant_id` (nullable, only set for `role="merchant"`), `display_name`.
- `sessions` — opaque token (`secrets.token_urlsafe(32)`), `user_id`, `created_at`, `expires_at` (12-hour fixed lifetime).

Password hashing: stdlib `hashlib.pbkdf2_hmac("sha256", ..., 260_000)` — no new dependency, adequate for this local single-operator prototype, explicitly not a claim of production password-security compliance (see `SECURITY.md`).

## 3. Auth service (`app/services/auth_service.py`)

- `hash_password()` / `verify_password()` — salted PBKDF2, constant-time comparison via `secrets.compare_digest`.
- `create_user()` / `authenticate()` — `authenticate()` raises a single generic `AuthError("Invalid username or password.")` for both an unknown username and a wrong password, never revealing which.
- `create_session()` / `get_current_user()` / `invalidate_session()` — session lifecycle. `get_current_user()` returns `None` (never a fabricated identity) for a missing, unknown, or expired token.

**Bug found and fixed during implementation:** SQLite does not reliably round-trip timezone-aware `DateTime(timezone=True)` values — a session's `expires_at`, written as UTC-aware, came back **naive** on read, causing `TypeError: can't compare offset-naive and offset-aware datetimes` the first time an expiry check ran against real persisted data (caught immediately by the existing API test suite once auth was wired in — every previously-passing test failed with this error). Fixed with `_as_aware_utc()`, which treats any naive datetime read back from the database as UTC before comparing.

## 4. FastAPI enforcement (`app/api/dependencies.py`, `app/api/routes/*.py`)

- `get_current_user` — a dependency requiring a valid `Authorization: Bearer <token>` header; **401** `AUTHENTICATION_REQUIRED` for missing/invalid/expired.
- `require_role(*roles)` — layered on top of `get_current_user`; **403** `FORBIDDEN` if the role doesn't match. Always 401-before-403 (an unauthenticated request never becomes a 403).
- `GET /health`, `POST /auth/login`, `POST /auth/logout` are the only unauthenticated routes (health so the dashboard can show connection status pre-login).
- `POST /cases/{id}/review-actions` requires `role="reviewer"`; the actor is `user.actor_id` from the session, never a request field.
- `POST /cases/{id}/evidence` requires `role="merchant"` **and** `case.merchant_id == user.merchant_id` — a mismatch returns **404** (not 403), so a merchant gets no signal that another merchant's case even exists.
- `GET /cases` / `GET /cases/{id}` / `GET /cases/{id}/audit-events` require any authenticated role; a `merchant`-role caller's list/detail reads are filtered to their own `merchant_id` (`app/db/repositories.py::list_cases()` gained an optional `merchant_id` filter parameter for this).
- `GET /metrics` requires any authenticated role (read-only for everyone).

## 5. The approved breaking change

`reviewer_actor_id` and `merchant_actor_id` were removed entirely from `ReviewActionRequest` and `EvidenceSubmissionRequestBody` (`app/schemas/api_responses.py`). Before this milestone, any caller could claim to be any actor by simply naming them in the request body — auth would have been decorative without this change. See `docs/PHASE_2_AUTH_DESIGN.md` Section 6 for the approved rationale.

## 6. Dashboard integration

- `dashboard/components/login.py` — a login form shown whenever `st.session_state` has no `session_token`/`current_user`.
- `dashboard/api_client.py` — gained `login()`/`logout()`; every request now attaches `Authorization: Bearer <token>` when a session token is set on the client instance. `submit_review_action()`/`submit_evidence()` lost their actor-id parameters to match the backend change.
- `dashboard/streamlit_app.py` — gates all 5 pages behind login; `PAGES_BY_ROLE` filters the sidebar navigation per role (a **display convenience only** — the backend is the real authority, and an out-of-scope request still gets safely rejected server-side even if a page happened to be reachable):

  | Role | Visible pages |
  |---|---|
  | `reviewer` | Overview, Review Queue, Case Detail, Audit Timeline |
  | `merchant` | Merchant Response, Case Detail, Audit Timeline |
  | `risk_manager` | Overview, Audit Timeline |

  A "Sign out" button in the sidebar calls `client.logout()` and clears the session from `st.session_state`.
- `dashboard/components/reviewer_actions.py` and `evidence_form.py` — the hardcoded `DEMO_REVIEWER_ACTOR`/`DEMO_MERCHANT_ACTOR` constants are gone; identity now comes entirely from the logged-in session.

## 7. Demo accounts (`scripts/seed_demo_users.py`)

Three fixed, seeded accounts — not real people, same spirit as `merchant_demo_001` elsewhere in this codebase:

| Username | Role | Notes |
|---|---|---|
| `reviewer_demo` | `reviewer` | Can act on any case. |
| `merchant_demo` | `merchant` | Bound to `merchant_id=merchant_demo_0020` — the `seasonal_sale_false_positive_candidate` demo case, so the account can actually exercise the evidence-submission flow against a real seeded case. |
| `riskmanager_demo` | `risk_manager` | Read-only. |

Passwords are randomly generated (`secrets.token_urlsafe(9)`) and printed to the terminal once on first run; nothing is written to any file. Re-running the script for an already-seeded username is a safe no-op (reported, not duplicated or reset).

## 8. Tests

- `tests/test_auth_service.py` (12 tests) — hashing determinism/salting, `authenticate()`'s generic-message behavior for both wrong-password and unknown-username, session creation/expiry/invalidation.
- `tests/test_api_auth.py` (7 tests) — `POST /auth/login` success/failure, `GET /auth/me`, `POST /auth/logout` (including as a safe no-op with no token).
- `tests/test_api_cases.py`, `test_api_evidence.py`, `test_api_metrics.py` — updated to authenticate via a shared `tests/conftest.py::make_bearer_headers()` helper; added explicit tests for unauthenticated rejection, cross-merchant evidence-submission rejection (404), and reviewer-role evidence-submission rejection (403).
- `tests/test_dashboard_api_client.py` — updated for the new `headers` parameter on every mocked `httpx.request` call; added tests for `login()` storing and attaching the token, and `logout()` clearing it.
- Full suite: **402 passed** (up from 379 before this milestone).

## 9. How to run

```bash
rm -f clearrisk_recover.db
python3 scripts/seed_demo_cases.py
python3 scripts/seed_demo_users.py    # prints demo login credentials once -- save them
uvicorn app.main:app --reload          # terminal 1
streamlit run dashboard/streamlit_app.py  # terminal 2
```

Open [http://localhost:8501](http://localhost:8501) and sign in with any of the three printed accounts.

## 10. Known limitations (not fabricated, flagged instead)

- No MFA, no password reset flow, no login rate limiting/lockout, no OAuth/SSO, no external identity provider.
- Sessions are opaque server-side tokens with a fixed 12-hour lifetime — a local demo session, not a production refresh-token scheme.
- No self-registration; accounts are seeded, fixed demo identities only.
- `analyst` and `compliance-reviewer` (two distinct roles named in `CLAUDE.md`'s user-roles section) are merged into one `reviewer` permission tier for v1, since the current action set doesn't yet differentiate their permissions — a documented simplification, not an oversight.
