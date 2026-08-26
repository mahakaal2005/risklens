# Phase 2 Design — Authentication and Basic Roles

**Status: design proposal, not yet implemented. Awaiting approval before any code is written.**

This is the first Phase 2 workstream per `CLAUDE.md`'s roadmap ("Add authentication
and basic roles"). Written on branch `phase-2-auth-design`, isolated from `main`
(which stays the clean, working Phase 1 baseline) precisely so this can be
abandoned cleanly if it doesn't pan out — see Section 7.

## 1. Goal

Replace the current hardcoded actor constants (`DEMO_REVIEWER_ACTOR =
"analyst_demo_001"` in `dashboard/components/reviewer_actions.py`,
`DEMO_MERCHANT_ACTOR = "merchant_demo_001"` in `evidence_form.py`, and the
client-supplied `reviewer_actor_id`/`merchant_actor_id` request fields they feed
into the API) with real login, session-based identity, and three permission
tiers, so the audit trail records who actually performed an action instead of a
constant string.

## 2. Non-goals (explicitly out of scope for this pass)

- Production-grade auth: no MFA, no password reset flow, no login rate
  limiting/lockout, no OAuth/SSO, no external identity provider.
- Multi-tenant isolation, RBAC beyond the three tiers below.
- Encryption at rest for the SQLite file, secrets management, or any of the
  other items `SECURITY.md` already lists as out of scope for Phase 1/2.
- Real user self-registration. Users are seeded, fixed demo accounts — same
  spirit as `merchant_demo_001` today, just now requiring a real password to
  act as that identity instead of anyone being able to claim it.
- Changing any rule/model/scoring logic, the case state machine, or the audit
  event schema's *shape* (only who triggers events changes, not the schema).

## 3. Where auth is enforced (per your prior answer)

FastAPI-level, not dashboard-level decoration. Every write endpoint requires a
valid session token; the token — not the request body — determines the actor
recorded in the audit log. `GET /health` stays unauthenticated (the dashboard
needs it to show connection status before login). Read endpoints (`GET
/cases`, `GET /cases/{id}`, `GET /cases/{id}/audit-events`, `GET /metrics`)
require *some* authenticated session, but any role may read them (with one
restriction for the merchant role — Section 5).

## 4. Data model

Two new tables, added via SQLAlchemy models in `app/db/models.py` (this
project has no formal migration framework yet — `Base.metadata.create_all`
is how every existing table got created; new tables follow the same path,
so a fresh `rm -f clearrisk_recover.db` + reseed picks them up, exactly like
every other schema change so far).

```text
users
  id                  (pk)
  username            (unique)
  password_hash       (pbkdf2_hmac output, hex)
  password_salt       (random, per-user)
  role                ("reviewer" | "merchant" | "risk_manager")
  actor_id            (e.g. "analyst_demo_001" -- what gets recorded in audit events)
  merchant_id         (nullable; only set for role="merchant", restricts which
                        cases that user may submit evidence for)
  display_name
  created_at

sessions
  token               (pk, opaque random string, e.g. secrets.token_urlsafe(32))
  user_id             (fk -> users.id)
  created_at
  expires_at          (e.g. created_at + 12 hours -- a local demo session, not
                        a production refresh-token scheme)
```

Password hashing: stdlib `hashlib.pbkdf2_hmac("sha256", password, salt,
260_000)` — no new dependency (`requirements.txt` stays untouched), a
reasonable local-demo KDF, explicitly documented in `SECURITY.md` as adequate
for this prototype and *not* a claim of production password-security
compliance.

## 5. Role → permission mapping (3 tiers, per your prior answer)

| Role | Can do |
|---|---|
| `reviewer` (merges analyst + compliance-reviewer — CLAUDE.md doesn't yet differentiate their actions) | All existing reviewer actions on any case: request evidence, clear, mark false positive/operational issue/inconclusive, escalate, start review. |
| `merchant` | Submit evidence (`POST /cases/{id}/evidence`) **only** for cases whose `merchant_id` matches their own `merchant_id`. Read access to cases (see caveat below). |
| `risk_manager` | Read-only: `GET /metrics`, `GET /cases`, `GET /cases/{id}`, `GET /cases/{id}/audit-events`. No write endpoint accepts this role. |

**Read-access caveat, flagged for your input**: should a `merchant`-role user's
`GET /cases`/`GET /cases/{id}` be filtered to only their own `merchant_id`
(so one merchant can't browse another's case detail), or is unrestricted read
access acceptable for this local single-operator demo? I'd default to
**filtered** (merchants only ever see their own cases) since it costs little
and is the more defensible design, but it's your call — noted as an open
question in Section 9.

## 6. The breaking API change (still open — your last answer asked me to hold off deciding)

Today, `POST /cases/{id}/actions` accepts `reviewer_actor_id` in the body, and
`POST /cases/{id}/evidence` accepts `merchant_actor_id` — the caller names
whoever they want to be. Two options:

- **A — Remove both fields; derive the actor entirely from the session
  token.** This is the only way the auth layer is real enforcement rather
  than decoration: a valid session for user X cannot be used to attribute an
  action to user Y. Breaks `docs/API_CONTRACT.md`'s current request shape,
  `app/schemas/cases.py`/`evidence.py`, `dashboard/api_client.py`'s method
  signatures, and two existing tests in `tests/test_dashboard_api_client.py`
  (`test_review_action_sends_only_allowed_fields`,
  `test_evidence_submission_sends_only_allowed_fields`) that assert those
  fields are sent — those tests get rewritten, not just patched.
- **B — Keep both fields, but require they match the authenticated session's
  `actor_id`/`merchant_id` (else 403).** Preserves the current request shape
  and every existing test's expectations about the JSON body's field names.
  Weaker: it's cross-checking, not deriving — the field still nominally
  exists as attacker-controlled input, just rejected if it disagrees with
  the session. Functionally equivalent security outcome for a single-session
  actor, easier migration, more redundant surface area.

**Recommendation: A.** A field that exists only to be validated against the
thing that should have generated it in the first place is dead weight once
the session exists — it's the "trust but verify" pattern applied to
something that shouldn't need trusting at all. But this is exactly the kind
of decision worth confirming explicitly before 15+ files change, which is
why it's called out here rather than assumed.

## 7. Rollback / go/no-go plan (why this is on its own branch)

- `main` is frozen at the Phase 1 baseline commit (`9504e99`) — the working,
  fully-tested, demo-ready state. Nothing here touches it.
- All Phase 2 auth work happens on `phase-2-auth-design`. If, partway
  through, this doesn't pan out (time constraints before a submission
  deadline, or a decision to keep Phase 1 as the final submission), the
  branch is simply not merged — `main` is untouched and still demo-ready.
- If approved and completed, the branch merges into `main` (or becomes the
  new `main`) only once the full test suite passes and you've reviewed the
  running dashboard.
- Mid-flight checkpoint: after DB models + auth service + FastAPI routes are
  done but before the dashboard/tests are touched, I'll stop and report —
  matching the project's established "stop and report before continuing"
  discipline — so you can call it there if needed without half-finished
  dashboard code in the way.

## 8. File-by-file plan

New files:
- `app/services/auth_service.py` — hashing, session creation/validation, `seed_demo_users()`.
- `app/schemas/auth.py` — `LoginRequest`, `LoginResponse`, `CurrentUserResponse`.
- `app/api/routes/auth.py` — `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`.
- `scripts/seed_demo_users.py` — seeds the three fixed demo accounts (documented, printed credentials for the demo — these are placeholder accounts, not real people, consistent with `merchant_demo_001` etc. elsewhere in this codebase).
- `dashboard/components/login.py` — login form, shown before any page when unauthenticated.
- `docs/MILESTONE_9_AUTH.md` — the as-built report, following this project's per-milestone documentation pattern.
- `tests/test_auth_service.py`, `tests/test_api_auth.py`.

Changed files:
- `app/db/models.py` — add `User`, `Session` models.
- `app/api/dependencies.py` — add `get_current_user`, `require_role(*roles)`.
- `app/api/routes/cases.py`, `evidence.py`, `metrics.py` — apply auth dependencies; derive actor from session (pending Section 6's decision); merchant-role case filtering (pending Section 5's decision).
- `app/schemas/cases.py`, `evidence.py` — remove or adjust the actor-id request fields per Section 6's decision.
- `dashboard/api_client.py` — add `login()`/`logout()`; attach `Authorization: Bearer <token>` to every call; drop the actor-id parameters from `submit_review_action`/`submit_evidence` if option A is chosen.
- `dashboard/streamlit_app.py` — gate all pages behind login; role-based sidebar page visibility (e.g. `merchant` role sees only Merchant Response; `risk_manager` sees only Overview/Audit Timeline read-only).
- `dashboard/components/reviewer_actions.py`, `evidence_form.py` — remove hardcoded actor constants, use the logged-in session's identity.
- `docs/API_CONTRACT.md`, `SECURITY.md`, `ARCHITECTURE.md` — updated to describe the new auth layer and its explicit non-goals.
- `tests/test_api_cases.py`, `test_api_evidence.py`, `test_dashboard_api_client.py`, `test_dashboard_safety.py` — updated for the new required auth header and (if option A) the changed request shape.

## 9. Open questions for you

1. Section 6: option A (remove actor-id fields, derive from session) or B (keep fields, cross-check against session)? Recommendation: A.
2. Section 5: should merchant-role read access to `GET /cases`/`GET /cases/{id}` be filtered to their own `merchant_id`, or left unrestricted for this single-operator local demo? Recommendation: filtered.
3. Session lifetime — 12 hours suggested above (long enough for one demo/judging session, short enough to not feel like a permanent credential). Any preference?
4. Do you want the three demo accounts' passwords printed once by the seed script (visible in your terminal, not committed anywhere), or would you rather set them yourself via an environment variable at seed time?

Awaiting your answers and go-ahead before touching any file under Section 8.
