 # Security and Privacy Boundaries — MVP

 Status: Design proposal, rescoped 2026-08-22 to a single flagship scenario (merchant refund/chargeback loss risk). See `docs/RESCOPE_REVIEW.md`.

 ## Scope

 This document describes the security posture of a local, synthetic-data demonstration

 ## What the MVP protects against

 - Accidental inclusion of real payment credentials through schema restrictions.
 - Basic malformed input through API validation.
 - Accidental secret commits through `.env.example` and environment-variable configura
 - Loss of decision traceability through append-only application audit events.

 ## Non-negotiable data restrictions

 Never store or process:

 - Card PAN/card number
 - CVV
 - UPI PIN
 - Bank account credentials
 - Real Aadhaar/PAN
 - Real customer identity/contact data
 - Real merchant identity/bank details
 - Real payment tokens

 Use synthetic tokens such as `merchant_demo_001`. Per-transaction/customer/device tokens
 from the earlier four-scenario scope are no longer part of the scored entity (see
 `DATA_DICTIONARY.md`); if a lower-level synthetic transaction stream is used internally
 to build merchant-week aggregates, it must follow the same synthetic-token rule.

 Merchant appeal evidence is a free-text field plus evidence filename references. No
 real file upload pipeline exists or is planned for Phase 1 -- Phase 1's evidence
 references are simulated strings only (e.g. `invoice_demo_001.pdf`).

 **Phase 2 adds a real file upload pipeline** (implemented -- see
 `docs/PHASE_2_EVIDENCE_ATTACHMENTS_DESIGN.md`): local filesystem storage only
 (`data/evidence_attachments/`, gitignored), server-generated filenames (the
 client's filename is display-only, never used in a filesystem path),
 extension allowlist (`pdf`/`txt`/`png`/`jpg`/`jpeg`), a 5 MB size cap enforced
 at both the route and service layers, and a magic-byte content check so a
 renamed executable cannot pass as an allowed type just because of its
 filename. No malware/antivirus scanning exists -- an explicit, documented
 limitation of this local demo, not an oversight.

 ### Simulated evidence restrictions (implemented, Milestone 5)

 `app/schemas/evidence.py` validates every evidence reference string against a strict
 safe-pattern allowlist before it can be persisted:

 - Maximum 5 references per submission, 100 characters each.
 - Must match a safe demo filename/identifier pattern only (letters, digits,
   underscore, hyphen, optional short extension).
 - Rejected outright: path traversal (`../`), any `://` scheme (real URLs), shell
   metacharacters (`;`, `|`, `&`, `` ` ``, `$`), path separators (`/`, `\`), and blank
   strings.
 - No file upload, external URL retrieval, or document storage exists anywhere in this
   codebase — evidence is a list of validated strings only.

 ## MVP controls

 - Local SQLite database only (implemented, Milestone 5). A single local file
   (`clearrisk_recover.db` by default, configurable via the `DATABASE_URL` environment
   variable) accessed through SQLAlchemy. No network-exposed database service, no
   remote connection string is supported beyond what SQLAlchemy itself allows, and no
   multi-tenant isolation exists — this is a single local demo database.
 - Input schemas with Pydantic validation.
 - Configuration in environment variables.
 - No hard-coded secrets.
 - Sanitised audit payloads (implemented, Milestone 5): `app/services/audit_service.py`
   rejects any audit event payload containing `label_high_loss_next_30d`,
   `latent_state_for_demo_only`, or the enforcement terms `freeze`, `ban`, `terminate`,
   `hold settlement`, `reject payment` — the write is refused, not silently stripped.
 - Error messages must not leak internal stack traces in the UI.
 - Dependency versions pinned in requirements file where practical.
 - Confirmed (Milestone 5): no real PII, no real payment credentials, and no real
   financial/enforcement action (freeze, hold, ban, terminate, reject payment) exists in
   any database table, service, or script in this codebase. All persisted case,
   evidence, and audit records use only synthetic merchant tokens and simulated data.

 ## Local-demo authentication (implemented, Phase 2)

 `app/services/auth_service.py` adds local login and three roles (`reviewer`,
 `merchant`, `risk_manager`) -- see `docs/PHASE_2_AUTH_DESIGN.md` and
 `docs/MILESTONE_9_AUTH.md` for the full design and as-built report. This is
 **not** production-grade auth:

 - Password hashing: stdlib `hashlib.pbkdf2_hmac` (260,000 iterations),
   adequate for this local single-operator prototype, not a claim of
   production password-security compliance.
 - No MFA, no password reset flow, no login rate limiting/lockout, no
   OAuth/SSO, no external identity provider.
 - Sessions are opaque server-side tokens (not JWT) with a fixed 12-hour
   expiry -- a local demo session, not a production refresh-token scheme.
 - Accounts are seeded, fixed demo identities
   (`scripts/seed_demo_users.py`) -- not a real user-registration system.
 - Enforced at the FastAPI layer (every write endpoint requires a valid
   session; the actor recorded in the audit log is derived from the
   session, never from client-supplied input) -- real access control for
   this prototype, not decoration, but still local-only.
 - `POST /auth/login` is rate limited: 5 attempts/minute per client IP,
   in-memory, single-process (`app/services/rate_limit.py`) -- resets on
   restart and is not shared across workers/replicas, so this is a
   local-demo mitigation, not a production rate-limiting solution.
 - Every response carries `X-Content-Type-Options: nosniff` and
   `X-Frame-Options: DENY`; all routes except `/docs`/`/redoc`/`/openapi.json`
   additionally carry `Content-Security-Policy: default-src 'self'` (those
   three are exempted because Swagger UI loads its JS/CSS from a public CDN).
   `Strict-Transport-Security` is deliberately omitted -- this API serves
   plain HTTP on localhost, where HSTS would be meaningless rather than
   protective.

 ## Optional external dependency: Vertex AI evidence analysis

 `dashboard/components/case_detail.py` can optionally call Google Vertex AI
 (Gemini 1.5 Flash) to summarize a merchant's submitted evidence for a
 reviewer. This is the one genuine external network dependency in the
 dashboard, and it is deliberately narrow and fail-safe:

 - **Off by default.** The call only fires when both `GCP_PROJECT_ID` and
   `GCP_CREDENTIALS_JSON` are set in the environment. Neither is required to
   run the dashboard, and neither is committed anywhere in this repository.
 - **Never presented as more than it is.** If the credentials are absent, or
   the live call fails for any reason (bad credentials, quota, network),
   the page shows a visibly labeled illustrative example instead --
   `st.caption("⚠️ Illustrative example only -- ...")` -- never silently
   substituted for real output. The two states are never visually
   indistinguishable to a reviewer; see
   `tests/test_dashboard_safety.py::test_vertex_ai_call_has_a_labeled_fallback`.
 - **Failures are logged, not swallowed.** The real exception is written to
   the server-side log (`_logger.warning(...)`) even when the UI shows only
   the safe fallback message -- so a real credential/quota problem is
   debuggable, not invisible.
 - **Credential handling:** `GCP_CREDENTIALS_JSON` is a full GCP service
   account key. Treat it exactly like any other production secret -- set it
   only as a platform environment variable (e.g. a Render env var), never in
   `.env`, never committed, and scope the service account to the minimum
   Vertex AI permission needed. If you deploy this dashboard publicly with
   this feature enabled, that key is live infrastructure credential
   material, not demo data, even though everything it summarizes is
   synthetic.

 Separately, `dashboard/components/reviewer_actions.py` displays an
 illustrative example of a webhook payload after a reviewer action. This is
 static display text only -- no network request is ever made for it, the
 example target is `webhooks.example.org` (RFC 2606, reserved for
 documentation and guaranteed never to resolve to a real service), and the
 UI explicitly labels it "🎭 Simulated for demonstration only." See
 `tests/test_dashboard_safety.py::test_simulated_webhook_never_targets_a_real_domain`.

 ## Not implemented
 - Production-grade authentication hardening (MFA, password reset,
   distributed/production-grade rate limiting, external identity provider --
   see above for what's already implemented at local-demo grade).
 - Encryption at rest/in transit configuration.
 - Key management.
 - Secure secrets vault.
 - Tenant isolation.
 - Malware/antivirus scanning for uploaded evidence attachments (the upload
   pipeline itself is implemented at local-demo grade -- see above; scanning
   uploaded content is the specific gap).
 - Penetration testing.
 - Regulatory compliance certification.
 - Incident response/on-call operations.

 ## Audit-log limitation

 The MVP audit log is append-only at the **application layer only** (implemented,
 Milestone 5: `app/services/audit_service.py` and `app/db/repositories.py` expose only
 a create function and read functions for `AuditEvent` — no update or delete method
 exists in either module, verified by `tests/test_audit_service.py`). This is **not**
 a blockchain, WORM (write-once-read-many) storage, or any other cryptographically
 immutable ledger. A person with direct file-level access to the SQLite database file
 could still alter or delete audit rows outside this application's code path. Producing
 genuine tamper-evidence (e.g. a hash chain, external ledger, or WORM-backed store)
 is out of scope for this MVP.

 ## Secure development checks

 Before any demo/release:

 1. Search repository for prohibited fields and realistic credential patterns.
 2. Confirm `.env` is gitignored.
 3. Confirm demo data is synthetic.
 4. Run tests.
 5. Confirm UI labels state “Synthetic data / demonstration only.”
 6. Confirm no endpoint initiates payments or external financial actions.


