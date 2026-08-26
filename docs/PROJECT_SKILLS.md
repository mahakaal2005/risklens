# Project Skills — AI Risk Manager

Curated from the full set of globally available skills. Most global skills
(marketing, ads, SEO, growth, trading, IoT, agent-swarm orchestration, etc.)
do not apply to this project and are omitted. This list is a working
reference for which skills to invoke, and when, as the MVP moves through its
build phases.

## Phase 1 — data, validation, rules (current)

- **test-gaps** — Audit whether `tests/` actually covers the edge cases
  around the four risk scenarios (boundary thresholds, missing fields,
  combined-rule cases), not just the happy path. Run after each new rule or
  validation branch is added.
- **security-review** / **security-audit** — Run before any demo or release,
  matching the "Secure development checks" list in `SECURITY.md`: scan for
  prohibited fields, confirm `.env` is gitignored, confirm demo data is
  synthetic, confirm no endpoint could initiate a payment.
- **pii-detect** — Scan `demo_data/` and any sample payloads for accidental
  PII or realistic-looking credentials before committing, on top of the
  `assert_no_prohibited_fields` check already implemented in
  `ml/data_validation.py`.
- **dependency-check** — Vet any new Python dependency (license, maintenance,
  security advisories) before adding it to `requirements.txt`, per the
  "Mandatory quality check before adopting anything" section of
  `docs/OPEN_SOURCE_FOUNDATIONS.md.txt`.

## Phase 2 — ML baseline and evaluation

- **dataviz** — Use for the held-out-test metrics views (precision, recall,
  PR-AUC, confusion matrix, false-positive rate) so charts read as one
  consistent, accessible system instead of ad hoc plotting.

## Phase 3 — API, persistence, dashboard

- **api-docs** — Generate/refresh API documentation for the FastAPI scoring
  and case-review endpoints once `app/` is implemented.
- **ddd-aggregate** / **ddd-context** — Model the core entities (`Merchant`,
  `Transaction`, `ReviewCase`, `AuditEvent`, etc. from `ARCHITECTURE.md`
  Section 5) as clear bounded aggregates so the "modular monolith" keeps
  policy, scoring, case, and audit services cleanly separated instead of
  tangled together.
- **adr-create** / **adr-index** — Record short architecture decision
  records for choices like "SQLite over Postgres for the MVP" or "rules
  evaluated before the ML model, never after" so the reasoning in
  `ARCHITECTURE.md` / `RESEARCH.md` stays traceable as the code evolves.

## Cross-cutting, all phases

- **run** — Launch and smoke-test the Streamlit/FastAPI app locally before
  claiming a feature works, matching `CLAUDE.md`'s build-verification rule:
  run it locally where possible and report what was actually executed.
- **git-workflow** — Consistent commit hygiene as the repo grows past the
  single-commit stage.
- **simplify** — Periodic pass to remove duplication or premature
  abstraction without changing behavior, matching the "prefer the simplest
  architecture that demonstrates the value" principle in `CLAUDE.md`.

## Explicitly not used

Skills covering marketing, ads, copywriting, SEO, growth experimentation,
trading/finance-market prediction, IoT, browser-automation-for-scraping,
multi-agent swarm orchestration frameworks, and similar domains are not
relevant to a local synthetic-data payment-risk review prototype and are
intentionally left out of this list.
