# Phase 2 Design — Review SLA and Notification Simulation (as-built)

**Status: implemented, on branch `phase-2-auth-design`.** Per `CLAUDE.md`'s Phase 2 roadmap: "Add configurable review SLA and notification simulation." Confirmed with the user before implementation: SLA thresholds and whether they should be a config file vs. a code constant.

## 1. Goal

Give reviewers and risk managers a visible signal for cases that are overdue for action, without adding any new infrastructure (no background scheduler, no message queue, no real notification channel) — consistent with this project's "prefer the simplest architecture" principle and its explicit exclusion of Kafka/real-time streaming infrastructure.

## 2. Decisions (confirmed with the user)

1. **SLA thresholds, by recommendation severity**:
   | Recommendation | SLA window |
   |---|---|
   | `ESCALATE_TO_COMPLIANCE` | 24 hours |
   | `MANUAL_REVIEW_REQUIRED` | 48 hours |
   | `REQUEST_EVIDENCE` | 72 hours |
   | `ALLOW_WITH_MONITORING`, `APPROVE` | Not applicable (no SLA) |
2. **Configuration method: a code constant** (`SLA_HOURS_BY_RECOMMENDATION` in `app/services/sla_service.py`), not a new YAML file — a 3-value lookup table didn't justify a new config-file parsing path alongside `rules/risk_rules.yaml`.

## 3. Design: computed, not stored

The SLA deadline and breach status are **computed at read time** from fields the `ReviewCase` row already has (`recommendation`, `created_at`, `case_status`) — no new database column, no scheduled job checking for breaches, no push notification. This means:

- The SLA clock starts at case creation and stops the moment a case reaches `RESOLVED` or `ESCALATED` (a case that's been handled or handed off for compliance follow-up is no longer "at risk" of breaching, even if it took a while to get there — its `sla_hours` is still reported for the record, but `sla_breached` is always `False` and `hours_until_deadline` is `None` once terminal).
- Changing the threshold table takes effect immediately for every existing case on the next read — there is nothing to backfill or migrate.
- The tradeoff: there's no historical record of exactly when a case *became* breached, only whether it currently is, right now. Documented as a known limitation (Section 6).

## 4. "Notification simulation"

CLAUDE.md's Phase 2 roadmap uses the word "notification simulation." This codebase has no email/SMS/webhook integration anywhere (nor is one planned for Phase 1/2 per `ARCHITECTURE.md`), so the honest scope here is: a **simulated in-app breach indicator** — a warning banner in the Review Queue summary, a per-row SLA column, and a warning banner on a breached case's detail page — each explicitly labeled "simulated in-app indicator only — no real email/SMS notification exists." No external channel is implied or built.

## 5. API and dashboard surface

- `GET /cases` (`CaseSummary`) and `GET /cases/{case_id}` (`CaseDetailResponse`) both gained: `sla_hours` (int|null), `sla_deadline` (datetime|null), `hours_until_deadline` (float|null, negative once overdue), `sla_breached` (bool).
- Review Queue: a 5th summary metric ("SLA breached"), an `SLA` column in the case table (`"71.8h left"` / `"⏰ Overdue by 4.2h"` / `"Closed"` / `"N/A"`), and a warning banner when any breached case exists.
- Case Detail: a caption (on-track) or warning banner (breached) under the case header, using the same shared `dashboard/components/common.py::sla_display()` helper as Review Queue so the two pages can't drift out of sync in how they describe the same state.

## 6. Known limitations (not fabricated, flagged instead)

- No historical breach log — only current status is computable, not "when did this first breach."
- No real notification channel (email/SMS/webhook) — explicitly simulated, in-app only.
- Thresholds are a code constant for v1, not editable without a code change (a deliberate scope decision, not an oversight — see Section 2).
- No per-merchant or per-analyst SLA customization; one global table by recommendation only.

## 7. Tests

- `tests/test_sla_service.py` (8 tests) — no-SLA recommendations, within-window, past-deadline, per-tier thresholds, terminal-status clock-stop behavior (both `RESOLVED` and `ESCALATED`), naive-datetime handling.
- `tests/test_api_cases.py` — 3 new tests: SLA fields present in list/detail responses, and a resolved case reports `sla_breached=False` regardless of age.
- Dashboard smoke-tested headlessly (`AppTest`): SLA column/banner render correctly for both a no-SLA case (`ALLOW_WITH_MONITORING`, shows nothing) and an SLA-applicable case (`REQUEST_EVIDENCE`, shows "71.8h left").
