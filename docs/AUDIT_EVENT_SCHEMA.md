# Audit Event Schema — ClearRisk Recover

Status: Implemented, Milestone 5. **The audit log is append-only at the application layer only.** SQLite provides no cryptographic immutability or WORM (write-once-read-many) guarantee — a person with direct database file access could still alter rows outside this application. See `SECURITY.md` for the full boundary statement.

Every audit event has these columns (`app/db/models.py::AuditEvent`): `audit_event_id`, `case_id`, `event_timestamp`, `actor_type`, `actor_id`, `event_type`, `event_payload_json`, `event_sequence_number`. `event_sequence_number` is contiguous per case, assigned by `app/db/repositories.py::get_next_sequence_number()`, and the timeline is always read ordered by sequence number then timestamp (`app/services/audit_service.py::get_case_timeline()`).

## Event type reference

| Event type | Actor type | Required payload fields | Created by | State transition effect |
|---|---|---|---|---|
| `ASSESSMENT_GENERATED` | system | `risk_signal_intensity`, `rules_only_score`, `model_probability` | System, at case creation | None (informational) |
| `EXPLANATION_GENERATED` | system | `summary` | System, at case creation | None |
| `REVIEW_CASE_CREATED` | system | `case_id`, `recommendation` | System, at case creation | Case row created, status `OPEN` |
| `REVIEW_CASE_RECOMMENDED` | system | `recommendation`, `policy_explanation` | System, at case creation | None |
| `EVIDENCE_REQUEST_RECOMMENDED` | system (at creation) or analyst_demo (reviewer action) | `evidence_checklist` (system) or `action`/`note`/`previous_status`/`new_status` (reviewer) | Both — reused for the system's initial suggestion and the reviewer's explicit `REQUEST_EVIDENCE` action; see note below | System use: none. Reviewer use: `OPEN` → `EVIDENCE_REQUESTED` |
| `MANUAL_REVIEW_RECOMMENDED` | system | none required | System, at case creation, if recommendation is `MANUAL_REVIEW_REQUIRED` | None |
| `EVIDENCE_SUBMITTED` | merchant_demo | `evidence_id`, `evidence_references`, `explanation_length` | Merchant (simulated), via `app/services/evidence_service.py` | `EVIDENCE_REQUESTED` → `EVIDENCE_SUBMITTED` |
| `REVIEW_STARTED` | analyst_demo | none required | Analyst, via `start_review()` | `EVIDENCE_SUBMITTED` → `UNDER_REVIEW` |
| `CASE_CLEARED` | analyst_demo | `action`, `note`, `previous_status`, `new_status` | Analyst, `CLEAR_CASE` action | → `RESOLVED`, `final_outcome=CONFIRMED_RISK` |
| `CASE_MARKED_FALSE_POSITIVE` | analyst_demo | same as above | Analyst, `MARK_FALSE_POSITIVE` action | → `RESOLVED`, `final_outcome=FALSE_POSITIVE` |
| `CASE_MARKED_OPERATIONAL_ISSUE` | analyst_demo | same as above | Analyst, `MARK_OPERATIONAL_ISSUE` action | → `RESOLVED`, `final_outcome=OPERATIONAL_ISSUE` |
| `CASE_MARKED_INCONCLUSIVE` | analyst_demo | same as above | Analyst, `MARK_INCONCLUSIVE` action | → `RESOLVED`, `final_outcome=INCONCLUSIVE` |
| `CASE_ESCALATED` | analyst_demo | `action`/`note`/`previous_status`/`new_status` (first escalation) or `note` only (subsequent escalation notes) | Analyst, `ESCALATE_CASE` action, or `add_escalation_note()` on an already-`ESCALATED` case | First use: → `ESCALATED`. Subsequent uses: none (case stays `ESCALATED`) |

**Note on `EVIDENCE_REQUEST_RECOMMENDED` reuse:** this event type is deliberately used twice with different `actor_type` values — once by the system at case creation (a suggestion, no state change) and once by the analyst when they actually act on `REQUEST_EVIDENCE` (a real state transition, `OPEN` → `EVIDENCE_REQUESTED`). This was a documented design choice to stay within the fixed required event-type list rather than inventing an additional type; the two uses are distinguishable by `actor_type` and by whether a state transition occurred.

## Safe-data restrictions (enforced by `app/services/audit_service.py::_assert_payload_is_safe()`)

Every payload is checked before being written, and the write is rejected (raising `UnsafeAuditPayloadError`) if it contains:

- `label_high_loss_next_30d` or `latent_state_for_demo_only` (as literal substrings, case-sensitive)
- Any of the forbidden enforcement terms: `freeze`, `ban`, `terminate`, `hold settlement`, `reject payment` (case-insensitive)

Payloads never include raw model coefficients, prohibited PII, or sensitive financial credentials — the case-service layer only ever passes safe, already-sanitized case-packet fields (analyst summary, recommendation, policy explanation, evidence checklist) or reviewer-supplied note text into a payload.

## Append-only enforcement

`app/services/audit_service.py` and `app/db/repositories.py` expose exactly two operations on `AuditEvent`: `record_event()` (create) and `get_case_timeline()` / `get_audit_events_for_case()` (read). Neither module defines an update or delete function for audit events — verified by `tests/test_audit_service.py::test_audit_service_exposes_no_update_or_delete_function`, which inspects the module's public functions directly.
