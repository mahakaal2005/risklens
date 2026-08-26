# Milestone 5 — SQLite Persistence, Case Workflow, Merchant Evidence Simulation, and Append-Only Audit Log

Status: Implemented and run end to end against real seeded demo data. **This remains a local synthetic-data demonstration — no real payment, settlement, hold, ban, or termination action exists anywhere in this codebase.**

---

## 1. Case entity diagram

```text
┌───────────────────────────────┐        ┌──────────────────────────────┐
│ ReviewCase                    │ 1    * │ EvidenceSubmission           │
│ case_id (PK)                  ├───────►│ evidence_id (PK)             │
│ merchant_id, week_start       │        │ case_id (FK)                 │
│ case_status                   │        │ submitted_by_actor_type      │
│ risk_signal_intensity         │        │ merchant_explanation_text    │
│ model_probability             │        │ evidence_references_json     │
│ selected_threshold             │        │ submitted_at                 │
│ rules_only_score               │        │ status                       │
│ recommendation                 │        └──────────────────────────────┘
│ policy_explanation             │
│ analyst_summary                 │        ┌──────────────────────────────┐
│ merchant_safe_explanation (JSON)│ 1    * │ AuditEvent                   │
│ triggered_rules_json            ├───────►│ audit_event_id (PK)          │
│ evidence_checklist_json         │        │ case_id (FK)                 │
│ model_version, rules_version    │        │ event_timestamp              │
│ synthetic_data_notice           │        │ actor_type, actor_id          │
│ created_at, updated_at          │        │ event_type                   │
│ resolved_at (nullable)          │        │ event_payload_json            │
│ final_outcome (nullable)        │        │ event_sequence_number         │
│ reviewer_note (nullable)        │        └──────────────────────────────┘
│ reviewer_actor (nullable)       │
│ decision_timestamp (nullable)   │
└───────────────────────────────┘
```

## 2. State machine diagram

```text
        REQUEST_EVIDENCE
   ┌───────────────────────► EVIDENCE_REQUESTED ──────┐
   │                              │                     │ merchant
   │                              │ ESCALATE_CASE       │ evidence
   │                              ▼                     │ submission
   │                          ESCALATED ◄────────────┐  ▼
 OPEN                                                │  EVIDENCE_SUBMITTED
   │                                                  │       │
   │ CLEAR_CASE / MARK_FALSE_POSITIVE /               │       │ reviewer
   │ MARK_OPERATIONAL_ISSUE / MARK_INCONCLUSIVE        │       │ begins review
   │                                                   │       ▼
   └──────────────────────────────────────────────────┼──► UNDER_REVIEW
                                                        │       │
                                                        │       │ CLEAR_CASE / MARK_FALSE_POSITIVE /
                                                        │       │ MARK_OPERATIONAL_ISSUE / MARK_INCONCLUSIVE /
                                                        │       │ ESCALATE_CASE
                                                        │       ▼
                                                        └── RESOLVED (immutable) / ESCALATED
```

`RESOLVED` has no outgoing edges — enforced in code (`app/services/case_service.py::apply_reviewer_action` raises `InvalidTransitionError` if `case_status == "RESOLVED"` before touching anything else). `ESCALATED` permits no further status-changing action — only `add_escalation_note()` may append an additional note-only audit event.

## 3. Action / transition table

| From status | Action | To status | Final outcome set? |
|---|---|---|---|
| `OPEN` | `REQUEST_EVIDENCE` | `EVIDENCE_REQUESTED` | No |
| `OPEN` | `CLEAR_CASE` | `RESOLVED` | `CONFIRMED_RISK` |
| `OPEN` | `MARK_FALSE_POSITIVE` | `RESOLVED` | `FALSE_POSITIVE` |
| `OPEN` | `MARK_OPERATIONAL_ISSUE` | `RESOLVED` | `OPERATIONAL_ISSUE` |
| `OPEN` | `MARK_INCONCLUSIVE` | `RESOLVED` | `INCONCLUSIVE` |
| `OPEN` | `ESCALATE_CASE` | `ESCALATED` | No |
| `EVIDENCE_REQUESTED` | (merchant evidence submission) | `EVIDENCE_SUBMITTED` | No |
| `EVIDENCE_REQUESTED` | `ESCALATE_CASE` | `ESCALATED` | No |
| `EVIDENCE_SUBMITTED` | (reviewer begins review) | `UNDER_REVIEW` | No |
| `UNDER_REVIEW` | `CLEAR_CASE` / `MARK_FALSE_POSITIVE` / `MARK_OPERATIONAL_ISSUE` / `MARK_INCONCLUSIVE` | `RESOLVED` | Matching outcome |
| `UNDER_REVIEW` | `ESCALATE_CASE` | `ESCALATED` | No |
| `ESCALATED` | (add escalation note only) | `ESCALATED` (unchanged) | No |
| `RESOLVED` | (any action) | **rejected — immutable** | — |

**Action-to-outcome mapping design note:** `CLEAR_CASE` maps to `CONFIRMED_RISK`. This reads as "clear this case [for confirmed-risk handling]," matching the pre-existing outcome vocabulary in `RISK_POLICY.md` (`CONFIRMED_RISK`/`FALSE_POSITIVE`/`INCONCLUSIVE`/`OPERATIONAL_ISSUE`), where reviewer *actions* are verbs and *outcomes* are nouns. This mapping was not fully spelled out in the Milestone 5 instructions and is documented here as an explicit design decision.

Reviewer actions never map to `BANNED`, `FROZEN`, `TERMINATED`, `PAYMENT_REJECTED`, or `SETTLEMENT_HELD` — those values do not exist anywhere in this codebase (verified by `tests/test_case_service.py::test_prohibited_outcomes_are_never_reachable`).

## 4. Full seasonal-sale false-positive walkthrough

Run via `python3 scripts/demo_case_workflow.py` (after `python3 scripts/seed_demo_cases.py`):

1. Load seeded case `case_preview_b84e09e48542385c` (from the Milestone 4 `seasonal_sale_false_positive_candidate` packet), status `OPEN`.
2. Reviewer requests evidence: `apply_reviewer_action(..., "REQUEST_EVIDENCE", "Please share refund records and an explanation for the recent volume/refund increase.")` → status `EVIDENCE_REQUESTED`.
3. Merchant submits simulated evidence: explanation text plus `["invoice_demo_001.pdf", "refund_policy_demo_url"]` → status `EVIDENCE_SUBMITTED`.
4. Reviewer begins review: `start_review(...)` → status `UNDER_REVIEW`.
5. Reviewer marks `MARK_FALSE_POSITIVE` with a note confirming the refund spike is a legitimate seasonal sale → status `RESOLVED`, `final_outcome = FALSE_POSITIVE`.

**Final state:** `status=RESOLVED`, `final_outcome=FALSE_POSITIVE`, `reviewer_note="Refund spike is explained by a legitimate seasonal sale; chargeback rate stayed normal. No further concern."`

**Complete ordered audit timeline (actual run output):**

```text
[1] ASSESSMENT_GENERATED       (actor=system:system)
[2] EXPLANATION_GENERATED      (actor=system:system)
[3] REVIEW_CASE_CREATED        (actor=system:system)
[4] REVIEW_CASE_RECOMMENDED    (actor=system:system)
[5] EVIDENCE_REQUEST_RECOMMENDED (actor=system:system)       -- system's initial suggestion
[6] EVIDENCE_REQUEST_RECOMMENDED (actor=analyst_demo:analyst_demo) -- reviewer's actual action
[7] EVIDENCE_SUBMITTED         (actor=merchant_demo:merchant_demo)
[8] REVIEW_STARTED             (actor=analyst_demo:analyst_demo)
[9] CASE_MARKED_FALSE_POSITIVE (actor=analyst_demo:analyst_demo)
```

## 5. Full operational-issue walkthrough

1. Load seeded case `case_preview_f1b93f3619c22b3e` (Milestone 4 `operational_fulfilment_problem` packet), status `OPEN`.
2. Reviewer requests evidence (fulfilment/delivery proof and support records).
3. Merchant submits `["delivery_proof_demo_001.pdf", "support_log_summary_demo"]` plus an explanation citing a warehouse migration → `EVIDENCE_SUBMITTED`.
4. Reviewer begins review → `UNDER_REVIEW`.
5. Reviewer marks `MARK_OPERATIONAL_ISSUE` with a note confirming a temporary fulfilment issue, not fraud → `RESOLVED`, `final_outcome = OPERATIONAL_ISSUE`.

Audit timeline follows the identical 9-event shape as the seasonal-sale case above, ending in `CASE_MARKED_OPERATIONAL_ISSUE` instead.

## 6. Safety boundaries

- No FastAPI, Streamlit, authentication, real file upload, email/SMS/WhatsApp notification, external gateway integration, or automatic retraining exists in this milestone.
- `create_case_from_packet()` refuses to persist a packet if `label_high_loss_next_30d` or `latent_state_for_demo_only` appear anywhere in its serialized JSON (`_assert_packet_has_no_leakage`), on top of `ml/case_packet.py`'s own Milestone 4 guard.
- Every audit payload is checked for the same two leakage terms plus the five forbidden enforcement terms (`freeze`, `ban`, `terminate`, `hold settlement`, `reject payment`) before it is written — `app/services/audit_service.py::_assert_payload_is_safe()` raises rather than silently stripping.
- Evidence references are validated against a strict safe-pattern allowlist (`app/schemas/evidence.py`) — no path traversal, shell metacharacters, or real URLs are accepted; maximum 5 references, 100 characters each.
- Reviewer actions always require a non-empty note, validated before any database mutation.
- Every allowed final outcome, recommendation, and reviewer action is drawn from a fixed, tested enum — no enforcement value (ban/freeze/terminate/reject/hold) exists anywhere in the type system, not just at runtime.

## 7. Append-only limitations

The audit log is append-only **at the application layer only** — enforced by `app/services/audit_service.py` and `app/db/repositories.py` exposing no update or delete function for `AuditEvent`. This is **not** cryptographic immutability, a hash chain, or WORM (write-once-read-many) storage: a person with direct file-level access to the SQLite database could still alter or delete rows outside this application's code path. See `SECURITY.md` for the full statement. Producing genuine tamper-evidence (e.g. a hash chain or external ledger) is out of scope for this MVP and would belong to a later phase, if ever required.
