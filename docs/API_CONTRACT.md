# API Contract — ClearRisk Recover

Status: Implemented, Milestone 6. **Every endpoint in this API is a local synthetic-data demonstration only.** There is no authentication, no real payment gateway connection, and no endpoint that freezes funds, holds settlement, bans a merchant, terminates an account, rejects a payment, processes a payment, issues a refund, or transfers funds — none of these exist anywhere in this codebase, verified by `tests/test_api_cases.py::test_no_route_or_response_contains_prohibited_enforcement_words`, which scans the live OpenAPI schema for these terms.

Base URL (local): `http://127.0.0.1:8000`. Interactive docs: `http://127.0.0.1:8000/docs`.

---

## GET /health

Returns service status. No parameters.

**Response 200:**
```json
{
  "status": "ok",
  "service": "clearrisk-recover",
  "environment": "local-demo",
  "data_mode": "synthetic-only",
  "payment_actions_enabled": false
}
```

## GET /cases

Paginated, filterable list of persisted review cases.

**Query parameters (all optional):**
| Param | Type | Constraint |
|---|---|---|
| `status` | string | one of `CaseStatus` |
| `recommendation` | string | one of the 5 allowed recommendations |
| `intensity` | string | `Low` / `Medium` / `High` |
| `limit` | int | default 50, min 1, max 100 (over-100 returns 422) |
| `offset` | int | default 0, min 0 |

**Response 200:**
```json
{
  "items": [
    {
      "case_id": "case_preview_51376c9cafa69de2",
      "merchant_id": "merchant_demo_0010",
      "week_start": "2025-11-17",
      "case_status": "OPEN",
      "risk_signal_intensity": "High",
      "recommendation": "MANUAL_REVIEW_REQUIRED",
      "created_at": "2026-08-23T17:41:33Z",
      "updated_at": "2026-08-23T17:41:33Z",
      "final_outcome": null,
      "synthetic_data_notice": "Local synthetic-data demonstration only."
    }
  ],
  "limit": 50,
  "offset": 0,
  "total": 4,
  "synthetic_data_notice": "Local synthetic-data demonstration only."
}
```
Sorted deterministically by `created_at` descending, then `case_id` ascending. Never includes `label_high_loss_next_30d`, `latent_state_for_demo_only`, `support_ticket_rate` (diagnostic-only), or any prohibited/enforcement field.

## GET /cases/{case_id}

Safe full case detail.

**Response 200:** case_id, merchant_id, week_start, case_status, risk_signal_intensity, recommendation, policy_explanation, analyst_summary, merchant_safe_explanation, triggered_rules, evidence_checklist, model_version, rules_version, created_at, updated_at, resolved_at, final_outcome, reviewer_note, evidence_submissions (summary only: evidence_id, submitted_at, status, evidence_references — no audit events here), synthetic_data_notice.

**Response 404** (unknown case_id):
```json
{"error": {"code": "CASE_NOT_FOUND", "message": "No review case exists for the provided case ID.", "synthetic_data_notice": "Local synthetic-data demonstration only."}}
```

## GET /cases/{case_id}/audit-events

Ordered audit timeline (ascending by `event_sequence_number`).

**Response 200:**
```json
{
  "case_id": "case_preview_51376c9cafa69de2",
  "events": [
    {"event_sequence_number": 1, "event_timestamp": "...", "actor_type": "system", "actor_id": "system", "event_type": "ASSESSMENT_GENERATED", "event_payload": {"...": "..."}}
  ],
  "synthetic_data_notice": "Local synthetic-data demonstration only."
}
```
Payloads are re-checked against the existing safety guard (`app/services/audit_service.py`'s payload-safety function) at read time, on top of the write-time check. **404** for unknown case_id.

## POST /cases/{case_id}/review-actions

**Request:**
```json
{"action": "REQUEST_EVIDENCE", "reviewer_actor_id": "analyst_demo_001", "reviewer_note": "Please provide fulfilment proof and refund records."}
```

Allowed `action` values: `CLEAR_CASE`, `MARK_FALSE_POSITIVE`, `REQUEST_EVIDENCE`, `MARK_OPERATIONAL_ISSUE`, `ESCALATE_CASE`, `MARK_INCONCLUSIVE`, `START_REVIEW`.

- Unknown/invalid `action` value → **422** (Pydantic enum validation, automatic).
- Blank `reviewer_actor_id` or `reviewer_note` → **422** `VALIDATION_ERROR`.
- Unknown `case_id` → **404** `CASE_NOT_FOUND`.
- Invalid state transition (e.g. `START_REVIEW` from `OPEN`) → **409** `INVALID_CASE_TRANSITION`, no mutation.
- Success → **200**, `{"case": <CaseDetailResponse>, "new_audit_events": [...], "synthetic_data_notice": "..."}`.

Every call is routed through `app/services/case_service.py::apply_reviewer_action()` or `start_review()` — no route ever mutates a repository directly.

## POST /cases/{case_id}/evidence

**Request:**
```json
{
  "merchant_actor_id": "merchant_demo_001",
  "merchant_explanation_text": "A seasonal sale increased returns. Refunds were processed within policy.",
  "evidence_references": ["refund_records_demo_001.pdf", "seasonal_sale_summary_demo_001.txt"]
}
```

- Only accepted while the case is in `EVIDENCE_REQUESTED` status; otherwise **409** `INVALID_CASE_TRANSITION`.
- Invalid evidence reference (path traversal, real URL, shell metacharacters, blank, >100 chars, >5 items) → **422** `INVALID_EVIDENCE_REFERENCE`.
- Blank `merchant_explanation_text` → **422** `INVALID_EVIDENCE_REFERENCE`.
- Unknown case_id → **404** `CASE_NOT_FOUND`.
- Success → **200**, `{"evidence_id": "...", "case_id": "...", "case_status": "EVIDENCE_SUBMITTED", "submitted_at": "...", "evidence_references": [...], "new_audit_event": {...}, "synthetic_data_notice": "..."}`.

No real file upload or external URL retrieval exists anywhere — evidence references are validated strings only.

## GET /metrics

Reads a previously saved evaluation-report artifact from `ml/artifacts/evaluation_report.json`. **Never retrains or re-scores during the request.**

**Response 200 (available):**
```json
{
  "status": "available",
  "data_mode": "synthetic-only",
  "message": "Evaluation report loaded from ml/artifacts/evaluation_report.json.",
  "dataset_seed": 42,
  "dataset_version": "0.1.0",
  "held_out_test_date_range": {"min": "2025-11-10", "max": "2025-12-29"},
  "selected_threshold": 0.1,
  "rules_only_metrics": {"precision": 0.269, "recall": 0.425, "pr_auc": 0.246},
  "logistic_regression_metrics": {"precision": 0.529, "recall": 0.863, "pr_auc": 0.653},
  "combined_policy_metrics": {"precision": 0.375, "recall": 0.881, "pr_auc": 0.653},
  "near_perfect_investigation_status": "APPROVED",
  "limitation": "Synthetic-data metrics demonstrate prototype workflow only and do not prove real-world chargeback-risk performance.",
  "synthetic_data_notice": "Local synthetic-data demonstration only."
}
```

**Response 200 (not available — Beginning in Milestone 7, the offline evaluation pipeline persists a validated synthetic evaluation report at ml/artifacts/latest_evaluation_report.json. GET /metrics reads this artifact without retraining or recomputing metrics at request time.):**
```json
{
  "status": "not_available",
  "message": "No saved evaluation-report artifact was found at ml/artifacts/evaluation_report.json. Run the local evaluation pipeline ... and persist its output to that path to populate this endpoint.",
  "synthetic_data_notice": "Local synthetic-data demonstration only."
}
```
Never returns a 500 for a missing artifact.

---

## Error response schema

```json
{"error": {"code": "CASE_NOT_FOUND", "message": "...", "synthetic_data_notice": "Local synthetic-data demonstration only."}}
```

| Code | HTTP status | When |
|---|---|---|
| `CASE_NOT_FOUND` | 404 | Unknown `case_id` |
| `INVALID_CASE_TRANSITION` | 409 | State machine rejects the requested action |
| `INVALID_EVIDENCE_REFERENCE` | 422 | Evidence reference or explanation text fails validation |
| `INVALID_REVIEW_ACTION` | — | Reserved; currently unknown actions are rejected at the Pydantic-enum layer as `VALIDATION_ERROR` (422) before reaching service code |
| `VALIDATION_ERROR` | 422 | Request body/query fails schema validation |
| `METRICS_NOT_AVAILABLE` | 200 | When the persisted evaluation report is unavailable, `GET /metrics` returns HTTP 200 with `status = not_available` and `error_code = METRICS_NOT_AVAILABLE`. The response includes the local generation command and a synthetic-data notice. |
| `METRICS_ARTIFACT_INVALID` | 200 | When the persisted evaluation report exists but is corrupt or fails schema/safety validation, `GET /metrics` returns HTTP 200 with `status = not_available` and `error_code = METRICS_ARTIFACT_INVALID`. The response includes the local generation command and a synthetic-data notice, and never the underlying parse error or file path. |
| `INTERNAL_SAFE_ERROR` | 500 | Unhandled exception — no stack trace is ever returned |

## Non-existent enforcement endpoints (explicit negative list)

None of the following exist, are planned, or are reachable through any route, alias, or query parameter in this API: `freeze`, `hold settlement`, `ban`, `terminate`, `reject payment`, `process payment`, `issue refund`, `transfer funds`. Verified continuously by `tests/test_api_cases.py::test_no_route_or_response_contains_prohibited_enforcement_words`, which scans the live OpenAPI schema.

## Enums

- **CaseStatus:** `OPEN`, `EVIDENCE_REQUESTED`, `EVIDENCE_SUBMITTED`, `UNDER_REVIEW`, `RESOLVED`, `ESCALATED`
- **FinalOutcome:** `CONFIRMED_RISK`, `FALSE_POSITIVE`, `OPERATIONAL_ISSUE`, `INCONCLUSIVE`
- **Recommendation:** `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`
- **ReviewActionAPI:** `CLEAR_CASE`, `MARK_FALSE_POSITIVE`, `REQUEST_EVIDENCE`, `MARK_OPERATIONAL_ISSUE`, `ESCALATE_CASE`, `MARK_INCONCLUSIVE`, `START_REVIEW`
- **RiskSignalIntensity:** `Low`, `Medium`, `High`
- **ActorType:** `system`, `analyst_demo`, `merchant_demo`
