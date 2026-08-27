  # Architecture — ClearRisk Recover MVP

  Status: Design proposal, rescoped 2026-08-22 to a single flagship loss class (merchant refund/chargeback loss risk) evaluated at merchant-week granularity. See `docs/RESCOPE_REVIEW.md`.

  ## 1. Architecture objective

  Provide an explainable, local risk-review workflow for simulated merchant-week refund/chargeback loss risk. The system predicts, explains, and routes for human review — it never decides or enforces on its own.

  ## 2. High-level architecture

                       OFFLINE / BUILD PIPELINE
┌─────────────────────────────────────────────────────────┐
│ Synthetic data generator                                 │
│ - 5-state latent merchant simulation → merchant-week rows │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Data validation + feature engineering                    │
│ - schema checks, weekly aggregates, quality checks        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Model training and evaluation                            │
│ - time-based train / validation / held-out test split    │
│ - rules-only baseline, then Logistic Regression baseline │
│ - precision, recall, PR-AUC, false-positive rate          │
│ - near-perfect result → investigate, do not report as-is │
└──────────────────────────┬──────────────────────────────┘
                           ▼
                 ┌──────────────────────┐
                 │ Versioned model      │
                 │ artifact + metrics   │
                 └──────────┬───────────┘
                            │
                            ▼

                       DEMO / APP FLOW
┌─────────────────────────────────────────────────────────┐
│ Merchant-week record                                      │
│ - synthetic input from CSV, form, or seeded SQLite        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Input validation + feature service                        │
│ - refund/chargeback trend, volume trend, evidence, support│
└──────────────────────────┬──────────────────────────────┘
                           ▼
              ┌────────────┴─────────────┐
              ▼                          ▼
┌───────────────────────────┐  ┌──────────────────────────┐
│ Rules engine               │  │ ML scoring service        │
│ - YAML risk policies       │  │ - model probability       │
│ - rule triggers/severity   │  │ - top contributing factors│
└──────────────┬────────────┘  └────────────┬─────────────┘
               └──────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│ Decision + explanation service                            │
│ - risk score and tier                                      │
│ - safe reason categories with before/after trend values     │
│ - recommendation only, never direct enforcement             │
└──────────────────────────┬──────────────────────────────┘
                           ▼
                  ┌────────┴─────────┐
                  ▼                  ▼
┌─────────────────────────┐  ┌────────────────────────────┐
│ Low-risk outcome         │  │ Medium/high-risk case       │
│ Approve or monitor       │  │ Create review case          │
└──────────────┬──────────┘  └────────────┬───────────────┘
               │                          ▼
               │              ┌───────────────────────────┐
               │              │ Human reviewer             │
               │              │ - clear case               │
               │              │ - request evidence         │
               │              │ - mark false positive       │
               │              │ - mark operational issue    │
               │              │ - escalate to compliance     │
               │              │ - mark inconclusive          │
               │              └────────────┬──────────────┘
               │                           ▼
               │              ┌───────────────────────────┐
               │              │ Merchant appeal simulation │
               │              │ - evidence text / fake refs │
               │              └────────────┬──────────────┘
               └───────────────────────────┴──────────────┐
                                                            ▼
┌─────────────────────────────────────────────────────────┐
│ SQLite persistence + append-only application audit log    │
│ - events, scores, rules, cases, notes, appeals, outcomes  │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ FastAPI + Streamlit dashboard                              │
│ - merchant-week feed - risk detail - review queue           │
│ - appeal view - audit timeline - metric dashboard            │
└─────────────────────────────────────────────────────────┘

3. Component responsibilities

| Component | Responsibility | Phase 1 status | Phase 2/3 need |
|---|---|---|---|
| Synthetic data generator | Creates labelled, non-identifying merchant-week demo data from a 5-state latent simulation | Planned for Phase 1 MVP | Replace/augment only with authorized, de-identified data |
| Feature builder | Produces refund-rate trend, chargeback-rate trend, volume trend, delivery-evidence coverage, support-ticket trend features per merchant-week | Planned for Phase 1 MVP | Extend for gateway/webhook-shaped event fields; add drift monitoring (Evidently) |
| Rules engine | Evaluates documented refund/chargeback-trend rules | Planned for Phase 1 MVP | Policy management, approvals, version governance |
| ML model (Logistic Regression) | Produces interpretable `label_high_loss_next_30d` probability | Planned for Phase 1 MVP | Model validation, monitoring, champion/challenger models |
| Explanation layer | Converts signals into safe explanations with concrete before/after trend values | Planned for Phase 1 MVP | Extend explanation storage for evidence-attachment context; no change to safety rules |
| Policy engine | Selects a recommendation, never enforcement | Planned for Phase 1 MVP | **Review SLA and notification simulation: implemented, Phase 2** (`app/services/sla_service.py` -- computed at read time, no scheduler; see docs/PHASE_2_REVIEW_SLA_DESIGN.md); approval workflow remains unbuilt |
| Auth service | Local login, session tokens, 3 roles (`reviewer`/`merchant`/`risk_manager`) | Out of scope (Phase 1) | **Implemented, Phase 2** (`app/services/auth_service.py`) — local-demo grade only (see SECURITY.md); real identity provider / MFA / production hardening remains Phase 3 |
| Case service | Creates/retrieves/updates review cases (one per flagged merchant-week) | Planned for Phase 1 MVP | **Authentication and basic roles: implemented, Phase 2** (actor derived from session, not client input; merchant-role reads filtered to own `merchant_id`); real reviewer queues and escalation (Phase 3) |
| Appeals service | Stores simulated merchant evidence and appeal note | Planned for Phase 1 MVP | **Evidence attachments: implemented, Phase 2** (`app/services/evidence_attachment_service.py` -- local filesystem, extension allowlist, magic-byte content check, 5MB cap, no malware scanning); secure/cloud evidence storage (Phase 3) |
| Audit service | Appends event history | Planned for Phase 1 MVP | Tamper-evident retention and access controls |
| Dashboard | Displays cases, explanation, outcomes, metrics, rules-only vs. rules+ML comparison | Planned for Phase 1 MVP | **Authentication/roles: implemented, Phase 2** (login gate, role-based page visibility); formal compliance/privacy reporting (Phase 3) |
| Gateway connector | Would ingest real gateway events | Out of scope | Requires partnership and approved API/data contract |
| Real enforcement service | Holds/funds restrictions/termination | Prohibited | Human-authorized gateway workflow only |

4. Data flow

 1. Synthetic merchant-week record is generated or loaded.
 2. Input validator checks schema and field ranges.
 3. Feature builder calculates weekly trend features (refund/chargeback rate change, volume change, evidence coverage, support-ticket trend).
 4. Rules engine evaluates transparent refund/chargeback-trend rules.
 5. Model service returns `label_high_loss_next_30d` probability and top contributing factors, unless disabled.
 6. Policy engine combines rule severity and model band into an action recommendation.
 7. Explanation service creates analyst and merchant-safe explanation variants, each with concrete before/after trend values.
 8. Case service creates a case for elevated-risk recommendations.
 9. Audit service appends each event.
10. Dashboard/API exposes the case for review, appeal, and outcome capture.
11. Resolved outcomes become feedback records for analytics, not automatic retraining.

5. Core entities

  Merchant
  MerchantWeek
  RiskAssessment
  RuleEvaluation
  ReviewCase
  EvidenceSubmission
  ReviewerDecision
  AuditEvent
  ModelRun

Note: `Transaction` and `CustomerToken` are no longer top-level scored/audited entities. Per-transaction records may still exist internally as a generation detail that rolls up into `MerchantWeek` aggregates (see `DATA_DICTIONARY.md`), but they are not validated, scored, or cased individually in Phase 1.

6. Risk-decision boundaries

The scoring service may generate a recommendation. It cannot execute monetary or account-level action.

  Scoring outcome → Recommendation → Human review → External system decision (outside MVP boundary)

Allowed system recommendations:

    APPROVE
    ALLOW_WITH_MONITORING
    REQUEST_EVIDENCE
    MANUAL_REVIEW_REQUIRED
    ESCALATE_TO_COMPLIANCE

Resolved (approved 2026-08-22, see `docs/RESCOPE_REVIEW.md` Section 8): `STEP_UP_VERIFICATION_RECOMMENDED` is permanently removed from the Phase 1 enum — it was tied to the now-removed device/dispute scenario and does not apply to a merchant-week refund/chargeback loss decision.

7. Failure handling

| Failure | Behaviour |
|---|---|
| Invalid merchant-week schema | Reject request with clear validation error; write validation audit event where appropriate |
| Model artifact unavailable | Rules-only fallback; record fallback event |
| Database unavailable | Return safe error; do not claim a decision was saved |
| Explanation service error | Return reason category and rule summary; mark explanation as degraded |
| Unknown merchant | Create no risk decision unless demo policy allows a synthetic default; label assumption |
| Incomplete label horizon (trailing weeks) | Exclude from labeled train/validation/test data rather than defaulting to label = 0 |

8. Security boundaries

    Local-only development prototype.
    Synthetic data only.
    No real payment credentials, banking credentials, or payment initiation.
    Use fake merchant tokens instead of real identity.
    Environment variables for any configuration; no secrets in repository.
    SQLite is for demo persistence only.
    Application audit log is append-only by application design but not cryptographically immutable.

9. Production requirements not included

A real deployment would require confirmed gateway integrations, data agreements, identity/access management, encryption and key management, network segmentation, monitoring/alerting, secure evidence storage, privacy retention policies, model governance, independent validation, red-team testing, incident response, legal review, and authorized human enforcement operations. See the Phase 2/Phase 3 roadmap for the staged path toward these (not part of the Phase 1 pitch or demo).
