# Data Dictionary — Synthetic Data Only

Status: Design proposal, rescoped 2026-08-22. The primary scored/audited entity is now `merchant_week` (one row per merchant per week), replacing the earlier per-transaction dataset as the flagship entity. See `docs/RESCOPE_REVIEW.md`.

## Data status

All data used by this MVP is synthetic and non-identifying. Field names are inspired by real payment-risk concepts but no field contains real customer, merchant, or payment data.

## Merchant-week dataset (primary scored entity)

| Field | Type | Example | Description | Sensitivity |
|---|---|---|---|---|
| merchant_id | string | merchant_demo_0001 | Synthetic merchant token | Low |
| week_start | date | 2026-01-05 | Monday-aligned start of the merchant-week | Low |
| merchant_category | enum | apparel/electronics/grocery/travel/digital_services/food_delivery | Simulated onboarding category | Medium |
| merchant_age_days | integer | 185 | Synthetic age since onboarding, as of `week_start` | Low |
| transaction_count_30d | integer | 420 | Simulated trailing-30-day transaction count | Low |
| transaction_volume_30d | decimal | 840000.00 | Simulated trailing-30-day payment volume (INR) | Medium |
| transaction_volume_previous_30d | decimal | 310000.00 | Simulated prior 30-day payment volume (INR) | Medium |
| transaction_volume_change_30d | float | 1.71 | Derived: `(volume_30d - volume_previous_30d) / volume_previous_30d` | Low |
| refund_count_30d | integer | 24 | Simulated trailing-30-day refund count | Medium |
| refund_rate_30d | float | 0.058 | Simulated trailing-30-day refund rate | Medium |
| refund_rate_previous_30d | float | 0.011 | Simulated prior 30-day refund rate | Medium |
| refund_rate_change_30d | float | 0.047 | Derived: `refund_rate_30d - refund_rate_previous_30d` | Low |
| chargeback_count_30d | integer | 9 | Simulated trailing-30-day chargeback count | Medium |
| chargeback_rate_30d | float | 0.022 | Simulated trailing-30-day chargeback rate | Medium |
| chargeback_rate_previous_30d | float | 0.004 | Simulated prior 30-day chargeback rate | Medium |
| chargeback_rate_change_30d | float | 0.018 | Derived: `chargeback_rate_30d - chargeback_rate_previous_30d` | Low |
| top_dispute_reason_category | enum | item_not_as_described/not_received/duplicate_charge/quality_issue/other | Simulated dominant dispute reason for the week | Medium |
| delivery_evidence_coverage | float | 0.62 | Simulated fraction of disputed transactions with adequate delivery/fulfilment evidence on file | Medium |
| support_ticket_rate | float | 0.031 | Simulated support tickets per transaction | Medium |
| average_support_resolution_time_hours | float | 36.5 | Simulated average time to resolve a support ticket | Medium |
| previous_review_outcome | enum | none/confirmed_risk/false_positive/inconclusive/operational_issue | Most recent prior reviewer outcome for this merchant, if any | Medium |
| label_high_loss_next_30d | boolean | true | Simulated ground-truth target — see "Label definition" below | Medium |
| latent_state_for_demo_only | enum | stable_merchant/seasonal_sale_legitimate_returns/operational_fulfilment_failure/high_risk_merchant_behaviour/early_hidden_risk | The hidden latent state that generated this row, exposed only for demonstration and evaluation (e.g. confirming false-positive/false-negative opportunities exist). **Must never be used as a model feature** — it is not observable in a real deployment and would make evaluation circular. | Low |

### Field notes

- `transaction_count_30d`, `transaction_volume_30d`, `refund_count_30d`, `chargeback_count_30d`, and related `_previous_30d` fields are rolling 30-day windows computed as of `week_start`. They may be produced internally from a lower-level synthetic transaction stream, but that transaction stream is a generation implementation detail — it is not separately validated, scored, or cased in Phase 1 (see `ARCHITECTURE.md` Section 5).
- **Resolved (approved 2026-08-22):** declared-vs-observed category mismatch is permanently dropped from Phase 1 — it is not a feature, rule, scenario, or label input. `merchant_category` is kept only for descriptive segmentation and optional segmented metrics (see `MODEL_CARD.md` Evaluation section). Profile/KYC mismatch is mentioned only under Future Work.

## Label definition: `label_high_loss_next_30d`

**This is a design proposal pending final sign-off before training** (per `MODEL_CARD.md`'s requirement that "the exact definition must be frozen before training and documented").

- `1` = the merchant enters a simulated elevated refund/chargeback-loss state during the 30 days following `week_start`.
- `0` = the merchant does not enter that state during the following 30 days.
- **Generation mechanism:** the label is drawn as `Bernoulli(p)`, where `p` is determined by the merchant's hidden/latent state for that week (see below), plus a small adjustment for how many consecutive weeks the merchant has been drifting toward a more severe state. **The label is never computed as a direct deterministic function of the same fields/thresholds the rules engine checks** (e.g. it is not `1 if chargeback_rate_30d > X and refund_rate_30d > Y else 0`). This is the specific design choice that keeps rules-only and ML evaluation honest rather than circular.
- **Label horizon:** because the label looks 30 days (~4-5 weeks) forward, the last ~4-5 weeks of any generated history cannot have a fully observed label. Those trailing weeks are excluded from labeled train/validation/test data, not defaulted to `0`.

### Hidden/latent merchant states (generation-only, not a stored field the model sees)

| State | Approx. share of merchant-weeks | `label_high_loss_next_30d` probability | Purpose |
|---|---|---|---|
| 1. Stable | ~70% | ~1-2% | Baseline population |
| 2. Seasonal / legitimate high-return | ~10% | ~8-12% | Deliberately creates false-positive opportunities |
| 3. Operational fulfilment failure | ~8% | ~30-40% | Genuine moderate-risk population |
| 4. High-risk | ~7% | ~70-85% | Genuine high-risk population |
| 5. Early hidden-risk | ~5% | ~45-60% | Deliberately creates false-negative opportunities (weak current-week signal, real forward risk) |

See `MODEL_CARD.md` for the full generation design (state persistence, feature-distribution overlap, and why a near-perfect held-out result should be investigated rather than reported as-is).

## Case dataset

| Field | Type | Description |
|---|---|---|
| case_id | string | Synthetic review case ID |
| entity_type | enum | merchant_week |
| entity_id | string | `{merchant_id}:{week_start}` composite reference |
| risk_score | float | Risk score from 0 to 100 |
| risk_tier | enum | low/medium/high |
| recommendation | enum | Allowed recommendation list only (see `RISK_POLICY.md`) |
| case_status | enum | open/evidence_requested/under_review/resolved/escalated |
| outcome | enum | confirmed_risk/false_positive/inconclusive/operational_issue |
| reviewer_note | text | Simulated reviewer note |
| created_at | datetime | Case creation timestamp |
| resolved_at | datetime | Resolution timestamp, nullable |

## Audit event dataset

| Field | Type | Description |
|---|---|---|
| audit_event_id | string | Synthetic audit event ID |
| event_timestamp | datetime | Time event was written |
| actor_type | enum | system/analyst/merchant/demo_admin |
| actor_id | string | Synthetic actor token |
| entity_type | enum | merchant_week/case/appeal/model |
| entity_id | string | Related entity ID |
| event_type | enum | scoring_completed/case_created/evidence_requested/etc. |
| payload_json | JSON | Structured event details; must not include sensitive payment data |

## Prohibited fields

The MVP must never include:

- Raw card PAN/card number
- CVV
- UPI PIN
- Bank password or login token
- Aadhaar number
- Real PAN number
- Real account number/IFSC
- Real phone number/email/address
- Real biometric/device fingerprint
- Real customer or merchant name
