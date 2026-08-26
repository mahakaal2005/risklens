# Razorpay-Shaped Payment-Event Adapter

**Status: local, synthetic-fixture demonstration only. Not a Razorpay integration.**

This adapter does not call any Razorpay API, does not hold Razorpay credentials, and
does not process a real webhook. It reads local, synthetic, anonymized JSON fixture
files whose event *names and general shape* are modeled on Razorpay's publicly
documented webhook events, so that a ClearRisk Recover demo can show a plausible path
from "payment-platform events" to a scored merchant-week — without ever claiming to
be connected to Razorpay or any other real gateway.

Every artifact this adapter produces — fixtures, normalized events, aggregates, and
the mapping/data-quality report — carries the literal label
`"razorpay_shaped_demo_fixture_not_live_razorpay_data"`.

## 1. Verified event names used

The event-name strings below were confirmed against Razorpay's public webhook
documentation (checked 2026-08-25). They are **company-reported/documented event
names**, not invented by this project. Fields inside each fixture event beyond the
top-level name/timestamp/amount/currency are this project's own simplification for
demo purposes — they are not claimed to be a byte-for-byte real Razorpay payload.

| Generic event_type (this adapter) | Razorpay-documented source event name |
|---|---|
| `payment_captured` | `payment.captured` |
| `payment_failed` | `payment.failed` |
| `refund_created` | `refund.created` |
| `refund_processed` | `refund.processed` |
| `refund_failed` | `refund.failed` |
| `dispute_created` | `payment.dispute.created` (also documented as `dispute.created`) |
| `dispute_won` | `dispute.won` |
| `dispute_lost` | `dispute.lost` |
| `settlement_processed` | `settlement.processed` |

Sources: Razorpay Docs — [Disputes Webhook Events](https://razorpay.com/docs/webhooks/disputes/),
[Dispute Payloads](https://razorpay.com/docs/webhooks/payloads/disputes/),
[Settlements Webhook Events](https://razorpay.com/docs/webhooks/settlements/),
[About Webhooks](https://razorpay.com/docs/webhooks/).

**Design decision (this project, not a Razorpay claim):** Razorpay's own domain model
does not have a field literally called "chargeback" — it uses "dispute" for the
customer/bank-initiated challenge that ClearRisk Recover's existing merchant-week
schema calls `chargeback_count_30d` / `chargeback_rate_30d`. This adapter treats a
`dispute_created` event as the chargeback-count signal. That mapping is a **design
proposal** of this project, not a verified Razorpay statement that "dispute" and
"chargeback" are the same thing in all contexts.

Amounts follow Razorpay's documented convention of the smallest currency unit (paise
for INR); the adapter converts to rupees when aggregating.

## 2. Generic normalized event schema

Raw fixture events are first normalized into one flat, generic schema so the
aggregator never has to branch on Razorpay-specific payload shape:

| Field | Type | Meaning |
|---|---|---|
| `event_id` | str | Synthetic event token, e.g. `evt_demo_00001` |
| `event_type` | str | One of the 9 generic types in the table above |
| `source_event_name` | str | The Razorpay-documented event name this was modeled on |
| `merchant_id` | str | Synthetic token, must match `merchant_demo_\d+` |
| `occurred_at` | str (ISO 8601) | Event timestamp |
| `amount_paise` | int or null | Amount in paise, where applicable (payments, refunds, settlements) |
| `currency` | str | Always `"INR"` in these fixtures |
| `dispute_reason_category` | str or null | Present only on `dispute_created`; one of the same closed enum already used in `DATA_DICTIONARY.md`'s `top_dispute_reason_category` (`item_not_as_described`/`not_received`/`duplicate_charge`/`quality_issue`/`other`) |
| `fixture_label` | str | Always `"razorpay_shaped_demo_fixture_not_live_razorpay_data"` |

## 3. Merchant-week aggregation

Events are grouped by `merchant_id` and by the ISO calendar week containing
`occurred_at`, producing one row per merchant-week with:

- `transaction_count_30d` — count of `payment_captured` events in the week
- `transaction_volume_30d` — sum of `payment_captured` amounts (rupees) in the week
- `refund_count_30d` — count of `refund_processed` events in the week
- `refund_rate_30d` — `refund_count_30d / transaction_count_30d`, `0.0` if no transactions
- `chargeback_count_30d` — count of `dispute_created` events in the week
- `chargeback_rate_30d` — `chargeback_count_30d / transaction_count_30d`, `0.0` if no transactions
- `top_dispute_reason_category` — most frequent `dispute_reason_category` that week, or `"none"` if no disputes
- `transaction_volume_previous_30d`, `refund_rate_previous_30d`, `chargeback_rate_previous_30d` — the same three metrics computed for that merchant's immediately preceding week bucket in the fixture set, or `null` if there is no prior week

**Naming note:** the existing merchant-week schema (`ml/features.py`) uses a `_30d`
suffix for what is really a rolling 30-day window. This adapter's fixtures are
grouped by calendar week for simplicity, so the `_30d`-suffixed field names here
approximate — rather than exactly reproduce — that rolling-window definition. This is
stated here so the naming reuse is never mistaken for a claim of true 30-day
rolling aggregation.

## 4. Fields this adapter cannot honestly produce

Payment/refund/dispute/settlement events alone do not contain merchant profile or
support-desk data. This adapter does **not** invent values for these
`ml/features.py` / `DATA_DICTIONARY.md` fields — it reports them as unavailable
rather than fabricating plausible-looking numbers:

- `merchant_category`
- `merchant_age_days`
- `delivery_evidence_coverage`
- `support_ticket_rate`
- `average_support_resolution_time_hours`
- `previous_review_outcome`

A real Phase 2+ integration would need a separate merchant-profile/support-system
feed to fill these in. The adapter's mapping/data-quality report (Section 5) lists
this gap explicitly for every run.

## 5. PII / prohibited-field checks and the mapping report

Before aggregating, the adapter runs the same prohibited-field-name check already
used for synthetic training data (`ml/data_validation.py::PROHIBITED_FIELD_NAMES` —
card PAN, CVV, UPI PIN, bank credentials, Aadhaar/PAN numbers, phone/email/address,
device fingerprints, real customer/merchant names) against every fixture event's
keys, and validates every `merchant_id` against the same `merchant_demo_\d+` token
pattern used everywhere else in this project. Any violation raises
`RazorpayAdapterValidationError` and aggregation does not proceed — per `CLAUDE.md`'s
"fail safely" principle, this adapter never silently drops or repairs bad fixture
rows.

`build_mapping_report()` returns a dict (also written to
`ml/artifacts/razorpay_adapter_report.json` when run as a script) with:

- `fixture_label`: the mandatory demo-fixture disclaimer
- `event_count_by_type`: how many of each of the 9 event types were read
- `merchant_week_count`: how many merchant-week rows were produced
- `unavailable_fields`: the Section 4 list, always present
- `prohibited_fields_found`: always `[]` if validation passed (the run would have
  raised otherwise)
- `merchant_ids_seen`: the distinct synthetic merchant tokens found

## 6. How this feeds ClearRisk Recover's existing workflow

`ml/razorpay_adapter.py::aggregate_to_merchant_weeks()` returns a `pandas.DataFrame`
with the columns from Section 3 — a strict subset of `ml.features.RAW_INPUT_FIELDS`.
It does not call the rules engine or the model itself, and it does not write to the
application database. A future milestone could supplement the missing Section 4
fields (e.g. from a stubbed merchant-profile fixture) and pass the combined row
through the existing `ml/features.py::build_feature_row()` /
`ml/rules_engine.py` path — that wiring is intentionally left undone here so this
adapter is not mistaken for a fully live scoring path.

## 7. Known limitations

- Fixtures are generated (seeded, deterministic) synthetic data — 5 merchants
  (`merchant_demo_9001`-`merchant_demo_9005`) x 4 weeks x varied event mixes
  (122 events total, 20 merchant-week rows), not a claim of representative
  Razorpay event volume or timing.
- Only 9 event types are modeled; Razorpay documents additional dispute states
  (e.g. under-review, closed, action-required) not used in this demo.
- No signature verification is implemented or claimed, because no real webhook is
  ever received — see `SECURITY.md` for the project's live-webhook non-claim.
- This adapter cannot and does not produce `label_high_loss_next_30d` or
  `latent_state_for_demo_only` — those remain exclusively the output of
  `ml/generate_synthetic_data.py`'s latent-state simulation, per `CLAUDE.md`.
