"""Local Razorpay-shaped payment-event adapter.

Reads synthetic, local JSON fixture files whose event names are modeled on
Razorpay's publicly documented webhook events (see docs/RAZORPAY_ADAPTER.md
for verified sources), normalizes them into a generic event schema, and
aggregates them into merchant-week rows compatible with a subset of
ml/features.py's RAW_INPUT_FIELDS.

This module never calls a Razorpay API, never claims to receive a real
webhook, and never invents fields a payment/refund/dispute/settlement event
stream cannot honestly contain (merchant profile / support-desk data --
see docs/RAZORPAY_ADAPTER.md Section 4). Every output carries
FIXTURE_LABEL. This module also never produces label_high_loss_next_30d or
latent_state_for_demo_only -- those remain exclusively the output of
ml/generate_synthetic_data.py's latent-state simulation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ml.data_validation import PROHIBITED_FIELD_NAMES

FIXTURE_LABEL = "razorpay_shaped_demo_fixture_not_live_razorpay_data"

MERCHANT_ID_PATTERN = re.compile(r"^merchant_demo_\d+$")

GENERIC_EVENT_TYPES = frozenset(
    {
        "payment_captured",
        "payment_failed",
        "refund_created",
        "refund_processed",
        "refund_failed",
        "dispute_created",
        "dispute_won",
        "dispute_lost",
        "settlement_processed",
    }
)

DISPUTE_REASON_CATEGORIES = frozenset(
    {"item_not_as_described", "not_received", "duplicate_charge", "quality_issue", "other"}
)

UNAVAILABLE_MERCHANT_WEEK_FIELDS = [
    "merchant_category",
    "merchant_age_days",
    "delivery_evidence_coverage",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
    "previous_review_outcome",
]

REQUIRED_EVENT_KEYS = {"event_id", "event_type", "source_event_name", "merchant_id", "occurred_at", "currency"}


class RazorpayAdapterValidationError(Exception):
    """Raised when fixture events fail schema, PII, or honesty validation.
    Per CLAUDE.md's fail-safely principle, aggregation never proceeds past
    this -- bad fixture rows are never silently dropped or repaired."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Razorpay-shaped fixture validation failed:\n- " + "\n- ".join(issues))


def load_fixture_events(fixture_dir: str | Path) -> list[dict]:
    """Read every *.json fixture file in fixture_dir into a flat list of
    raw event dicts. Each file is expected to contain a JSON list of events."""

    fixture_dir = Path(fixture_dir)
    events: list[dict] = []
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise RazorpayAdapterValidationError([f"{path}: fixture file must contain a JSON list of events"])
        events.extend(payload)
    return events


def validate_events(events: list[dict]) -> list[str]:
    """Return a list of validation issues (empty if the events are clean).
    Does not raise -- callers decide whether to raise via normalize_events()."""

    issues: list[str] = []
    for i, event in enumerate(events):
        missing_keys = REQUIRED_EVENT_KEYS - set(event.keys())
        if missing_keys:
            issues.append(f"event[{i}]: missing required key(s) {sorted(missing_keys)}")
            continue

        prohibited_present = PROHIBITED_FIELD_NAMES & set(event.keys())
        if prohibited_present:
            issues.append(f"event[{i}] ({event.get('event_id')}): prohibited field(s) present {sorted(prohibited_present)}")

        if event["event_type"] not in GENERIC_EVENT_TYPES:
            issues.append(f"event[{i}] ({event.get('event_id')}): unknown event_type {event['event_type']!r}")

        if not MERCHANT_ID_PATTERN.match(str(event.get("merchant_id", ""))):
            issues.append(f"event[{i}] ({event.get('event_id')}): merchant_id {event.get('merchant_id')!r} is not a synthetic token matching '{MERCHANT_ID_PATTERN.pattern}'")

        if event.get("fixture_label") != FIXTURE_LABEL:
            issues.append(f"event[{i}] ({event.get('event_id')}): missing or incorrect fixture_label (must be {FIXTURE_LABEL!r})")

        reason = event.get("dispute_reason_category")
        if reason is not None and reason not in DISPUTE_REASON_CATEGORIES:
            issues.append(f"event[{i}] ({event.get('event_id')}): dispute_reason_category {reason!r} is not one of {sorted(DISPUTE_REASON_CATEGORIES)}")

    return issues


def normalize_events(events: list[dict]) -> pd.DataFrame:
    """Validate and flatten raw fixture events into the generic event schema
    as a DataFrame. Raises RazorpayAdapterValidationError on any issue."""

    issues = validate_events(events)
    if issues:
        raise RazorpayAdapterValidationError(issues)

    df = pd.DataFrame(events)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"])
    if "amount_paise" not in df.columns:
        df["amount_paise"] = None
    if "dispute_reason_category" not in df.columns:
        df["dispute_reason_category"] = None
    return df


def _week_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return (timestamp - pd.to_timedelta(timestamp.dayofweek, unit="D")).normalize()


def aggregate_to_merchant_weeks(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized events into one row per merchant + ISO calendar
    week. Output columns are a strict subset of ml.features.RAW_INPUT_FIELDS
    -- see docs/RAZORPAY_ADAPTER.md Section 3 for the exact formulas and
    Section 4 for the merchant-week fields this cannot honestly produce."""

    working = events_df.copy()
    working["week_start"] = working["occurred_at"].apply(_week_start)

    rows: dict[tuple[str, pd.Timestamp], dict] = {}
    for (merchant_id, week_start), group in working.groupby(["merchant_id", "week_start"]):
        captured = group[group["event_type"] == "payment_captured"]
        refunded = group[group["event_type"] == "refund_processed"]
        disputed = group[group["event_type"] == "dispute_created"]

        transaction_count = int(len(captured))
        transaction_volume = float(captured["amount_paise"].fillna(0).sum()) / 100.0
        refund_count = int(len(refunded))
        chargeback_count = int(len(disputed))

        reasons = disputed["dispute_reason_category"].dropna()
        top_reason = Counter(reasons).most_common(1)[0][0] if len(reasons) else "none"

        rows[(merchant_id, week_start)] = {
            "merchant_id": merchant_id,
            "week_start": week_start.strftime("%Y-%m-%d"),
            "transaction_count_30d": transaction_count,
            "transaction_volume_30d": transaction_volume,
            "refund_count_30d": refund_count,
            "refund_rate_30d": (refund_count / transaction_count) if transaction_count else 0.0,
            "chargeback_count_30d": chargeback_count,
            "chargeback_rate_30d": (chargeback_count / transaction_count) if transaction_count else 0.0,
            "top_dispute_reason_category": top_reason,
        }

    result = pd.DataFrame(rows.values()).sort_values(["merchant_id", "week_start"]).reset_index(drop=True)

    prev_volume = []
    prev_refund_rate = []
    prev_chargeback_rate = []
    last_by_merchant: dict[str, dict] = {}
    for _, row in result.iterrows():
        prior = last_by_merchant.get(row["merchant_id"])
        prev_volume.append(prior["transaction_volume_30d"] if prior is not None else None)
        prev_refund_rate.append(prior["refund_rate_30d"] if prior is not None else None)
        prev_chargeback_rate.append(prior["chargeback_rate_30d"] if prior is not None else None)
        last_by_merchant[row["merchant_id"]] = row

    result["transaction_volume_previous_30d"] = prev_volume
    result["refund_rate_previous_30d"] = prev_refund_rate
    result["chargeback_rate_previous_30d"] = prev_chargeback_rate

    return result


def build_mapping_report(events_df: pd.DataFrame, merchant_weeks: pd.DataFrame) -> dict:
    """Build the mapping/data-quality report required by docs/RAZORPAY_ADAPTER.md
    Section 5. Never raises -- callers should have already validated via
    normalize_events()."""

    return {
        "fixture_label": FIXTURE_LABEL,
        "event_count_by_type": events_df["event_type"].value_counts().to_dict(),
        "merchant_week_count": int(len(merchant_weeks)),
        "unavailable_fields": UNAVAILABLE_MERCHANT_WEEK_FIELDS,
        "prohibited_fields_found": [],
        "merchant_ids_seen": sorted(events_df["merchant_id"].unique().tolist()),
    }


def run_adapter(fixture_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """End-to-end: load -> normalize/validate -> aggregate -> report."""

    events = load_fixture_events(fixture_dir)
    events_df = normalize_events(events)
    merchant_weeks = aggregate_to_merchant_weeks(events_df)
    report = build_mapping_report(events_df, merchant_weeks)
    return merchant_weeks, report


if __name__ == "__main__":
    fixture_directory = Path("demo_data/razorpay_fixtures")
    weeks_df, mapping_report = run_adapter(fixture_directory)

    artifacts_dir = Path("ml/artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    report_path = artifacts_dir / "razorpay_adapter_report.json"
    with report_path.open("w") as f:
        json.dump(mapping_report, f, indent=2, default=str)

    print(f"Fixture label: {FIXTURE_LABEL}")
    print(f"Read fixtures from: {fixture_directory}")
    print(f"Merchant-week rows produced: {len(weeks_df)}")
    print(weeks_df.to_string(index=False))
    print(f"\nMapping/data-quality report written to: {report_path}")
    print(json.dumps(mapping_report, indent=2, default=str))
