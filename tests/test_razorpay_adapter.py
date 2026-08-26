"""Tests for the local Razorpay-shaped payment-event adapter (ml/razorpay_adapter.py).
See docs/RAZORPAY_ADAPTER.md for the schema and honesty boundaries this exercises.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.razorpay_adapter import (
    FIXTURE_LABEL,
    RazorpayAdapterValidationError,
    aggregate_to_merchant_weeks,
    build_mapping_report,
    load_fixture_events,
    normalize_events,
    run_adapter,
)

FIXTURE_DIR = Path("demo_data/razorpay_fixtures")


def _valid_event(**overrides) -> dict:
    event = {
        "event_id": "evt_demo_00001",
        "event_type": "payment_captured",
        "source_event_name": "payment.captured",
        "merchant_id": "merchant_demo_0001",
        "occurred_at": "2026-07-06T10:00:00",
        "amount_paise": 100000,
        "currency": "INR",
        "fixture_label": FIXTURE_LABEL,
    }
    event.update(overrides)
    return event


def test_load_fixture_events_reads_real_demo_fixtures():
    events = load_fixture_events(FIXTURE_DIR)
    assert len(events) > 0
    assert all(e["fixture_label"] == FIXTURE_LABEL for e in events)


def test_run_adapter_end_to_end_on_real_demo_fixtures():
    merchant_weeks, report = run_adapter(FIXTURE_DIR)
    assert len(merchant_weeks) > 0
    assert report["fixture_label"] == FIXTURE_LABEL
    assert report["prohibited_fields_found"] == []
    assert "merchant_demo_9001" in report["merchant_ids_seen"]


def test_normalize_events_accepts_valid_events():
    df = normalize_events([_valid_event()])
    assert len(df) == 1
    assert df.iloc[0]["event_type"] == "payment_captured"


def test_normalize_events_rejects_prohibited_field():
    with pytest.raises(RazorpayAdapterValidationError) as exc_info:
        normalize_events([_valid_event(card_number="4111111111111111")])
    assert "prohibited field" in str(exc_info.value).lower()


def test_normalize_events_rejects_non_synthetic_merchant_id():
    with pytest.raises(RazorpayAdapterValidationError) as exc_info:
        normalize_events([_valid_event(merchant_id="acme_corp_real_name")])
    assert "synthetic token" in str(exc_info.value).lower()


def test_normalize_events_rejects_missing_fixture_label():
    with pytest.raises(RazorpayAdapterValidationError):
        normalize_events([_valid_event(fixture_label="not_the_right_label")])


def test_normalize_events_rejects_unknown_event_type():
    with pytest.raises(RazorpayAdapterValidationError):
        normalize_events([_valid_event(event_type="payment_refunded_somehow")])


def test_normalize_events_rejects_unknown_dispute_reason():
    with pytest.raises(RazorpayAdapterValidationError):
        normalize_events([_valid_event(event_type="dispute_created", dispute_reason_category="not_a_real_reason")])


def test_normalize_events_rejects_missing_required_key():
    bad_event = _valid_event()
    del bad_event["occurred_at"]
    with pytest.raises(RazorpayAdapterValidationError):
        normalize_events([bad_event])


def test_aggregate_computes_refund_and_chargeback_rates():
    events = [
        _valid_event(event_id="e1", event_type="payment_captured", occurred_at="2026-07-06T10:00:00", amount_paise=100000),
        _valid_event(event_id="e2", event_type="payment_captured", occurred_at="2026-07-07T10:00:00", amount_paise=100000),
        _valid_event(event_id="e3", event_type="refund_processed", occurred_at="2026-07-07T12:00:00", amount_paise=100000),
        _valid_event(event_id="e4", event_type="dispute_created", occurred_at="2026-07-07T13:00:00", dispute_reason_category="not_received"),
    ]
    df = normalize_events(events)
    weeks = aggregate_to_merchant_weeks(df)
    assert len(weeks) == 1
    row = weeks.iloc[0]
    assert row["transaction_count_30d"] == 2
    assert row["refund_count_30d"] == 1
    assert row["refund_rate_30d"] == pytest.approx(0.5)
    assert row["chargeback_count_30d"] == 1
    assert row["chargeback_rate_30d"] == pytest.approx(0.5)
    assert row["top_dispute_reason_category"] == "not_received"
    assert pd.isna(row["transaction_volume_previous_30d"])


def test_aggregate_produces_previous_week_trend_fields():
    events = [
        _valid_event(event_id="e1", event_type="payment_captured", occurred_at="2026-07-06T10:00:00", amount_paise=100000),
        _valid_event(event_id="e2", event_type="payment_captured", occurred_at="2026-07-13T10:00:00", amount_paise=200000),
    ]
    df = normalize_events(events)
    weeks = aggregate_to_merchant_weeks(df).sort_values("week_start").reset_index(drop=True)
    assert len(weeks) == 2
    assert pd.isna(weeks.iloc[0]["transaction_volume_previous_30d"])
    assert weeks.iloc[1]["transaction_volume_previous_30d"] == pytest.approx(1000.0)


def test_zero_transaction_week_has_zero_rates_not_divide_by_zero():
    events = [
        _valid_event(event_id="e1", event_type="refund_processed", occurred_at="2026-07-06T10:00:00", amount_paise=1000),
    ]
    df = normalize_events(events)
    weeks = aggregate_to_merchant_weeks(df)
    row = weeks.iloc[0]
    assert row["transaction_count_30d"] == 0
    assert row["refund_rate_30d"] == 0.0
    assert row["chargeback_rate_30d"] == 0.0


def test_mapping_report_lists_unavailable_merchant_profile_fields():
    df = normalize_events([_valid_event()])
    weeks = aggregate_to_merchant_weeks(df)
    report = build_mapping_report(df, weeks)
    assert "merchant_category" in report["unavailable_fields"]
    assert "support_ticket_rate" in report["unavailable_fields"]
    assert "previous_review_outcome" in report["unavailable_fields"]


def test_no_week_start_top_dispute_reason_when_no_disputes():
    events = [_valid_event()]
    df = normalize_events(events)
    weeks = aggregate_to_merchant_weeks(df)
    assert weeks.iloc[0]["top_dispute_reason_category"] == "none"


def test_output_columns_are_subset_of_raw_input_fields():
    from ml.features import RAW_INPUT_FIELDS

    df = normalize_events([_valid_event()])
    weeks = aggregate_to_merchant_weeks(df)
    assert set(weeks.columns).issubset(set(RAW_INPUT_FIELDS))
