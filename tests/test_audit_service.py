import inspect
import json

import pytest

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.services import audit_service
from app.services.audit_service import UnsafeAuditPayloadError, get_case_timeline, record_event
from app.services.case_service import create_case_from_packet


@pytest.fixture()
def session_factory():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(scope="module")
def demo_packets():
    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_audit_service_exposes_no_update_or_delete_function():
    public_functions = {name for name, obj in inspect.getmembers(audit_service, inspect.isfunction)}
    for forbidden in ("update_event", "delete_event", "edit_event", "remove_event"):
        assert forbidden not in public_functions


def test_record_event_is_append_only_through_service_layer(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["high_risk_combined_loss_case"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        record_event(session, case_id, "analyst_demo", "analyst_demo", "REVIEW_STARTED", {"note": "manual test event"})

    with session_scope(session_factory) as session:
        timeline = get_case_timeline(session, case_id)
        assert len(timeline) >= 1
        # No mechanism exists to mutate an existing event's payload; confirm
        # the timeline is only ever grown by calling record_event again.
        before_count = len(timeline)

    with session_scope(session_factory) as session:
        record_event(session, case_id, "analyst_demo", "analyst_demo", "REVIEW_STARTED", {"note": "second event"})

    with session_scope(session_factory) as session:
        timeline = get_case_timeline(session, case_id)
        assert len(timeline) == before_count + 1


def test_forbidden_enforcement_terms_rejected_in_payload(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["high_risk_combined_loss_case"])
        case_id = case.case_id

    forbidden_payloads = [
        {"note": "we should freeze this account"},
        {"note": "recommend a ban"},
        {"note": "terminate the merchant"},
        {"note": "hold settlement pending review"},
        {"note": "reject payment immediately"},
    ]
    for payload in forbidden_payloads:
        with pytest.raises(UnsafeAuditPayloadError):
            with session_scope(session_factory) as session:
                record_event(session, case_id, "analyst_demo", "analyst_demo", "REVIEW_STARTED", payload)


def test_timeline_ordered_by_sequence_then_timestamp(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["high_risk_combined_loss_case"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        timeline = get_case_timeline(session, case_id)
        sequence_numbers = [e["event_sequence_number"] for e in timeline]
        assert sequence_numbers == sorted(sequence_numbers)
