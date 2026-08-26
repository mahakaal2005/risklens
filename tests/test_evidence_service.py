import json

import pytest

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.db.repositories import get_case
from app.services.case_service import InvalidTransitionError, apply_reviewer_action, create_case_from_packet
from app.services.evidence_service import submit_evidence


@pytest.fixture()
def session_factory():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(scope="module")
def demo_packets():
    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _create_case_in_evidence_requested(session_factory, demo_packets, packet_name="seasonal_sale_false_positive_candidate"):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets[packet_name])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "REQUEST_EVIDENCE", "Please provide evidence.")

    return case_id


def test_evidence_can_only_be_submitted_in_evidence_requested_status(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["seasonal_sale_false_positive_candidate"])
        case_id = case.case_id

    with pytest.raises(InvalidTransitionError):
        with session_scope(session_factory) as session:
            submit_evidence(session, case_id, "Explanation.", ["invoice_demo_001.pdf"])


@pytest.mark.parametrize("bad_reference", [
    "../../etc/passwd",
    "http://example.com/file.pdf",
    "rm -rf /",
    "file; rm -rf /",
    "",
    "a" * 200,
])
def test_invalid_evidence_references_are_rejected(session_factory, demo_packets, bad_reference):
    case_id = _create_case_in_evidence_requested(session_factory, demo_packets)
    with pytest.raises(ValueError):
        with session_scope(session_factory) as session:
            submit_evidence(session, case_id, "Explanation text.", [bad_reference])


def test_too_many_evidence_references_rejected(session_factory, demo_packets):
    case_id = _create_case_in_evidence_requested(session_factory, demo_packets)
    with pytest.raises(ValueError):
        with session_scope(session_factory) as session:
            submit_evidence(session, case_id, "Explanation.", [f"file_demo_{i}.pdf" for i in range(6)])


def test_blank_merchant_explanation_rejected(session_factory, demo_packets):
    case_id = _create_case_in_evidence_requested(session_factory, demo_packets)
    with pytest.raises(ValueError):
        with session_scope(session_factory) as session:
            submit_evidence(session, case_id, "   ", ["invoice_demo_001.pdf"])


def test_valid_evidence_submission_creates_record_transition_and_audit_event(session_factory, demo_packets):
    case_id = _create_case_in_evidence_requested(session_factory, demo_packets)

    with session_scope(session_factory) as session:
        evidence, audit_event = submit_evidence(
            session, case_id, "Seasonal sale explanation.", ["invoice_demo_001.pdf", "refund_policy_demo_url"],
        )
        assert evidence.case_id == case_id
        assert evidence.evidence_references_json == ["invoice_demo_001.pdf", "refund_policy_demo_url"]
        assert audit_event["event_type"] == "EVIDENCE_SUBMITTED"

    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        assert case.case_status == "EVIDENCE_SUBMITTED"


def test_valid_safe_demo_references_accepted(session_factory, demo_packets):
    case_id = _create_case_in_evidence_requested(session_factory, demo_packets)
    with session_scope(session_factory) as session:
        evidence, _ = submit_evidence(
            session, case_id, "Explanation.",
            ["invoice_demo_001.pdf", "delivery_proof_demo_001.pdf", "refund_policy_demo_url"],
        )
        assert len(evidence.evidence_references_json) == 3
