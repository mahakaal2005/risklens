import json

import pytest

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.db.repositories import get_audit_events_for_case, get_case
from app.schemas.cases import PROHIBITED_OUTCOMES, FinalOutcome, ReviewerAction
from app.services.case_service import (
    ACTION_TO_AUDIT_EVENT_TYPE,
    ACTION_TO_FINAL_OUTCOME,
    CaseNotFoundError,
    InvalidTransitionError,
    apply_reviewer_action,
    create_case_from_packet,
    start_review,
)
from app.services.evidence_service import submit_evidence
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.model_utils import ALLOWED_RECOMMENDATIONS


@pytest.fixture()
def session_factory():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(scope="module")
def demo_packets():
    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_initializes_in_temporary_sqlite_db():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)  # should not raise


def test_approve_packet_does_not_create_case(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, events = create_case_from_packet(session, demo_packets["stable_merchant"])
        assert case is None
        assert events == []


def test_non_approve_packet_creates_case_and_required_initial_events(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, events = create_case_from_packet(session, demo_packets["high_risk_combined_loss_case"])
        assert case is not None
        assert case.case_status == "OPEN"
        event_types = {e["event_type"] for e in events}
        assert {"ASSESSMENT_GENERATED", "EXPLANATION_GENERATED", "REVIEW_CASE_CREATED", "REVIEW_CASE_RECOMMENDED"}.issubset(event_types)
        assert "MANUAL_REVIEW_RECOMMENDED" in event_types


def test_target_label_and_latent_state_never_persist(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, events = create_case_from_packet(session, demo_packets["high_risk_combined_loss_case"])

    with session_scope(session_factory) as session:
        stored = get_case(session, case.case_id)
        serialized_case = json.dumps({
            "merchant_safe_explanation": stored.merchant_safe_explanation,
            "analyst_summary": stored.analyst_summary,
            "triggered_rules_json": stored.triggered_rules_json,
            "evidence_checklist_json": stored.evidence_checklist_json,
        })
        assert LABEL_COLUMN not in serialized_case
        assert LATENT_STATE_COLUMN not in serialized_case

        stored_events = get_audit_events_for_case(session, case.case_id)
        for e in stored_events:
            payload_text = json.dumps(e.event_payload_json)
            assert LABEL_COLUMN not in payload_text
            assert LATENT_STATE_COLUMN not in payload_text


def test_event_sequence_numbers_are_contiguous_and_ordered(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["operational_fulfilment_problem"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "REQUEST_EVIDENCE", "Please provide evidence.")

    with session_scope(session_factory) as session:
        submit_evidence(session, case_id, "Operational issue explanation.", ["delivery_proof_demo_001.pdf"])

    with session_scope(session_factory) as session:
        start_review(session, case_id)

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "MARK_OPERATIONAL_ISSUE", "Confirmed operational issue.")

    with session_scope(session_factory) as session:
        events = get_audit_events_for_case(session, case_id)
        sequence_numbers = [e.event_sequence_number for e in events]
        assert sequence_numbers == list(range(1, len(events) + 1))
        assert [e.event_timestamp for e in events] == sorted(e.event_timestamp for e in events)


def test_valid_state_transitions_work(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["seasonal_sale_false_positive_candidate"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        case = apply_reviewer_action(session, case_id, "REQUEST_EVIDENCE", "note")
        assert case.case_status == "EVIDENCE_REQUESTED"


def test_invalid_transition_fails_without_mutation_or_audit_event(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["seasonal_sale_false_positive_candidate"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        events_before = get_audit_events_for_case(session, case_id)
        count_before = len(events_before)
        status_before = get_case(session, case_id).case_status

    with pytest.raises(InvalidTransitionError):
        with session_scope(session_factory) as session:
            # MARK_OPERATIONAL_ISSUE is not a valid action from OPEN in this
            # implementation's table only via EVIDENCE flow -- instead test a
            # truly invalid one: starting review from OPEN.
            start_review(session, case_id)

    with session_scope(session_factory) as session:
        status_after = get_case(session, case_id).case_status
        events_after = get_audit_events_for_case(session, case_id)
        assert status_after == status_before
        assert len(events_after) == count_before


def test_reviewer_action_requires_non_empty_note(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["seasonal_sale_false_positive_candidate"])
        case_id = case.case_id

    with pytest.raises(ValueError):
        with session_scope(session_factory) as session:
            apply_reviewer_action(session, case_id, "CLEAR_CASE", "   ")


def test_resolved_cases_cannot_be_overwritten(session_factory, demo_packets):
    with session_scope(session_factory) as session:
        case, _ = create_case_from_packet(session, demo_packets["seasonal_sale_false_positive_candidate"])
        case_id = case.case_id

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "CLEAR_CASE", "Confirmed as risk on first look.")

    with pytest.raises(InvalidTransitionError):
        with session_scope(session_factory) as session:
            apply_reviewer_action(session, case_id, "MARK_FALSE_POSITIVE", "Trying to change my mind.")

    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        assert case.case_status == "RESOLVED"
        assert case.final_outcome == "CONFIRMED_RISK"


def test_case_not_found_raises(session_factory):
    with pytest.raises(CaseNotFoundError):
        with session_scope(session_factory) as session:
            apply_reviewer_action(session, "nonexistent_case", "CLEAR_CASE", "note")


def test_all_persisted_recommendations_and_outcomes_belong_to_approved_enums(session_factory, demo_packets):
    for name, packet in demo_packets.items():
        with session_scope(session_factory) as session:
            case, _ = create_case_from_packet(session, packet)
            if case is not None:
                assert case.recommendation in ALLOWED_RECOMMENDATIONS

    for outcome in ACTION_TO_FINAL_OUTCOME.values():
        assert outcome in {o.value for o in FinalOutcome}
    for outcome in FinalOutcome:
        assert outcome.value not in PROHIBITED_OUTCOMES


def test_prohibited_outcomes_are_never_reachable():
    reachable_outcomes = set(ACTION_TO_FINAL_OUTCOME.values())
    assert reachable_outcomes.isdisjoint(PROHIBITED_OUTCOMES)


def test_reviewer_action_enum_matches_state_machine_actions():
    from app.services.case_service import VALID_TRANSITIONS

    all_actions_in_transitions = {a for transitions in VALID_TRANSITIONS.values() for a in transitions}
    assert all_actions_in_transitions.issubset({a.value for a in ReviewerAction})
