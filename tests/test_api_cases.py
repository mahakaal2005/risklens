import json

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.main import app
from app.services.case_service import create_case_from_packet
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from tests.conftest import make_bearer_headers

FORBIDDEN_ENFORCEMENT_WORDS = ["freeze", "ban", "terminate", "hold settlement", "reject payment", "process payment", "issue refund", "transfer funds"]


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test_api_cases.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    init_db(engine)
    session_factory = make_session_factory(engine)

    def override_get_db():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        packets = json.load(f)
    with session_scope(session_factory) as session:
        for packet in packets.values():
            create_case_from_packet(session, packet)

    reviewer_headers = make_bearer_headers(session_factory, "reviewer", "analyst_demo_001", "Demo Reviewer")
    test_client = TestClient(app)
    test_client.headers.update(reviewer_headers)
    yield test_client
    app.dependency_overrides.clear()


def test_unauthenticated_request_is_rejected(client):
    unauthenticated = TestClient(app)  # deliberately no Authorization header
    response = unauthenticated.get("/cases")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


def test_list_cases_returns_paginated_safe_summaries(client):
    response = client.get("/cases")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4  # 5 demo packets minus 1 APPROVE
    assert len(body["items"]) == 4
    assert body["limit"] == 50
    assert body["offset"] == 0
    for item in body["items"]:
        assert set(item.keys()) >= {"case_id", "merchant_id", "week_start", "case_status", "risk_signal_intensity", "recommendation", "created_at", "updated_at"}


def test_list_cases_filter_by_recommendation(client):
    response = client.get("/cases", params={"recommendation": "MANUAL_REVIEW_REQUIRED"})
    assert response.status_code == 200
    body = response.json()
    assert all(item["recommendation"] == "MANUAL_REVIEW_REQUIRED" for item in body["items"])


def test_list_cases_pagination_limit_default_and_maximum(client):
    default_response = client.get("/cases")
    assert default_response.json()["limit"] == 50

    over_limit_response = client.get("/cases", params={"limit": 500})
    assert over_limit_response.status_code == 422

    max_limit_response = client.get("/cases", params={"limit": 100})
    assert max_limit_response.status_code == 200


def test_list_response_excludes_forbidden_fields(client):
    response = client.get("/cases")
    serialized = json.dumps(response.json())
    assert LABEL_COLUMN not in serialized
    assert LATENT_STATE_COLUMN not in serialized
    assert "support_ticket_rate" not in serialized
    for word in FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in serialized.lower()


def test_case_detail_returns_safe_detail(client):
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == case_id
    assert "analyst_summary" in body
    assert "merchant_safe_explanation" in body
    assert "triggered_rules" in body
    assert "evidence_checklist" in body
    assert "policy_explanation" in body
    serialized = json.dumps(body)
    assert LABEL_COLUMN not in serialized
    assert LATENT_STATE_COLUMN not in serialized
    assert "support_ticket_rate" not in serialized


def test_case_detail_exposes_model_probability_and_rules_only_score_in_analyst_area(client):
    """Regression test for the Milestone 8 integration gap: these two fields
    exist on the ReviewCase DB row from Milestone 5 but were never mapped
    into CaseDetailResponse until this fix."""
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    body = response.json()

    assert "model_probability" in body
    assert "rules_only_score" in body
    assert isinstance(body["rules_only_score"], int)
    assert body["model_probability"] is None or isinstance(body["model_probability"], float)


def test_model_probability_and_rules_only_score_absent_from_merchant_safe_explanation(client):
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.get(f"/cases/{case_id}")
    merchant_safe_serialized = json.dumps(response.json()["merchant_safe_explanation"])
    assert "model_probability" not in merchant_safe_serialized
    assert "rules_only_score" not in merchant_safe_serialized


def test_unknown_case_returns_structured_404(client):
    response = client.get("/cases/does_not_exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "CASE_NOT_FOUND"
    assert body["error"]["synthetic_data_notice"]


def test_audit_events_ordered_and_safe(client):
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.get(f"/cases/{case_id}/audit-events")
    assert response.status_code == 200
    body = response.json()
    sequence_numbers = [e["event_sequence_number"] for e in body["events"]]
    assert sequence_numbers == sorted(sequence_numbers)
    serialized = json.dumps(body)
    assert LABEL_COLUMN not in serialized
    assert LATENT_STATE_COLUMN not in serialized
    for word in FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in serialized.lower()


def test_audit_events_unknown_case_returns_404(client):
    response = client.get("/cases/does_not_exist/audit-events")
    assert response.status_code == 404


def test_valid_reviewer_action_succeeds_and_creates_audit_events(client):
    list_response = client.get("/cases", params={"recommendation": "REQUEST_EVIDENCE"})
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "REQUEST_EVIDENCE", "reviewer_note": "Please provide evidence."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case"]["case_status"] == "EVIDENCE_REQUESTED"
    assert len(body["new_audit_events"]) >= 1


def test_invalid_reviewer_transition_returns_409_with_no_mutation(client):
    list_response = client.get("/cases", params={"recommendation": "MANUAL_REVIEW_REQUIRED"})
    case_id = list_response.json()["items"][0]["case_id"]

    before = client.get(f"/cases/{case_id}").json()

    response = client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "START_REVIEW", "reviewer_note": "trying to skip ahead"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_CASE_TRANSITION"

    after = client.get(f"/cases/{case_id}").json()
    assert before["case_status"] == after["case_status"]
    assert before["updated_at"] == after["updated_at"]


def test_empty_reviewer_note_returns_422(client):
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "REQUEST_EVIDENCE", "reviewer_note": "   "},
    )
    assert response.status_code == 422


def test_unknown_reviewer_action_returns_422(client):
    list_response = client.get("/cases")
    case_id = list_response.json()["items"][0]["case_id"]

    response = client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "FREEZE_ACCOUNT", "reviewer_note": "note"},
    )
    assert response.status_code == 422


def test_no_route_or_response_contains_prohibited_enforcement_words(client):
    openapi_text = json.dumps(app.openapi()).lower()
    for word in FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in openapi_text
