"""Tests for POST /auth/login, POST /auth/logout, GET /auth/me."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory
from app.main import app
from app.services.rate_limit import reset_all as reset_rate_limits
from tests.conftest import DEMO_PASSWORD, make_bearer_headers


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture()
def session_factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test_api_auth.db'}")
    init_db(engine)
    factory = make_session_factory(engine)

    def override_get_db():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield factory
    app.dependency_overrides.clear()


@pytest.fixture()
def client(session_factory):
    return TestClient(app)


def _seed_reviewer(session_factory):
    from app.db.database import session_scope
    from app.services import auth_service

    with session_scope(session_factory) as session:
        auth_service.create_user(
            session, username="reviewer_login_test", password=DEMO_PASSWORD, role="reviewer",
            actor_id="analyst_demo_001", display_name="Demo Reviewer",
        )


def test_login_with_correct_credentials_returns_session_token(client, session_factory):
    _seed_reviewer(session_factory)
    response = client.post("/auth/login", json={"username": "reviewer_login_test", "password": DEMO_PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["session_token"]
    assert body["role"] == "reviewer"
    assert body["actor_id"] == "analyst_demo_001"
    assert body["synthetic_data_notice"]


def test_login_with_wrong_password_returns_401_generic_message(client, session_factory):
    _seed_reviewer(session_factory)
    response = client.post("/auth/login", json={"username": "reviewer_login_test", "password": "wrong"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
    assert "invalid username or password" in body["error"]["message"].lower()


def test_login_with_unknown_username_returns_401_same_message(client, session_factory):
    response = client.post("/auth/login", json={"username": "does_not_exist", "password": "anything"})
    assert response.status_code == 401
    assert "invalid username or password" in response.json()["error"]["message"].lower()


def test_get_me_requires_authentication(client, session_factory):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_get_me_returns_current_user_identity(client, session_factory):
    headers = make_bearer_headers(session_factory, "reviewer", "analyst_demo_001", "Demo Reviewer")
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "reviewer"
    assert body["actor_id"] == "analyst_demo_001"
    assert body["display_name"] == "Demo Reviewer"


def test_logout_invalidates_the_session_token(client, session_factory):
    _seed_reviewer(session_factory)
    login_response = client.post("/auth/login", json={"username": "reviewer_login_test", "password": DEMO_PASSWORD})
    token = login_response.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/auth/me", headers=headers).status_code == 200

    logout_response = client.post("/auth/logout", headers=headers)
    assert logout_response.status_code == 200

    assert client.get("/auth/me", headers=headers).status_code == 401


def test_logout_without_a_token_is_a_safe_no_op(client, session_factory):
    response = client.post("/auth/logout")
    assert response.status_code == 200


def test_login_is_rate_limited_after_five_attempts_per_client(client, session_factory):
    _seed_reviewer(session_factory)
    for _ in range(5):
        response = client.post("/auth/login", json={"username": "reviewer_login_test", "password": "wrong"})
        assert response.status_code == 401

    sixth = client.post("/auth/login", json={"username": "reviewer_login_test", "password": "wrong"})
    assert sixth.status_code == 429
    assert sixth.json()["error"]["code"] == "RATE_LIMITED"

    # Even a correct password is rejected once the window's attempt budget
    # is spent -- the limit is per-client, not per-outcome.
    still_limited = client.post("/auth/login", json={"username": "reviewer_login_test", "password": DEMO_PASSWORD})
    assert still_limited.status_code == 429


def test_security_headers_present_on_a_normal_response(client, session_factory):
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"] == "default-src 'self'"


def test_docs_route_is_exempt_from_strict_csp_but_keeps_other_headers(client, session_factory):
    response = client.get("/docs")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" not in response.headers
