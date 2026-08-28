"""Unit tests for app/services/auth_service.py -- password hashing,
authentication, and session lifecycle. See docs/PHASE_2_AUTH_DESIGN.md."""

from __future__ import annotations

import datetime as dt

import pytest

from app.db.database import create_db_engine, init_db, make_session_factory
from app.services import auth_service


@pytest.fixture()
def db_session(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test_auth_service.db'}")
    init_db(engine)
    factory = make_session_factory(engine)
    session = factory()
    yield session
    session.close()


def test_hash_password_is_salted_and_deterministic_given_the_same_salt():
    hash_a, salt_a = auth_service.hash_password("correct-horse-battery-staple")
    hash_b, _ = auth_service.hash_password("correct-horse-battery-staple", salt=salt_a)
    hash_c, salt_c = auth_service.hash_password("correct-horse-battery-staple")

    assert hash_a == hash_b  # same password + same salt -> same hash
    assert salt_a != salt_c  # freshly generated salts differ
    assert hash_a != hash_c  # different salt -> different hash despite same password


def test_verify_password_accepts_correct_and_rejects_wrong_password():
    password_hash, salt = auth_service.hash_password("correct-horse-battery-staple")
    assert auth_service.verify_password("correct-horse-battery-staple", password_hash, salt) is True
    assert auth_service.verify_password("wrong-password", password_hash, salt) is False


def test_create_user_rejects_invalid_role(db_session):
    with pytest.raises(ValueError):
        auth_service.create_user(
            db_session, username="bad_role_user", password="x", role="admin", actor_id="x", display_name="X",
        )


def test_authenticate_succeeds_with_correct_credentials(db_session):
    auth_service.create_user(
        db_session, username="reviewer1", password="s3cret-pass", role="reviewer",
        actor_id="analyst_demo_001", display_name="Demo Reviewer",
    )
    db_session.commit()

    user = auth_service.authenticate(db_session, "reviewer1", "s3cret-pass")
    assert user.username == "reviewer1"
    assert user.role == "reviewer"


def test_authenticate_fails_with_wrong_password_and_message_is_generic(db_session):
    auth_service.create_user(
        db_session, username="reviewer2", password="s3cret-pass", role="reviewer",
        actor_id="analyst_demo_002", display_name="Demo Reviewer 2",
    )
    db_session.commit()

    with pytest.raises(auth_service.AuthError) as exc_info:
        auth_service.authenticate(db_session, "reviewer2", "wrong-password")
    assert "invalid username or password" in str(exc_info.value).lower()


def test_authenticate_fails_for_unknown_username_with_same_generic_message(db_session):
    with pytest.raises(auth_service.AuthError) as exc_info:
        auth_service.authenticate(db_session, "no_such_user", "anything")
    assert "invalid username or password" in str(exc_info.value).lower()


def test_create_session_and_get_current_user_round_trip(db_session):
    user = auth_service.create_user(
        db_session, username="merchant1", password="pw", role="merchant",
        actor_id="merchant_demo_actor_1", display_name="Demo Merchant", merchant_id="merchant_demo_0001",
    )
    db_session.commit()

    session = auth_service.create_session(db_session, user)
    db_session.commit()

    fetched = auth_service.get_current_user(db_session, session.token)
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.merchant_id == "merchant_demo_0001"


def test_get_current_user_returns_none_for_unknown_token(db_session):
    assert auth_service.get_current_user(db_session, "not-a-real-token") is None


def test_get_current_user_returns_none_for_expired_session(db_session):
    user = auth_service.create_user(
        db_session, username="reviewer3", password="pw", role="reviewer",
        actor_id="analyst_demo_003", display_name="Demo Reviewer 3",
    )
    session = auth_service.create_session(db_session, user)
    # Force expiry into the past.
    session.expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    db_session.commit()

    assert auth_service.get_current_user(db_session, session.token) is None


def test_invalidate_session_makes_token_unusable(db_session):
    user = auth_service.create_user(
        db_session, username="reviewer4", password="pw", role="reviewer",
        actor_id="analyst_demo_004", display_name="Demo Reviewer 4",
    )
    session = auth_service.create_session(db_session, user)
    db_session.commit()

    assert auth_service.get_current_user(db_session, session.token) is not None
    auth_service.invalidate_session(db_session, session.token)
    db_session.commit()
    assert auth_service.get_current_user(db_session, session.token) is None
