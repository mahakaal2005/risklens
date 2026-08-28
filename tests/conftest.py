"""Shared test helpers for the Phase 2 auth layer.

Each test file still builds its own isolated in-memory/tmp SQLite
database via its own `client` fixture (existing pattern, unchanged) --
this module only adds a way to seed demo users and mint session tokens
directly against that same session_factory, without going through the
network /auth/login round trip.
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from app.db.database import session_scope
from app.services import auth_service

DEMO_PASSWORD = "test-demo-password-1234"


def make_bearer_headers(
    session_factory: sessionmaker,
    role: str,
    actor_id: str,
    display_name: str,
    merchant_id: str | None = None,
    username: str | None = None,
) -> dict[str, str]:
    """Creates a demo user with the given role/identity directly against
    the test database and returns an Authorization header carrying a
    freshly minted session token for it."""
    username = username or f"{role}_{actor_id}"
    with session_scope(session_factory) as session:
        user = auth_service.create_user(
            session,
            username=username,
            password=DEMO_PASSWORD,
            role=role,
            actor_id=actor_id,
            display_name=display_name,
            merchant_id=merchant_id,
        )
        user_session = auth_service.create_session(session, user)
        token = user_session.token
    return {"Authorization": f"Bearer {token}"}
