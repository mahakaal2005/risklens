"""FastAPI dependencies: one scoped SQLAlchemy session per request, and a
shared helper for structured JSON error responses.

Tests must override get_db() to point at an isolated temporary SQLite
database -- never the developer's local clearrisk_recover.db file.
"""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, sessionmaker

from app.db.database import create_db_engine, init_db, make_session_factory
from app.db.models import User
from app.schemas.api_responses import SYNTHETIC_DATA_NOTICE, ErrorCode
from app.services.auth_service import get_current_user as _get_current_user

_bearer_scheme = HTTPBearer(auto_error=False)

_session_factory: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    """Lazily creates the default engine/session factory on first real use.

    This must stay lazy: tests override get_db() entirely via FastAPI's
    dependency_overrides, and importing this module (or app.main) must
    never touch the developer's real clearrisk_recover.db file as a
    side effect of import order alone.
    """
    global _session_factory
    if _session_factory is None:
        engine = create_db_engine()
        init_db(engine)
        _session_factory = make_session_factory(engine)
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def raise_api_error(status_code: int, code: ErrorCode, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"error": {"code": code.value, "message": message, "synthetic_data_notice": SYNTHETIC_DATA_NOTICE}},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Requires a valid, unexpired session token. Raises 401 -- never
    fabricates or defaults an identity -- for a missing, unknown, or
    expired token. See docs/PHASE_2_AUTH_DESIGN.md."""
    if credentials is None or not credentials.credentials:
        raise_api_error(401, ErrorCode.AUTHENTICATION_REQUIRED, "A valid session token is required.")
    user = _get_current_user(db, credentials.credentials)
    if user is None:
        raise_api_error(401, ErrorCode.AUTHENTICATION_REQUIRED, "Session is missing, invalid, or expired.")
    return user


def require_role(*allowed_roles: str):
    """Dependency factory: raises 403 if the authenticated user's role is
    not in allowed_roles. Never used in place of get_current_user -- always
    layered on top of it, so an unauthenticated request is a 401, not a 403."""

    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise_api_error(403, ErrorCode.FORBIDDEN, f"This action requires role in {sorted(allowed_roles)}.")
        return user

    return _check
