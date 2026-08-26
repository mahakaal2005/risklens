"""FastAPI dependencies: one scoped SQLAlchemy session per request, and a
shared helper for structured JSON error responses.

Tests must override get_db() to point at an isolated temporary SQLite
database -- never the developer's local clearrisk_recover.db file.
"""

from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException
from sqlalchemy.orm import Session, sessionmaker

from app.db.database import create_db_engine, init_db, make_session_factory
from app.schemas.api_responses import SYNTHETIC_DATA_NOTICE, ErrorCode

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
