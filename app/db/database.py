"""SQLite database setup for RiskLens.

Local-only, single-file SQLite database. There is no authentication, no
production hardening, and no real payment/gateway connection here -- see
SECURITY.md. The application audit log built on top of this database is
append-only at the application layer only; it is not cryptographically
immutable or WORM storage (see docs/AUDIT_EVENT_SCHEMA.md).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

DEFAULT_DATABASE_URL = "sqlite:///./clearrisk_recover.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_db_engine(database_url: str | None = None):
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine) -> None:
    Base.metadata.create_all(bind=engine)


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker) -> Session:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
