"""Local-demo authentication service.

Not production-grade auth -- see docs/PHASE_2_AUTH_DESIGN.md and
SECURITY.md for the explicit non-goals (no MFA, no password reset, no
login rate limiting, no external identity provider). Password hashing
uses the stdlib pbkdf2_hmac KDF: adequate for this local single-operator
prototype, not a claim of production password-security compliance.

Demo accounts are seeded, fixed identities (scripts/seed_demo_users.py),
same spirit as "merchant_demo_001" elsewhere in this codebase -- not real
people.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User, UserSession

PBKDF2_ITERATIONS = 260_000
SESSION_LIFETIME = dt.timedelta(hours=12)

VALID_ROLES = {"reviewer", "merchant", "risk_manager"}


class AuthError(Exception):
    """Raised for any login/session failure. Message is always generic
    ("invalid username or password") -- never reveals which part was wrong,
    and never echoes back user-supplied input."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Returns (password_hash_hex, salt_hex). Generates a new random salt if
    none is given."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate_hash, password_hash)


def create_user(
    session: Session,
    username: str,
    password: str,
    role: str,
    actor_id: str,
    display_name: str,
    merchant_id: str | None = None,
) -> User:
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}.")
    password_hash, salt = hash_password(password)
    user = User(
        id=f"user_{uuid.uuid4().hex[:12]}",
        username=username,
        password_hash=password_hash,
        password_salt=salt,
        role=role,
        actor_id=actor_id,
        merchant_id=merchant_id,
        display_name=display_name,
    )
    session.add(user)
    session.flush()
    return user


def authenticate(session: Session, username: str, password: str) -> User:
    """Raises AuthError with a generic message on any failure -- unknown
    username and wrong password are indistinguishable to the caller."""
    stmt = select(User).where(User.username == username)
    user = session.execute(stmt).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash, user.password_salt):
        raise AuthError("Invalid username or password.")
    return user


def create_session(session: Session, user: User) -> UserSession:
    now = _utcnow()
    user_session = UserSession(
        token=secrets.token_urlsafe(32),
        user_id=user.id,
        created_at=now,
        expires_at=now + SESSION_LIFETIME,
    )
    session.add(user_session)
    session.flush()
    return user_session


def _as_aware_utc(value: dt.datetime) -> dt.datetime:
    """SQLite has no native timezone-aware column type -- a DateTime(timezone=True)
    value written as UTC can still come back naive on read, depending on
    driver/dialect behavior. Treat any naive value as UTC rather than
    letting it silently compare wrong (or raise) against an aware value."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.timezone.utc)


def get_current_user(session: Session, token: str) -> User | None:
    """Returns None for a missing, unknown, or expired token -- callers
    (the FastAPI dependency) turn that into a 401, never a fabricated
    identity."""
    stmt = select(UserSession).where(UserSession.token == token)
    user_session = session.execute(stmt).scalar_one_or_none()
    if user_session is None:
        return None
    if _as_aware_utc(user_session.expires_at) < _utcnow():
        return None
    return session.get(User, user_session.user_id)


def invalidate_session(session: Session, token: str) -> None:
    stmt = select(UserSession).where(UserSession.token == token)
    user_session = session.execute(stmt).scalar_one_or_none()
    if user_session is not None:
        session.delete(user_session)
        session.flush()
