"""Local-demo login/logout/current-user routes.

Not production-grade auth -- see docs/PHASE_2_AUTH_DESIGN.md and
SECURITY.md. Demo accounts only, seeded by scripts/seed_demo_users.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, raise_api_error
from app.db.models import User
from app.schemas.api_responses import ErrorCode
from app.schemas.auth import CurrentUserResponse, LoginRequest, LoginResponse
from app.services.auth_service import AuthError, authenticate, create_session, invalidate_session
from app.services.rate_limit import RateLimitExceededError, check_rate_limit

router = APIRouter(tags=["auth"])
_bearer_scheme = HTTPBearer(auto_error=False)

# Per the project's standing 5-security-checks requirement (Prompt 3 item 5):
# authentication endpoints must be rate limited. In-memory, per-client-IP,
# sliding window -- adequate for this local single-process demo, not a
# distributed production rate limiter.
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60.0


@router.post("/auth/login", response_model=LoginResponse)
def login_route(body: LoginRequest, request: Request, db: Session = Depends(get_db)) -> LoginResponse:
    client_ip = request.client.host if request.client else "unknown"
    try:
        check_rate_limit(f"login:{client_ip}", LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS)
    except RateLimitExceededError:
        raise_api_error(429, ErrorCode.RATE_LIMITED, "Too many login attempts. Please wait a minute and try again.")

    try:
        user = authenticate(db, body.username, body.password)
    except AuthError as exc:
        raise_api_error(401, ErrorCode.AUTHENTICATION_REQUIRED, str(exc))

    session = create_session(db, user)
    return LoginResponse(
        session_token=session.token,
        role=user.role,
        actor_id=user.actor_id,
        display_name=user.display_name,
        merchant_id=user.merchant_id,
        expires_at=session.expires_at,
    )


@router.post("/auth/logout")
def logout_route(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> dict:
    if credentials is not None and credentials.credentials:
        invalidate_session(db, credentials.credentials)
    return {"status": "logged_out"}


@router.get("/auth/me", response_model=CurrentUserResponse)
def current_user_route(user: User = Depends(get_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse(
        role=user.role,
        actor_id=user.actor_id,
        display_name=user.display_name,
        merchant_id=user.merchant_id,
    )
