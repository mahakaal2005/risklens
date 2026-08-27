"""Local-demo login/logout/current-user routes.

Not production-grade auth -- see docs/PHASE_2_AUTH_DESIGN.md and
SECURITY.md. Demo accounts only, seeded by scripts/seed_demo_users.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, raise_api_error
from app.db.models import User
from app.schemas.api_responses import ErrorCode
from app.schemas.auth import CurrentUserResponse, LoginRequest, LoginResponse
from app.services.auth_service import AuthError, authenticate, create_session, invalidate_session

router = APIRouter(tags=["auth"])
_bearer_scheme = HTTPBearer(auto_error=False)


@router.post("/auth/login", response_model=LoginResponse)
def login_route(body: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
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
