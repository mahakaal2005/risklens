"""Pydantic schemas for the local-demo authentication layer.

See docs/PHASE_2_AUTH_DESIGN.md for the full design and explicit
non-goals. Not production-grade auth.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

from app.schemas.api_responses import SYNTHETIC_DATA_NOTICE


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    session_token: str
    role: str
    actor_id: str
    display_name: str
    merchant_id: str | None = None
    expires_at: dt.datetime
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class CurrentUserResponse(BaseModel):
    role: str
    actor_id: str
    display_name: str
    merchant_id: str | None = None
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE
