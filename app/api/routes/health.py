"""Health check -- local synthetic-data demonstration only."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.api_responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse()
