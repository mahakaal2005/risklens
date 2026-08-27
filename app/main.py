"""FastAPI application entrypoint for ClearRisk Recover.

Local synthetic-data demonstration only. No authentication, no real
payment/gateway integration, and no financial-enforcement action exists
anywhere in this application. Run with:

    uvicorn app.main:app --reload

Then see http://127.0.0.1:8000/docs for interactive API docs.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import auth, cases, evidence, health, metrics
from app.schemas.api_responses import SYNTHETIC_DATA_NOTICE, ErrorCode

app = FastAPI(
    title="ClearRisk Recover API",
    description="Local synthetic-data demonstration only. No real payment, settlement, or enforcement action exists in this API.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(evidence.router)
app.include_router(metrics.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # exc.detail is already {"error": {...}} for every raise_api_error() call in
    # this codebase; unwrap it so the top-level response body matches the
    # documented error schema exactly (no extra "detail" wrapper key).
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": ErrorCode.INTERNAL_SAFE_ERROR.value, "message": str(exc.detail), "synthetic_data_notice": SYNTHETIC_DATA_NOTICE}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Request validation failed.",
                "details": exc.errors(),
                "synthetic_data_notice": SYNTHETIC_DATA_NOTICE,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": ErrorCode.INTERNAL_SAFE_ERROR.value,
                "message": "An unexpected error occurred. No internal details are exposed.",
                "synthetic_data_notice": SYNTHETIC_DATA_NOTICE,
            }
        },
    )
