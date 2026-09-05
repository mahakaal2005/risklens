"""FastAPI application entrypoint for RiskLens.

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
    title="RiskLens API",
    description="Local synthetic-data demonstration only. No real payment, settlement, or enforcement action exists in this API.",
    version="0.1.0",
)


# FastAPI's own interactive docs load Swagger UI's JS/CSS from a public CDN,
# so a strict same-origin CSP would silently break /docs and /redoc in the
# browser. Those paths keep X-Content-Type-Options/X-Frame-Options but skip
# the CSP header; every other response gets the full set.
_CSP_EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Basic security headers on every response -- part of the project's
    standing pre-deploy security checklist. Strict-Transport-Security is
    deliberately omitted: this API serves plain HTTP on localhost, and HSTS
    over an insecure connection is meaningless (or actively wrong) rather
    than protective."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path not in _CSP_EXEMPT_PATHS:
        response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


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
