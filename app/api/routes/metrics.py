"""Static evaluation-metrics endpoint.

Reads the persisted evaluation-report artifact from disk (see
ml/evaluation_report.py); never retrains or re-scores during an API
request. Always returns HTTP 200 -- a missing or invalid artifact is a
valid, expected state for this local prototype, not a server error.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
from app.db.models import User
from app.schemas.api_responses import MetricsResponse
from ml.evaluation_report import DEFAULT_REPORT_PATH, ReportNotFoundError, load_report, validate_report

router = APIRouter(tags=["metrics"])

GENERATION_COMMAND = "python3 -m ml.evaluate_model"

NOT_AVAILABLE_MESSAGE = (
    f"No saved evaluation report was found. Run `{GENERATION_COMMAND}` to generate one."
)

INVALID_MESSAGE = (
    f"A saved evaluation report exists but failed validation. Run `{GENERATION_COMMAND}` "
    "to regenerate a valid report."
)


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics_route(user: User = Depends(get_current_user)) -> MetricsResponse:
    try:
        report = load_report(DEFAULT_REPORT_PATH)
    except ReportNotFoundError:
        return MetricsResponse(
            status="not_available",
            error_code="METRICS_NOT_AVAILABLE",
            message=NOT_AVAILABLE_MESSAGE,
            generation_command=GENERATION_COMMAND,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return MetricsResponse(
            status="not_available",
            error_code="METRICS_ARTIFACT_INVALID",
            message=INVALID_MESSAGE,
            generation_command=GENERATION_COMMAND,
        )

    issues = validate_report(report)
    if issues:
        return MetricsResponse(
            status="not_available",
            error_code="METRICS_ARTIFACT_INVALID",
            message=INVALID_MESSAGE,
            generation_command=GENERATION_COMMAND,
        )

    return MetricsResponse(
        status="available",
        message="Evaluation report loaded from the local artifact.",
        dataset_seed=report["dataset"].get("seed"),
        dataset_version=report["dataset"].get("generator_version"),
        held_out_test_date_range={"start": report["split"]["test"]["date_start"], "end": report["split"]["test"]["date_end"]},
        selected_threshold=report["model"].get("selected_threshold"),
        rules_only_metrics=report["methods"].get("rules_only"),
        logistic_regression_metrics=report["methods"].get("logistic_regression"),
        random_forest_metrics=report["methods"].get("random_forest"),
        gradient_boosting_metrics=report["methods"].get("gradient_boosting"),
        trajectory_transformer_metrics=report["methods"].get("trajectory_transformer"),
        cost_analysis=report.get("cost_analysis") or None,
        calibration=report.get("calibration") or None,
        combined_policy_metrics=report["methods"].get("combined_policy"),
        scenario_difficulty=report.get("scenario_difficulty"),
        near_perfect_investigation_status=report["near_perfect_score_investigation"].get("status"),
        limitation=report.get("synthetic_data_notice", MetricsResponse.model_fields["limitation"].default),
    )
