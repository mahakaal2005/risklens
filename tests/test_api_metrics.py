import json

import pytest
from fastapi.testclient import TestClient

import app.api.routes.metrics as metrics_module
from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory
from app.main import app
from ml.evaluate_model import evaluate
from ml.evaluation_report import build_report, save_report
from tests.conftest import make_bearer_headers


@pytest.fixture()
def client(tmp_path):
    # /metrics now requires authentication, which requires a DB session --
    # this must stay isolated from the developer's real clearrisk_recover.db,
    # exactly like every other test file's client fixture (see
    # app/api/dependencies.py's docstring).
    db_path = tmp_path / "test_api_metrics.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    init_db(engine)
    session_factory = make_session_factory(engine)

    def override_get_db():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Any authenticated role may read /metrics -- risk_manager is the most
    # representative choice since it is the read-only reporting role.
    headers = make_bearer_headers(session_factory, "risk_manager", "riskmanager_demo_001", "Demo Risk Manager")
    test_client = TestClient(app)
    test_client.headers.update(headers)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def real_report():
    result = evaluate()
    with open("demo_data/synthetic_data_metadata.json", "r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)
    with open("ml/artifacts/logistic_regression_v0.1.0_metadata.json", "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    return build_report(result, dataset_metadata, model_metadata)


def test_metrics_returns_not_available_when_artifact_missing(client, monkeypatch, tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(metrics_module, "DEFAULT_REPORT_PATH", missing_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_available"
    assert body["error_code"] == "METRICS_NOT_AVAILABLE"
    assert body["generation_command"] == "python3 -m ml.evaluate_model"
    assert body["synthetic_data_notice"]


def test_metrics_returns_not_available_for_corrupt_json(client, monkeypatch, tmp_path):
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not valid json at all")
    monkeypatch.setattr(metrics_module, "DEFAULT_REPORT_PATH", corrupt_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_available"
    assert body["error_code"] == "METRICS_ARTIFACT_INVALID"
    # Must not leak the parse error or the file path.
    assert str(corrupt_path) not in json.dumps(body)
    assert "JSONDecodeError" not in json.dumps(body)


def test_metrics_returns_not_available_for_schema_invalid_report(client, monkeypatch, tmp_path):
    invalid_report = {"report_version": "1.0", "data_mode": "synthetic-only"}  # missing required sections
    report_path = tmp_path / "invalid_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(invalid_report, f)
    monkeypatch.setattr(metrics_module, "DEFAULT_REPORT_PATH", report_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_available"
    assert body["error_code"] == "METRICS_ARTIFACT_INVALID"


def test_metrics_returns_available_summary_when_artifact_valid(client, monkeypatch, tmp_path, real_report):
    report_path = tmp_path / "latest_evaluation_report.json"
    save_report(real_report, path=report_path, also_timestamped=False)
    monkeypatch.setattr(metrics_module, "DEFAULT_REPORT_PATH", report_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["error_code"] is None
    assert body["dataset_seed"] == real_report["dataset"]["seed"]
    assert body["selected_threshold"] == real_report["model"]["selected_threshold"]
    assert body["rules_only_metrics"]["precision"] == real_report["methods"]["rules_only"]["precision"]
    assert body["logistic_regression_metrics"]["recall"] == real_report["methods"]["logistic_regression"]["recall"]
    assert body["combined_policy_metrics"]["pr_auc"] == real_report["methods"]["combined_policy"]["pr_auc"]
    assert "do not prove real-world chargeback-risk performance" in body["limitation"]

    assert body["scenario_difficulty"] == real_report["scenario_difficulty"]
    if "random_forest" in real_report["methods"]:
        assert body["random_forest_metrics"]["precision"] == real_report["methods"]["random_forest"]["precision"]
    if "gradient_boosting" in real_report["methods"]:
        assert body["gradient_boosting_metrics"]["precision"] == real_report["methods"]["gradient_boosting"]["precision"]

    serialized = json.dumps(body)
    for prohibited in ["stable_merchant", "seasonal_sale_legitimate_returns", "operational_fulfilment_failure", "high_risk_merchant_behaviour"]:
        assert prohibited not in serialized


def test_metrics_never_trains_or_evaluates_on_request(client, monkeypatch, tmp_path):
    import ml.train_baseline_model as train_module
    import ml.evaluate_model as evaluate_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("metrics endpoint must never call train() or evaluate()")

    monkeypatch.setattr(train_module, "train", fail_if_called)
    monkeypatch.setattr(evaluate_module, "evaluate", fail_if_called)

    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(metrics_module, "DEFAULT_REPORT_PATH", missing_path)

    response = client.get("/metrics")
    assert response.status_code == 200
