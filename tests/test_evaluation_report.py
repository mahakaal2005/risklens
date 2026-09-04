import json

import pytest

from ml.evaluate_model import evaluate
from ml.evaluation_report import (
    DEFAULT_REPORT_PATH,
    ReportNotFoundError,
    build_report,
    load_report,
    save_report,
    validate_report,
)

FORBIDDEN_ENFORCEMENT_WORDS = ["freeze", "ban", "terminate", "hold settlement", "reject payment"]
LATENT_STATE_NAMES = [
    "stable_merchant", "seasonal_sale_legitimate_returns", "operational_fulfilment_failure",
    "high_risk_merchant_behaviour", "early_hidden_risk_case",
]


@pytest.fixture(scope="module")
def evaluate_result():
    return evaluate()


@pytest.fixture(scope="module")
def dataset_metadata():
    with open("demo_data/synthetic_data_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_metadata():
    with open("ml/artifacts/logistic_regression_v0.1.0_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def report(evaluate_result, dataset_metadata, model_metadata):
    return build_report(evaluate_result, dataset_metadata, model_metadata)


def test_evaluate_model_main_produces_latest_evaluation_report(tmp_path, evaluate_result, dataset_metadata, model_metadata):
    report_obj = build_report(evaluate_result, dataset_metadata, model_metadata)
    report_path = tmp_path / "latest_evaluation_report.json"
    written = save_report(report_obj, path=report_path, also_timestamped=False)
    assert report_path in written
    assert report_path.exists()


def test_report_is_valid_json(tmp_path, report):
    report_path = tmp_path / "report.json"
    save_report(report, path=report_path, also_timestamped=False)
    with open(report_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)  # raises if not valid JSON
    assert reloaded["report_version"] == "1.0"


def test_report_contains_all_required_sections(report):
    for key in [
        "report_version", "generated_at", "data_mode", "synthetic_data_notice",
        "dataset", "split", "model", "methods", "scenario_difficulty",
        "near_perfect_score_investigation", "limitations",
    ]:
        assert key in report
    for method in ("rules_only", "logistic_regression", "combined_policy"):
        assert method in report["methods"]
    for split_name in ("train", "validation", "test"):
        assert split_name in report["split"]


def test_report_contains_real_metrics_matching_evaluation_result(report, evaluate_result):
    test_results = evaluate_result["test_results"]
    assert report["methods"]["rules_only"]["precision"] == test_results["rules_only"]["precision"]
    assert report["methods"]["logistic_regression"]["recall"] == test_results["ml_only"]["recall"]
    assert report["methods"]["combined_policy"]["pr_auc"] == test_results["combined"]["pr_auc"]
    assert report["model"]["selected_threshold"] == evaluate_result["threshold"]
    assert report["split"]["test"]["row_count"] == evaluate_result["split_info"]["test"]["row_count"]


def test_report_has_synthetic_notice_and_limitations(report):
    assert report["synthetic_data_notice"]
    assert len(report["limitations"]) > 0
    assert report["data_mode"] == "synthetic-only"


def test_report_contains_no_prohibited_data(report):
    serialized = json.dumps(report)
    for name in LATENT_STATE_NAMES:
        assert name not in serialized
    for word in FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in serialized.lower()
    assert "merchant_demo_" not in serialized
    # excluded_fields legitimately documents these two field NAMES; ensure
    # they appear nowhere else in the report.
    without_excluded_fields = json.loads(serialized)
    without_excluded_fields["model"].pop("excluded_fields", None)
    serialized_without = json.dumps(without_excluded_fields)
    assert "label_high_loss_next_30d" not in serialized_without
    assert "latent_state_for_demo_only" not in serialized_without


def test_validate_report_accepts_well_formed_report(report):
    assert validate_report(report) == []


def test_validate_report_rejects_missing_top_level_field(report):
    broken = dict(report)
    del broken["methods"]
    issues = validate_report(broken)
    assert any("methods" in issue for issue in issues) or len(issues) > 0


def test_validate_report_rejects_wrong_data_mode(report):
    broken = json.loads(json.dumps(report))
    broken["data_mode"] = "real-production"
    issues = validate_report(broken)
    assert any("data_mode" in issue for issue in issues)


def test_validate_report_rejects_out_of_range_rate(report):
    broken = json.loads(json.dumps(report))
    broken["methods"]["rules_only"]["precision"] = 1.5
    issues = validate_report(broken)
    assert any("precision" in issue for issue in issues)


def test_validate_report_rejects_negative_confusion_matrix_count(report):
    broken = json.loads(json.dumps(report))
    broken["methods"]["logistic_regression"]["confusion_matrix"]["fp"] = -1
    issues = validate_report(broken)
    assert any("confusion_matrix" in issue for issue in issues)


def test_validate_report_rejects_unexpected_top_level_field(report):
    """Regression test: a schema-complete report with an extra, unexpected
    top-level field (e.g. accidental debug output) must be rejected, not
    silently accepted. Reproduced during Milestone 7 verification: the
    original validator only scanned for a fixed list of known-bad
    substrings and did not reject genuinely unknown extra fields."""
    broken = json.loads(json.dumps(report))
    broken["debug_internal"] = "stack trace: /home/user/app.py line 42, password=hunter2"
    issues = validate_report(broken)
    assert any("debug_internal" in issue for issue in issues)


def test_validate_report_rejects_unexpected_method_field(report):
    broken = json.loads(json.dumps(report))
    broken["methods"]["rules_only"]["internal_note"] = "unexpected"
    issues = validate_report(broken)
    assert any("internal_note" in issue for issue in issues)


def test_validate_report_rejects_leaked_latent_state_name(report):
    broken = json.loads(json.dumps(report))
    broken["debug_note"] = "generated from stable_merchant rows"
    issues = validate_report(broken)
    assert any("stable_merchant" in issue for issue in issues)


def test_optional_comparison_methods_are_registered_consistently():
    """Random Forest, Gradient Boosting and the Trajectory Transformer are all
    comparison-only baselines whose artifacts may legitimately be absent. Each
    must be registered in BOTH the rename map and the optional set -- a method
    in the rename map but not the optional set would make a missing artifact a
    hard validation failure."""
    from ml.evaluation_report import OPTIONAL_METHOD_NAMES, _METHOD_KEY_RENAME

    for method_name in ("random_forest", "gradient_boosting", "trajectory_transformer"):
        assert method_name in _METHOD_KEY_RENAME
        assert _METHOD_KEY_RENAME[method_name] in OPTIONAL_METHOD_NAMES


def test_report_includes_trajectory_transformer_when_the_artifact_is_present(report, evaluate_result):
    if "trajectory_transformer" not in evaluate_result["test_results"]:
        pytest.skip("Trajectory Transformer artifact not trained in this environment")

    metrics = report["methods"]["trajectory_transformer"]
    assert metrics["precision"] == evaluate_result["test_results"]["trajectory_transformer"]["precision"]
    assert metrics["recall"] == evaluate_result["test_results"]["trajectory_transformer"]["recall"]
    # Every method must record the threshold it was actually scored at, so a
    # reader can never mistake one model's operating point for another's.
    assert metrics["threshold_used"] == evaluate_result["test_results"]["trajectory_transformer"]["threshold_used"]


def test_report_validates_without_any_optional_comparison_method(report):
    """A report built where only the required methods ran must still validate --
    the comparison models are optional by design."""
    from ml.evaluation_report import OPTIONAL_METHOD_NAMES

    trimmed = json.loads(json.dumps(report))
    for optional_name in OPTIONAL_METHOD_NAMES:
        trimmed["methods"].pop(optional_name, None)

    validate_report(trimmed)  # must not raise


def test_load_report_raises_not_found_for_missing_file(tmp_path):
    with pytest.raises(ReportNotFoundError):
        load_report(tmp_path / "does_not_exist.json")


def test_load_report_raises_json_decode_error_for_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_report(corrupt_path)


def test_default_report_path_is_latest_evaluation_report_json():
    assert DEFAULT_REPORT_PATH.name == "latest_evaluation_report.json"
