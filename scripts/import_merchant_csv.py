"""Imports an anonymized merchant-week CSV, validates it, scores every row
through the existing rules+model pipeline, and persists non-APPROVE rows
as real review cases -- exactly the same workflow the synthetic demo cases
go through. See docs/PHASE_2_EXTERNAL_DATA_IMPORT_DESIGN.md.

The bundled example (demo_data/external_import_fixtures/anonymized_merchant_export_demo.csv)
is a synthetic fixture standing in for a real willing merchant's export --
labeled throughout as exactly that, never claimed to be real data.

Usage:
    python3 scripts/import_merchant_csv.py [--csv-path PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.services.case_service import create_case_from_packet
from ml.external_data_import import (
    ExternalImportValidationError,
    build_mapping_report,
    score_import_rows,
    validate_import_dataframe,
)
from ml.rules_engine import DEFAULT_RULES_PATH, load_rules_config

DEFAULT_CSV_PATH = Path("demo_data/external_import_fixtures/anonymized_merchant_export_demo.csv")
DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_REPORT_PATH = Path("ml/artifacts/external_import_report.json")


def run_import(csv_path: Path, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    df = pd.read_csv(csv_path)

    try:
        validate_import_dataframe(df)
        validation_passed = True
        validation_issues: list[str] = []
    except ExternalImportValidationError as exc:
        validation_passed = False
        validation_issues = exc.issues

    report = build_mapping_report(df)
    report["validation_passed"] = validation_passed
    report["validation_issues"] = validation_issues

    if not validation_passed:
        report["cases_created"] = 0
        report["cases_approved_no_case"] = 0
        return report

    model_path = artifact_dir / f"logistic_regression_v{DEFAULT_MODEL_VERSION}.joblib"
    metadata_path = artifact_dir / f"logistic_regression_v{DEFAULT_MODEL_VERSION}_metadata.json"
    pipeline = None
    threshold = 0.5
    if model_path.exists() and metadata_path.exists():
        pipeline = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            threshold = json.load(f)["selected_threshold"]
    else:
        report["model_status"] = "not_available -- scored in rules-only degraded mode"

    rules_config = load_rules_config(DEFAULT_RULES_PATH)
    packets = score_import_rows(
        df, pipeline, rules_config, threshold, DEFAULT_MODEL_VERSION, rules_config["version"],
    )

    engine = create_db_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    created = 0
    approved_no_case = 0
    for packet in packets:
        with session_scope(session_factory) as session:
            case, _audit_events = create_case_from_packet(session, packet)
            if case is None:
                approved_no_case += 1
            else:
                created += 1

    report["cases_created"] = created
    report["cases_approved_no_case"] = approved_no_case
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and score an anonymized merchant-week CSV.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    report = run_import(Path(args.csv_path), Path(args.artifact_dir))

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if not report["validation_passed"]:
        print(f"Import REJECTED -- {len(report['validation_issues'])} issue(s) found:")
        for issue in report["validation_issues"]:
            print(f"  - {issue}")
    else:
        print(f"Import validated: {report['row_count']} row(s), {report['unique_merchant_count']} merchant(s).")
        print(f"Cases created: {report['cases_created']} (APPROVE, no case: {report['cases_approved_no_case']})")
    print(f"Full mapping/data-quality report written to {report_path}")


if __name__ == "__main__":
    main()
