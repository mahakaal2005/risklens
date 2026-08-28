"""Tests for scripts/import_merchant_csv.py -- the end-to-end
validate -> score -> persist pipeline for an anonymized merchant-week CSV
import. See docs/PHASE_2_EXTERNAL_DATA_IMPORT_DESIGN.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory
from app.main import app
from scripts.import_merchant_csv import run_import

FIXTURE_CSV_PATH = Path("demo_data/external_import_fixtures/anonymized_merchant_export_demo.csv")


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'test_import.db'}"
    # run_import() calls create_db_engine() with no arguments internally,
    # which reads DATABASE_URL -- set it so both this fixture's own
    # verification queries and run_import() itself hit the same isolated
    # file, never the developer's real clearrisk_recover.db.
    monkeypatch.setenv("DATABASE_URL", db_url)

    engine = create_db_engine(db_url)
    init_db(engine)
    factory = make_session_factory(engine)

    def override_get_db():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield factory
    app.dependency_overrides.clear()


def test_bundled_fixture_imports_successfully(isolated_db):
    report = run_import(FIXTURE_CSV_PATH)
    assert report["validation_passed"] is True
    assert report["row_count"] == 3
    assert report["cases_created"] + report["cases_approved_no_case"] == 3


def test_invalid_csv_is_rejected_and_creates_no_cases(isolated_db, tmp_path):
    row = {
        "merchant_id": "ext_merchant_bad", "week_start": "2025-06-02",
        "merchant_category": "electronics", "merchant_age_days": 900,
        "transaction_count_30d": 420, "transaction_volume_30d": 84000.0,
        "transaction_volume_previous_30d": 82000.0, "transaction_volume_change_30d": 2000.0,
        "refund_count_30d": 8, "refund_rate_30d": 0.019, "refund_rate_previous_30d": 0.018,
        "refund_rate_change_30d": 0.001,
        "chargeback_count_30d": 1, "chargeback_rate_30d": 0.0024, "chargeback_rate_previous_30d": 0.0022,
        "chargeback_rate_change_30d": 0.0002,
        "top_dispute_reason_category": "other", "delivery_evidence_coverage": 0.95,
        "support_ticket_rate": 0.01, "average_support_resolution_time_hours": 6.0,
        "previous_review_outcome": "none",
        "customer_email": "someone@example.com",
    }
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    report = run_import(csv_path)
    assert report["validation_passed"] is False
    assert report["cases_created"] == 0
    assert any("customer_email" in issue for issue in report["validation_issues"])


def test_imported_case_carries_import_specific_notice_via_api(isolated_db):
    from fastapi.testclient import TestClient

    from tests.conftest import make_bearer_headers

    run_import(FIXTURE_CSV_PATH)

    headers = make_bearer_headers(isolated_db, "reviewer", "analyst_demo_001", "Demo Reviewer")
    client = TestClient(app)
    response = client.get("/cases", headers=headers)
    items = response.json()["items"]
    assert items, "expected at least one non-APPROVE imported case"

    detail = client.get(f"/cases/{items[0]['case_id']}", headers=headers).json()
    assert "importing a CSV file" in detail["synthetic_data_notice"]
    assert "synthetic, demonstration-only data" not in detail["synthetic_data_notice"]
