"""Import and score an anonymized merchant-week CSV from an external
source (Phase 2 -- see docs/PHASE_2_EXTERNAL_DATA_IMPORT_DESIGN.md).

Confirmed with the user before implementation:
- The CSV must match RiskLens's own internal raw schema exactly (the
  same columns ml/data_validation.py validates for the synthetic
  dataset, minus the two synthetic-only columns) -- no flexible
  column-name mapping in v1.
- A prohibited/PII-suggestive column, or the presence of
  label_high_loss_next_30d / latent_state_for_demo_only (which a real
  merchant could never honestly provide), rejects the entire file. Never
  silently dropped and imported around.

This module has no dependency on app/ (mirrors every other module in
ml/) -- it validates and scores an already-loaded DataFrame; persistence
into the database is scripts/import_merchant_csv.py's job.
"""

from __future__ import annotations

import re

import pandas as pd

from ml.case_packet import build_case_packet
from ml.data_validation import (
    NON_NEGATIVE_COLUMNS,
    PROHIBITED_FIELD_NAMES,
    RATE_COLUMNS_0_1,
    REQUIRED_COLUMNS as SYNTHETIC_REQUIRED_COLUMNS,
)
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN, compute_feature_frame
from ml.model_utils import ML_FEATURE_COLUMNS, combined_policy
from ml.rules_engine import score_merchant_week

FIXTURE_LABEL = "anonymized_merchant_csv_import_demo_fixture_not_real_merchant_data"

# Overrides build_case_packet()'s built-in synthetic_data_notice, which is
# written for this project's own internally-generated synthetic dataset and
# would be false if applied to genuinely real anonymized merchant data. This
# is the notice actually persisted to the ReviewCase row and shown in the
# dashboard -- not just packet-level metadata -- so it must be accurate for
# whatever the real data source is, not silently inherited from a different
# pipeline. Update this text if this module is ever pointed at real data.
EXTERNAL_IMPORT_DATA_NOTICE = (
    f"This case packet was produced by importing a CSV file labeled '{FIXTURE_LABEL}'. "
    "It does not describe a real merchant, real transaction, or confirmed real-world outcome."
)

# The synthetic dataset's schema minus the two fields a real merchant could
# never honestly provide (they describe this project's own simulation, not
# anything observable in a real merchant's data).
IMPORT_REQUIRED_COLUMNS = [c for c in SYNTHETIC_REQUIRED_COLUMNS if c not in (LABEL_COLUMN, LATENT_STATE_COLUMN)]

# A real anonymized merchant token is not required to look like
# "merchant_demo_0001" (that pattern is specific to this project's own
# synthetic generator) -- just a safe, bounded identifier: no path
# separators, no shell metacharacters, no whitespace.
IMPORT_MERCHANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# ml.data_validation.PROHIBITED_FIELD_NAMES only catches an EXACT column
# name match (e.g. "email"), which a real merchant export would rarely use
# verbatim -- "customer_email", "buyer_phone_number", "cardholder_pan" all
# slip past an exact-match check. This substring list is the actual
# rejection mechanism for real-world column naming; PROHIBITED_FIELD_NAMES
# is kept as an extra exact-match belt-and-suspenders check.
PII_SUBSTRING_KEYWORDS = [
    "email", "phone", "card", "cvv", "pin", "aadhaar", "pan_number",
    "ssn", "account_number", "ifsc", "address", "biometric",
    "device_fingerprint", "customer_name", "cardholder", "upi_id",
    "bank_login", "bank_password",
]


def _find_pii_suspicious_columns(columns) -> set[str]:
    columns_lower = {c: c.lower() for c in columns}
    return {c for c, lower in columns_lower.items() if any(kw in lower for kw in PII_SUBSTRING_KEYWORDS)}


class ExternalImportValidationError(Exception):
    """Raised when the imported CSV fails schema or safety validation. The
    file is rejected in full -- never partially imported around a bad
    column or a bad row."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("External data import validation failed:\n- " + "\n- ".join(issues))


def validate_import_dataframe(df: pd.DataFrame) -> None:
    """Validates an imported merchant-week DataFrame. Raises
    ExternalImportValidationError listing every problem found; never
    repairs or drops rows -- a caller that wants a mapping/data-quality
    report on a failed file should catch this and use `.issues`."""

    issues: list[str] = []

    prohibited_present = PROHIBITED_FIELD_NAMES & set(df.columns)
    pii_suspicious_columns = _find_pii_suspicious_columns(df.columns)
    if prohibited_present:
        issues.append(f"Prohibited/PII-suggestive column(s) present: {sorted(prohibited_present)}")
    if pii_suspicious_columns:
        issues.append(f"Column name(s) suggest PII and are rejected on suspicion: {sorted(pii_suspicious_columns)}")

    forbidden_synthetic_only = {LABEL_COLUMN, LATENT_STATE_COLUMN} & set(df.columns)
    if forbidden_synthetic_only:
        issues.append(
            f"Column(s) describing this project's own simulation must never appear in imported data: "
            f"{sorted(forbidden_synthetic_only)}"
        )

    missing_columns = [c for c in IMPORT_REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    if missing_columns or prohibited_present or pii_suspicious_columns or forbidden_synthetic_only:
        # Cannot safely run the remaining row-level checks without the
        # exact expected column set, and must not proceed past a
        # prohibited-field or synthetic-only-field finding regardless.
        raise ExternalImportValidationError(issues)

    bad_merchant_ids = df.loc[~df["merchant_id"].astype(str).str.match(IMPORT_MERCHANT_ID_PATTERN), "merchant_id"].unique()
    if len(bad_merchant_ids) > 0:
        issues.append(f"merchant_id values are not safe anonymized tokens: {list(bad_merchant_ids)[:5]}")

    parsed_dates = pd.to_datetime(df["week_start"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        issues.append(f"{bad_count} week_start value(s) do not parse as a YYYY-MM-DD date")

    for col in NON_NEGATIVE_COLUMNS:
        if (df[col] < 0).any():
            issues.append(f"Column {col!r} contains negative value(s)")

    for col in RATE_COLUMNS_0_1:
        if ((df[col] < 0) | (df[col] > 1)).any():
            issues.append(f"Column {col!r} contains value(s) outside the expected 0.0-1.0 range")

    duplicate_count = int(df.duplicated(subset=["merchant_id", "week_start"]).sum())
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate (merchant_id, week_start) row(s) found")

    if issues:
        raise ExternalImportValidationError(issues)


def build_mapping_report(df: pd.DataFrame) -> dict:
    """A safe, aggregate-only mapping/data-quality report -- never
    includes a raw row. Callers may build this for both a file that
    passed and one that failed validation (pass validation_error=None or
    the caught exception's .issues)."""
    columns_found = [c for c in IMPORT_REQUIRED_COLUMNS if c in df.columns]
    columns_missing = [c for c in IMPORT_REQUIRED_COLUMNS if c not in df.columns]
    return {
        "fixture_label": FIXTURE_LABEL,
        "row_count": int(len(df)),
        "unique_merchant_count": int(df["merchant_id"].nunique()) if "merchant_id" in df.columns else None,
        "week_start_range": (
            {"min": str(df["week_start"].min()), "max": str(df["week_start"].max())}
            if "week_start" in df.columns and len(df) > 0
            else None
        ),
        "required_columns_found": columns_found,
        "required_columns_missing": columns_missing,
        "prohibited_fields_present": sorted(PROHIBITED_FIELD_NAMES & set(df.columns)),
        "pii_suspicious_columns_present": sorted(_find_pii_suspicious_columns(df.columns)),
        "synthetic_only_fields_present": sorted({LABEL_COLUMN, LATENT_STATE_COLUMN} & set(df.columns)),
    }


def score_import_rows(
    df: pd.DataFrame,
    pipeline,
    rules_config: dict,
    threshold: float,
    model_version: str | None,
    rules_version: str,
) -> list[dict]:
    """Scores every already-validated row and returns one safe case packet
    per row, via the exact same build_case_packet() used for the
    synthetic demo cases -- no separate/duplicated packet-building logic
    for imported data."""
    records = df.to_dict(orient="records")
    X = compute_feature_frame(df)[ML_FEATURE_COLUMNS]
    ml_probs = pipeline.predict_proba(X)[:, 1] if pipeline is not None else [None] * len(df)

    packets = []
    for i, record in enumerate(records):
        rules_result = score_merchant_week(record, rules_config)
        ml_probability = float(ml_probs[i]) if ml_probs[i] is not None else None
        decision = combined_policy(
            ml_probability if ml_probability is not None else 0.0,
            threshold,
            set(rules_result["triggered_rules"]),
        )
        packet = build_case_packet(
            record=record,
            rules_result=rules_result,
            ml_probability=ml_probability,
            selected_threshold=threshold,
            model_version=model_version,
            rules_version=rules_version,
            combined_decision=decision,
            pipeline=pipeline,
        )
        # Overrides the packet's built-in synthetic-data notice (see
        # EXTERNAL_IMPORT_DATA_NOTICE above) -- this is the actual text
        # that gets persisted to the ReviewCase row and shown in the
        # dashboard, so it must describe the real data source.
        packet["identification"]["synthetic_data_notice"] = EXTERNAL_IMPORT_DATA_NOTICE
        packet["identification"]["data_source"] = "external_csv_import"
        packet["identification"]["import_fixture_label"] = FIXTURE_LABEL
        packets.append(packet)
    return packets
