"""Milestone 4: build a reviewer-ready, safe, explainable case packet from a
merchant-week's rules + model outcome.

The case packet is an in-memory/JSON demonstration artifact only. It never
creates a real case in a database and never takes action against a
merchant. label_high_loss_next_30d and latent_state_for_demo_only are never
read by this module (the record passed in should already exclude them; if
present, they are dropped defensively and never appear in output).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd

from ml.explain_cases import build_analyst_summary, build_trend_values, build_uncertainty_statement, compute_top_factors
from ml.features import EXCLUDED_INPUT_COLUMNS, LABEL_COLUMN, LATENT_STATE_COLUMN, compute_features
from ml.model_utils import ML_FEATURE_COLUMNS

SYNTHETIC_DATA_NOTICE = (
    "This case packet is generated from synthetic, demonstration-only data. It does not "
    "describe a real merchant, real transaction, or confirmed real-world outcome."
)

FORBIDDEN_MERCHANT_SAFE_TERMS = ["fraud confirmed", "ban", "freeze", "terminate", "latent", "probability"]

# rule_id -> evidence checklist items contributed when that rule triggers.
EVIDENCE_ITEMS_BY_RULE = {
    "REFUND_RATE_SPIKE": ["Recent refund/cancellation records", "Explanation for any recent product, pricing, or listing changes"],
    "CHARGEBACK_RATE_SPIKE": ["Chargeback/dispute reason breakdown", "Proof of delivery or service fulfilment for disputed orders"],
    "EVIDENCE_COVERAGE_GAP": ["Fulfilment/delivery proof for a sample of disputed orders"],
    "SUPPORT_OPERATIONAL_STRESS": ["Customer-support response records", "Plan to address recurring operational issues"],
    "COMBINED_LOSS_SIGNAL": ["Plan to address recurring operational issues"],
}
VOLUME_CHANGE_EVIDENCE_THRESHOLD = 0.30

MERCHANT_SAFE_REASON_CATEGORIES = {
    "REFUND_RATE_SPIKE": "refund pattern review",
    "CHARGEBACK_RATE_SPIKE": "chargeback pattern review",
    "EVIDENCE_COVERAGE_GAP": "fulfilment evidence review",
    "SUPPORT_OPERATIONAL_STRESS": "support pattern review",
    "COMBINED_LOSS_SIGNAL": "combined account review",
}


def _case_preview_id(merchant_id: str, week_start: str) -> str:
    digest = hashlib.sha256(f"{merchant_id}:{week_start}".encode("utf-8")).hexdigest()[:16]
    return f"case_preview_{digest}"


def build_evidence_checklist(triggered_rule_ids: list[str], features: dict) -> list[str]:
    items: list[str] = []
    for rule_id in triggered_rule_ids:
        for item in EVIDENCE_ITEMS_BY_RULE.get(rule_id, []):
            if item not in items:
                items.append(item)

    volume_change = features.get("transaction_volume_change")
    if triggered_rule_ids and volume_change is not None and abs(volume_change) >= VOLUME_CHANGE_EVIDENCE_THRESHOLD:
        item = "Explanation for the recent transaction-volume change"
        if item not in items:
            items.append(item)

    return items


def build_merchant_safe_reason_category(triggered_rule_ids: list[str]) -> str:
    if not triggered_rule_ids:
        return "no review signal"
    for rule_id in ("COMBINED_LOSS_SIGNAL", "CHARGEBACK_RATE_SPIKE", "EVIDENCE_COVERAGE_GAP", "SUPPORT_OPERATIONAL_STRESS", "REFUND_RATE_SPIKE"):
        if rule_id in triggered_rule_ids:
            return MERCHANT_SAFE_REASON_CATEGORIES[rule_id]
    return "general account review"


def build_merchant_safe_explanation(triggered_rule_ids: list[str], rule_results: list, features: dict) -> dict:
    reason_category = build_merchant_safe_reason_category(triggered_rule_ids)
    merchant_reasons = [r.merchant_safe_explanation for r in rule_results if r.triggered]

    return {
        "reason_category": reason_category,
        "review_signal_statement": (
            "This is a review signal only. It is not a final determination of fraud, and no "
            "action has been taken against your account."
        ),
        "reasons": merchant_reasons,
        "suggested_evidence": build_evidence_checklist(triggered_rule_ids, features),
        "appeal_placeholder": (
            "If you believe this review signal does not reflect your business accurately, you "
            "may submit a written explanation and supporting evidence for a reviewer to consider. "
            "(Simulated appeal flow -- not yet implemented in this milestone.)"
        ),
    }


def build_audit_preview_events(recommendation: str, generated_at: str) -> list[dict]:
    events = [
        {"event_type": "ASSESSMENT_GENERATED", "timestamp": generated_at, "preview_only": True},
        {"event_type": "EXPLANATION_GENERATED", "timestamp": generated_at, "preview_only": True},
    ]
    if recommendation != "APPROVE":
        events.append({"event_type": "REVIEW_CASE_RECOMMENDED", "timestamp": generated_at, "preview_only": True})
    if recommendation == "REQUEST_EVIDENCE":
        events.append({"event_type": "EVIDENCE_REQUEST_RECOMMENDED", "timestamp": generated_at, "preview_only": True})
    if recommendation == "MANUAL_REVIEW_REQUIRED":
        events.append({"event_type": "MANUAL_REVIEW_RECOMMENDED", "timestamp": generated_at, "preview_only": True})
    if recommendation == "ESCALATE_TO_COMPLIANCE":
        events.append({"event_type": "ESCALATION_RECOMMENDED", "timestamp": generated_at, "preview_only": True})
    for event in events:
        event["note"] = "Preview event only -- not persisted to any database or audit log in this milestone."
    return events


def build_case_packet(
    record: dict,
    rules_result: dict,
    ml_probability: float | None,
    selected_threshold: float,
    model_version: str | None,
    rules_version: str,
    combined_decision,
    pipeline=None,
    prior_review_context: dict | None = None,
) -> dict:
    """record must be the raw merchant-week fields (label/latent-state, if
    present, are stripped defensively and never reach the packet)."""
    record = {k: v for k, v in record.items() if k not in EXCLUDED_INPUT_COLUMNS}
    merchant_id = record["merchant_id"]
    week_start = record["week_start"]
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    features = compute_features(record)
    rule_results = rules_result["rule_results"]
    triggered_rule_ids = rules_result["triggered_rules"]

    degraded_mode = ml_probability is None or pipeline is None
    top_factors = []
    if not degraded_mode:
        row_df = pd.DataFrame([features])[ML_FEATURE_COLUMNS]
        top_factors = compute_top_factors(pipeline, row_df)

    analyst_summary = build_analyst_summary(triggered_rule_ids, top_factors, combined_decision.recommendation, degraded_mode)
    uncertainty_statement = build_uncertainty_statement(ml_probability, degraded_mode)
    trend_values = build_trend_values(record, features)

    packet = {
        "identification": {
            "case_preview_id": _case_preview_id(merchant_id, week_start),
            "merchant_id": merchant_id,
            "week_start": week_start,
            "generated_at": generated_at,
            "synthetic_data_notice": SYNTHETIC_DATA_NOTICE,
        },
        "assessment": {
            "model_probability": round(float(ml_probability), 4) if ml_probability is not None else None,
            "selected_threshold": selected_threshold,
            "risk_signal_intensity": combined_decision.risk_signal_intensity,
            "rules_only_score": rules_result["risk_score"],
            "triggered_rules": triggered_rule_ids,
            "recommendation": combined_decision.recommendation,
            "policy_explanation": combined_decision.policy_explanation,
            "model_version": model_version,
            "rules_version": rules_version,
            "degraded_mode": degraded_mode,
        },
        "analyst_explanation": {
            "summary": analyst_summary,
            "top_model_factors": top_factors,
            "triggered_rule_explanations": [
                {"rule_id": r.rule_id, "severity": r.severity, "explanation": r.analyst_explanation}
                for r in rule_results if r.triggered
            ],
            "trend_values": trend_values,
            "uncertainty_statement": uncertainty_statement,
        },
        "merchant_safe_explanation": build_merchant_safe_explanation(triggered_rule_ids, rule_results, features),
        "evidence_checklist": build_evidence_checklist(triggered_rule_ids, features),
        "audit_preview_events": build_audit_preview_events(combined_decision.recommendation, generated_at),
    }

    if prior_review_context is not None:
        packet["prior_simulated_review_context"] = prior_review_context

    _assert_packet_is_safe(packet)
    return packet


def _assert_packet_is_safe(packet: dict) -> None:
    serialized = json.dumps(packet)
    if LABEL_COLUMN in serialized or LATENT_STATE_COLUMN in serialized:
        raise ValueError("Case packet must never reference the target label or latent state.")

    merchant_safe_text = json.dumps(packet["merchant_safe_explanation"]).lower()
    for term in FORBIDDEN_MERCHANT_SAFE_TERMS:
        if term in merchant_safe_text:
            raise ValueError(f"Merchant-safe explanation must not contain forbidden term: {term}")
