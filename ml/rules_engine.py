"""Transparent rules engine for ClearRisk Recover, loaded from rules/risk_rules.yaml.

Produces evidence and risk signals for human review. It never makes an
enforcement decision, and its output recommendation is always one of the
five allowed values (APPROVE, ALLOW_WITH_MONITORING, REQUEST_EVIDENCE,
MANUAL_REVIEW_REQUIRED, ESCALATE_TO_COMPLIANCE).

All thresholds are read from the YAML config -- nothing here hard-codes a
numeric threshold in Python, per the Milestone 2 requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ml.features import EXCLUDED_INPUT_COLUMNS, compute_features

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "risk_rules.yaml"

ALLOWED_RECOMMENDATIONS = {
    "APPROVE",
    "ALLOW_WITH_MONITORING",
    "REQUEST_EVIDENCE",
    "MANUAL_REVIEW_REQUIRED",
    "ESCALATE_TO_COMPLIANCE",
}


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    rule_version: str
    triggered: bool
    severity: str
    score_contribution: int
    analyst_explanation: str
    merchant_safe_explanation: str
    suggested_evidence: list[str]
    recommended_action: str
    input_snapshot: dict = field(default_factory=dict)


def load_rules_config(path: Path = DEFAULT_RULES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _snapshot(merged: dict, fields: list[str]) -> dict:
    return {f: merged.get(f) for f in fields if f not in EXCLUDED_INPUT_COLUMNS}


def _evaluate_rate_spike(merged: dict, rule_cfg: dict, prefix: str) -> RuleResult:
    current = merged.get(f"{prefix}_rate_30d")
    previous = merged.get(f"{prefix}_rate_previous_30d")
    change = merged.get(f"{prefix}_rate_change")

    thresholds = rule_cfg["thresholds"]
    triggered = False
    if current is not None and previous is not None and change is not None:
        if change >= thresholds["absolute_increase_min"]:
            triggered = True
        elif previous > 0 and current >= previous * thresholds["relative_multiple_min"]:
            triggered = True

    values = {
        f"{prefix}_rate_previous_30d_pct": _pct(previous),
        f"{prefix}_rate_30d_pct": _pct(current),
        f"{prefix}_rate_change_pct": _pct(change),
    }
    severity = rule_cfg["severity"]
    return RuleResult(
        rule_id=rule_cfg["rule_id"],
        rule_version=rule_cfg["version"],
        triggered=triggered,
        severity=severity,
        score_contribution=0,
        analyst_explanation=rule_cfg["analyst_explanation_template"].strip().format(**values) if triggered else "",
        merchant_safe_explanation=rule_cfg["merchant_safe_explanation"].strip() if triggered else "",
        suggested_evidence=list(rule_cfg["suggested_evidence"]) if triggered else [],
        recommended_action=rule_cfg["recommended_action"] if triggered else "APPROVE",
        input_snapshot=_snapshot(merged, rule_cfg["input_fields"]),
    )


def _evaluate_evidence_gap(merged: dict, rule_cfg: dict) -> RuleResult:
    gap = merged.get("delivery_evidence_gap")
    refund_change = merged.get("refund_rate_change") or 0.0
    chargeback_change = merged.get("chargeback_rate_change") or 0.0
    thresholds = rule_cfg["thresholds"]

    triggered = False
    if gap is not None and gap >= thresholds["evidence_gap_min"]:
        supporting = (
            refund_change >= thresholds["supporting_refund_change_min"]
            or chargeback_change >= thresholds["supporting_chargeback_change_min"]
        )
        triggered = supporting if thresholds["requires_supporting_signal"] else True

    values = {"delivery_evidence_gap_pct": _pct(gap)}
    return RuleResult(
        rule_id=rule_cfg["rule_id"],
        rule_version=rule_cfg["version"],
        triggered=triggered,
        severity=rule_cfg["severity"],
        score_contribution=0,
        analyst_explanation=rule_cfg["analyst_explanation_template"].strip().format(**values) if triggered else "",
        merchant_safe_explanation=rule_cfg["merchant_safe_explanation"].strip() if triggered else "",
        suggested_evidence=list(rule_cfg["suggested_evidence"]) if triggered else [],
        recommended_action=rule_cfg["recommended_action"] if triggered else "APPROVE",
        input_snapshot=_snapshot(merged, rule_cfg["input_fields"]),
    )


def _evaluate_support_stress(merged: dict, rule_cfg: dict) -> RuleResult:
    ticket_rate = merged.get("support_ticket_rate")
    resolution_hours = merged.get("average_support_resolution_time_hours")
    refund_change = merged.get("refund_rate_change") or 0.0
    chargeback_change = merged.get("chargeback_rate_change") or 0.0
    thresholds = rule_cfg["thresholds"]

    triggered = False
    if (
        ticket_rate is not None
        and resolution_hours is not None
        and ticket_rate >= thresholds["support_ticket_rate_min"]
        and resolution_hours >= thresholds["resolution_hours_min"]
    ):
        supporting = (
            refund_change >= thresholds["supporting_refund_change_min"]
            or chargeback_change >= thresholds["supporting_chargeback_change_min"]
        )
        triggered = supporting if thresholds["requires_supporting_signal"] else True

    values = {
        "support_ticket_rate_pct": _pct(ticket_rate),
        "average_support_resolution_time_hours": resolution_hours,
    }
    return RuleResult(
        rule_id=rule_cfg["rule_id"],
        rule_version=rule_cfg["version"],
        triggered=triggered,
        severity=rule_cfg["severity"],
        score_contribution=0,
        analyst_explanation=rule_cfg["analyst_explanation_template"].strip().format(**values) if triggered else "",
        merchant_safe_explanation=rule_cfg["merchant_safe_explanation"].strip() if triggered else "",
        suggested_evidence=list(rule_cfg["suggested_evidence"]) if triggered else [],
        recommended_action=rule_cfg["recommended_action"] if triggered else "APPROVE",
        input_snapshot=_snapshot(merged, rule_cfg["input_fields"]),
    )


def _evaluate_combined(merged: dict, rule_cfg: dict, triggered_by_id: dict[str, bool]) -> RuleResult:
    pairs = rule_cfg["thresholds"]["required_co_trigger_pairs"]
    co_triggered_pairs = [pair for pair in pairs if all(triggered_by_id.get(rid, False) for rid in pair)]
    triggered = len(co_triggered_pairs) > 0
    co_triggered_ids = sorted({rid for pair in co_triggered_pairs for rid in pair})

    values = {"co_triggered_rule_ids": ", ".join(co_triggered_ids)}
    return RuleResult(
        rule_id=rule_cfg["rule_id"],
        rule_version=rule_cfg["version"],
        triggered=triggered,
        severity=rule_cfg["severity"],
        score_contribution=0,
        analyst_explanation=rule_cfg["analyst_explanation_template"].strip().format(**values) if triggered else "",
        merchant_safe_explanation=rule_cfg["merchant_safe_explanation"].strip() if triggered else "",
        suggested_evidence=list(rule_cfg["suggested_evidence"]) if triggered else [],
        recommended_action=rule_cfg["recommended_action"] if triggered else "APPROVE",
        input_snapshot={},
    )


_RULE_EVALUATORS = {
    "REFUND_RATE_SPIKE": lambda merged, cfg, _t: _evaluate_rate_spike(merged, cfg, "refund"),
    "CHARGEBACK_RATE_SPIKE": lambda merged, cfg, _t: _evaluate_rate_spike(merged, cfg, "chargeback"),
    "EVIDENCE_COVERAGE_GAP": lambda merged, cfg, _t: _evaluate_evidence_gap(merged, cfg),
    "SUPPORT_OPERATIONAL_STRESS": lambda merged, cfg, _t: _evaluate_support_stress(merged, cfg),
    "COMBINED_LOSS_SIGNAL": lambda merged, cfg, triggered: _evaluate_combined(merged, cfg, triggered),
}


def evaluate_rules(record: dict, rules_config: dict | None = None) -> list[RuleResult]:
    """Evaluate every configured rule against a raw merchant-week record.

    ``record`` must not contain label_high_loss_next_30d or
    latent_state_for_demo_only as meaningful inputs -- they are stripped
    before feature computation and never read by any rule.
    """
    cfg = rules_config if rules_config is not None else load_rules_config()
    features = compute_features(record)
    merged = {**{k: v for k, v in record.items() if k not in EXCLUDED_INPUT_COLUMNS}, **features}

    severity_weights = cfg["severity_weights"]
    results: list[RuleResult] = []
    triggered_by_id: dict[str, bool] = {}

    # COMBINED_LOSS_SIGNAL depends on the other rules' outcomes, so it is
    # evaluated last using the already-computed triggered_by_id map.
    ordered_rule_ids = [r["rule_id"] for r in cfg["rules"] if r["rule_id"] != "COMBINED_LOSS_SIGNAL"]
    ordered_rule_ids.append("COMBINED_LOSS_SIGNAL")
    rules_by_id = {r["rule_id"]: r for r in cfg["rules"]}

    for rule_id in ordered_rule_ids:
        rule_cfg = rules_by_id[rule_id]
        if not rule_cfg.get("enabled", True):
            continue
        result = _RULE_EVALUATORS[rule_id](merged, rule_cfg, triggered_by_id)
        result = RuleResult(
            **{**result.__dict__, "score_contribution": severity_weights[result.severity] if result.triggered else 0}
        )
        results.append(result)
        triggered_by_id[rule_id] = result.triggered

    return results


def compute_score_and_tier(results: list[RuleResult], rules_config: dict) -> tuple[int, str]:
    score = min(100, sum(r.score_contribution for r in results))
    thresholds = rules_config["risk_tier_thresholds"]
    if score >= thresholds["high"]:
        tier = "high"
    elif score >= thresholds["medium"]:
        tier = "medium"
    else:
        tier = "low"
    return score, tier


def recommend(results: list[RuleResult], record: dict, rules_config: dict) -> str:
    by_id = {r.rule_id: r for r in results}
    combined = by_id.get("COMBINED_LOSS_SIGNAL")
    any_high = any(r.triggered and r.severity == "high" for r in results)
    any_medium = any(r.triggered and r.severity == "medium" for r in results)
    any_low = any(r.triggered and r.severity == "low" for r in results)
    repeat_confirmed_risk = record.get("previous_review_outcome") == "confirmed_risk"

    predicates = {
        "combined_signal_and_repeat_confirmed_risk": bool(combined and combined.triggered and repeat_confirmed_risk),
        "combined_signal_or_any_high_severity_rule": any_high,
        "any_medium_severity_rule": any_medium,
        "any_low_severity_rule": any_low,
        "default": True,
    }

    for entry in rules_config["recommendation_policy"]:
        if predicates.get(entry["when"], False):
            recommendation = entry["recommendation"]
            assert recommendation in ALLOWED_RECOMMENDATIONS
            return recommendation
    return "APPROVE"


def score_merchant_week(record: dict, rules_config: dict | None = None) -> dict:
    cfg = rules_config if rules_config is not None else load_rules_config()
    results = evaluate_rules(record, cfg)
    score, tier = compute_score_and_tier(results, cfg)
    recommendation = recommend(results, record, cfg)
    return {
        "risk_score": score,
        "risk_tier": tier,
        "recommendation": recommendation,
        "triggered_rules": [r.rule_id for r in results if r.triggered],
        "rule_results": results,
    }
