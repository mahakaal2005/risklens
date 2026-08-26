"""Milestone 1.1 data-quality inspection for the synthetic merchant-week dataset.

This is a verification step, not model implementation. It checks that the
dataset generated in Milestone 1 is fit to build on: latent states are
covered, observed-feature distributions overlap enough that the task isn't
trivially easy, no leakage-shaped columns exist among the candidate model
features, and a plain Logistic Regression smoke test (no rules engine)
produces a plausible, imperfect result rather than a suspiciously perfect
one. Writes docs/MILESTONE_1_DATA_QUALITY_REPORT.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ml.data_validation import REQUIRED_LATENT_STATES, validate_dataset

DEFAULT_CSV_PATH = Path("demo_data/synthetic_merchant_week_data.csv")
DEFAULT_REPORT_PATH = Path("docs/MILESTONE_1_DATA_QUALITY_REPORT.md")

LABEL_COLUMN = "label_high_loss_next_30d"
LATENT_STATE_COLUMN = "latent_state_for_demo_only"

# Matches MODEL_CARD.md's "Features" section exactly. This is the candidate
# feature set a Milestone 7 model would actually train on.
NUMERIC_CANDIDATE_FEATURES = [
    "refund_rate_30d",
    "refund_rate_change_30d",
    "chargeback_rate_30d",
    "chargeback_rate_change_30d",
    "transaction_volume_change_30d",
    "delivery_evidence_coverage",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
    "merchant_age_days",
]
CATEGORICAL_CANDIDATE_FEATURES = ["merchant_category", "previous_review_outcome"]
CANDIDATE_MODEL_FEATURES = NUMERIC_CANDIDATE_FEATURES + CATEGORICAL_CANDIDATE_FEATURES

# Everything present in the dataset that is NOT a candidate model feature,
# with the reason it's excluded.
EXCLUDED_COLUMNS = {
    "merchant_id": "identifier, not a predictive feature",
    "week_start": "identifier/time key, used for splitting, not fed to the model directly",
    "transaction_count_30d": "raw count; the model uses the derived rate/change fields instead",
    "transaction_volume_30d": "raw volume; the model uses transaction_volume_change_30d instead",
    "transaction_volume_previous_30d": "raw prior volume; superseded by transaction_volume_change_30d",
    "refund_count_30d": "raw count; the model uses refund_rate_30d / refund_rate_change_30d instead",
    "refund_rate_previous_30d": "superseded by refund_rate_change_30d",
    "chargeback_count_30d": "raw count; the model uses chargeback_rate_30d / chargeback_rate_change_30d instead",
    "chargeback_rate_previous_30d": "superseded by chargeback_rate_change_30d",
    "top_dispute_reason_category": "not in MODEL_CARD.md's frozen candidate feature list",
    LATENT_STATE_COLUMN: "hidden ground-truth generation state; not observable in a real deployment, would make evaluation circular",
    LABEL_COLUMN: "the prediction target itself",
}

FEATURE_OVERLAP_FIELDS = [
    "refund_rate_30d",
    "refund_rate_change_30d",
    "chargeback_rate_30d",
    "chargeback_rate_change_30d",
    "delivery_evidence_coverage",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
    "transaction_volume_change_30d",
]

LEAKAGE_TERMS = ["future", "next_30d", "outcome", "label", "risk_state"]
ALLOWED_LEAKAGE_EXCEPTIONS = {LABEL_COLUMN, LATENT_STATE_COLUMN}
# Matches the "outcome" leakage term but is not leakage: it is the outcome of
# a *prior* review, known before the prediction date, not a future/current one.
REVIEWED_NOT_LEAKAGE = {"previous_review_outcome"}

NEAR_PERFECT_PR_AUC = 0.98
NEAR_PERFECT_PRECISION = 0.98
NEAR_PERFECT_RECALL = 0.98


def load_dataset(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "row_count": int(len(df)),
        "merchant_count": int(df["merchant_id"].nunique()),
        "week_count": int(df["week_start"].nunique()),
        "date_range": {"min": str(df["week_start"].min()), "max": str(df["week_start"].max())},
        "label_positive_rate": round(float(df[LABEL_COLUMN].mean()), 4),
        "latent_state_distribution": df[LATENT_STATE_COLUMN].value_counts().to_dict(),
        "merchant_category_distribution": df.drop_duplicates("merchant_id")["merchant_category"].value_counts().to_dict(),
    }


def outcome_rates_by_state(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(LATENT_STATE_COLUMN)[LABEL_COLUMN].agg(["count", "mean"])
    grouped.columns = ["count", "positive_rate"]
    return grouped.reindex(REQUIRED_LATENT_STATES)


def seasonal_and_early_hidden_counts(df: pd.DataFrame) -> dict:
    def counts_for(state: str) -> dict:
        subset = df[df[LATENT_STATE_COLUMN] == state]
        return {
            "label_0_count": int((subset[LABEL_COLUMN] == 0).sum()),
            "label_1_count": int((subset[LABEL_COLUMN] == 1).sum()),
        }

    return {
        "seasonal_sale_legitimate_returns": counts_for("seasonal_sale_legitimate_returns"),
        "early_hidden_risk": counts_for("early_hidden_risk"),
    }


def feature_overlap_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for feature in FEATURE_OVERLAP_FIELDS:
        stats[feature] = {}
        for state in REQUIRED_LATENT_STATES:
            values = df.loc[df[LATENT_STATE_COLUMN] == state, feature]
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            stats[feature][state] = {"median": round(float(median), 4), "q1": round(float(q1), 4), "q3": round(float(q3), 4)}
    return stats


def _iqr_overlaps(a: dict, b: dict) -> bool:
    return a["q1"] <= b["q3"] and b["q1"] <= a["q3"]


def summarize_overlap(stats: dict) -> dict:
    pairs_of_interest = [
        ("stable_merchant", "early_hidden_risk"),
        ("seasonal_sale_legitimate_returns", "operational_fulfilment_failure"),
    ]
    summary = {}
    for state_a, state_b in pairs_of_interest:
        overlapping_features = [f for f in FEATURE_OVERLAP_FIELDS if _iqr_overlaps(stats[f][state_a], stats[f][state_b])]
        summary[f"{state_a}_vs_{state_b}"] = {
            "overlapping_feature_count": len(overlapping_features),
            "total_features": len(FEATURE_OVERLAP_FIELDS),
            "overlapping_features": overlapping_features,
        }
    return summary


def _overlap_is_sufficient(overlap_summary: dict) -> bool:
    # Not every feature needs to overlap between a state pair -- some fields
    # (e.g. delivery_evidence_coverage) are *meant* to differ sharply even
    # between "easy" pairs. At least one overlapping feature per designed
    # false-positive/false-negative pair is the qualitative signal; the
    # quantitative proof is the baseline sanity test in Section 5, which is
    # the authoritative check, not this per-feature heuristic alone.
    return all(s["overlapping_feature_count"] >= 1 for s in overlap_summary.values())


def scan_leakage_column_names(columns: list[str]) -> dict:
    flagged: dict[str, list[str]] = {}
    reviewed: list[str] = []
    for term in LEAKAGE_TERMS:
        matches = [c for c in columns if term in c.lower()]
        unexplained = [c for c in matches if c not in ALLOWED_LEAKAGE_EXCEPTIONS and c not in REVIEWED_NOT_LEAKAGE]
        if unexplained:
            flagged[term] = unexplained
        reviewed.extend(c for c in matches if c in REVIEWED_NOT_LEAKAGE and c not in reviewed)
    return {"unexplained_matches": flagged, "reviewed_not_leakage": sorted(set(reviewed))}


def correlation_screen(df: pd.DataFrame) -> dict:
    results = {}
    for feature in NUMERIC_CANDIDATE_FEATURES:
        corr = df[feature].corr(df[LABEL_COLUMN])
        results[feature] = {"type": "pearson_point_biserial", "correlation": round(float(corr), 4)}
    for feature in CATEGORICAL_CANDIDATE_FEATURES:
        rates = df.groupby(feature)[LABEL_COLUMN].mean().round(4).to_dict()
        results[feature] = {"type": "positive_rate_by_category", "rates": rates}
    return results


def _time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(df["week_start"].unique())
    n = len(weeks)
    train_cut = weeks[int(n * 0.70) - 1]
    val_cut = weeks[int(n * 0.85) - 1]
    train = df[df["week_start"] <= train_cut]
    val = df[(df["week_start"] > train_cut) & (df["week_start"] <= val_cut)]
    test = df[df["week_start"] > val_cut]
    return train, val, test


def _build_design_matrix(df: pd.DataFrame, encoded_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(df[CANDIDATE_MODEL_FEATURES], columns=CATEGORICAL_CANDIDATE_FEATURES)
    if encoded_columns is not None:
        encoded = encoded.reindex(columns=encoded_columns, fill_value=0)
    return encoded, list(encoded.columns)


def run_baseline_sanity_check(df: pd.DataFrame) -> dict:
    """Inspection-only Logistic Regression smoke test. Rules are not used here."""
    train, val, test = _time_based_split(df)

    X_train, feature_columns = _build_design_matrix(train)
    X_val, _ = _build_design_matrix(val, encoded_columns=feature_columns)
    X_test, _ = _build_design_matrix(test, encoded_columns=feature_columns)
    y_train, y_val, y_test = train[LABEL_COLUMN].values, val[LABEL_COLUMN].values, test[LABEL_COLUMN].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    pr_auc = float(average_precision_score(y_test, test_probs))
    precision = float(precision_score(y_test, test_preds, zero_division=0))
    recall = float(recall_score(y_test, test_preds, zero_division=0))
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "rules_used": False,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "operating_threshold": round(float(best_threshold), 2),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_near_perfect_gate(metrics: dict) -> tuple[str, str]:
    cm = metrics["confusion_matrix"]
    zero_fp_or_fn = cm["fp"] == 0 or cm["fn"] == 0
    near_perfect = (
        metrics["pr_auc"] >= NEAR_PERFECT_PR_AUC
        or (metrics["precision"] >= NEAR_PERFECT_PRECISION and metrics["recall"] >= NEAR_PERFECT_RECALL)
        or zero_fp_or_fn
    )
    if near_perfect:
        reason = (
            "Triggered because: "
            + ("PR-AUC >= 0.98. " if metrics["pr_auc"] >= NEAR_PERFECT_PR_AUC else "")
            + ("precision and recall both >= 0.98. " if metrics["precision"] >= NEAR_PERFECT_PRECISION and metrics["recall"] >= NEAR_PERFECT_RECALL else "")
            + ("zero false positives or zero false negatives on the test set. " if zero_fp_or_fn else "")
            + "Likely cause to check first: insufficient feature-distribution overlap between latent "
            + "states (see Section 3), or a candidate feature that inadvertently encodes the latent "
            + "state almost perfectly."
        )
        return "UNDER INVESTIGATION", reason
    return "APPROVED FOR MILESTONE 2", "Smoke-test model is moderately useful but imperfect, as expected."


def generate_report(csv_path: Path = DEFAULT_CSV_PATH, report_path: Path = DEFAULT_REPORT_PATH) -> dict:
    df = load_dataset(csv_path)
    validate_dataset(df)

    overview = dataset_overview(df)
    state_rates = outcome_rates_by_state(df)
    explicit_counts = seasonal_and_early_hidden_counts(df)
    overlap_stats = feature_overlap_stats(df)
    overlap_summary = summarize_overlap(overlap_stats)
    leakage_scan = scan_leakage_column_names(list(df.columns))
    correlations = correlation_screen(df)
    baseline_metrics = run_baseline_sanity_check(df)
    decision, decision_reason = evaluate_near_perfect_gate(baseline_metrics)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(overview, state_rates, explicit_counts, overlap_stats, overlap_summary, leakage_scan, correlations, baseline_metrics, decision, decision_reason))

    return {
        "overview": overview,
        "explicit_counts": explicit_counts,
        "overlap_summary": overlap_summary,
        "leakage_scan": leakage_scan,
        "baseline_metrics": baseline_metrics,
        "decision": decision,
        "decision_reason": decision_reason,
    }


def _render_markdown(overview, state_rates, explicit_counts, overlap_stats, overlap_summary, leakage_scan, correlations, baseline_metrics, decision, decision_reason) -> str:
    lines = []
    lines.append("# Milestone 1 Data Quality Report — ClearRisk Recover")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("Status: **" + decision + "**")
    lines.append("")
    lines.append(decision_reason)
    lines.append("")
    lines.append("This is a Milestone 1.1 verification report, not the Milestone 7 model. The Logistic Regression below is an inspection-only smoke test with no rules engine involved.")
    lines.append("")

    lines.append("## 1. Dataset overview")
    lines.append("")
    lines.append(f"- Row count: {overview['row_count']}")
    lines.append(f"- Merchant count: {overview['merchant_count']}")
    lines.append(f"- Week count: {overview['week_count']} ({overview['date_range']['min']} to {overview['date_range']['max']})")
    lines.append(f"- Target-label positive rate: {overview['label_positive_rate']}")
    lines.append("- Latent-state distribution:")
    for state, count in overview["latent_state_distribution"].items():
        lines.append(f"  - {state}: {count}")
    lines.append("- Merchant-category distribution (by merchant, not row):")
    for cat, count in overview["merchant_category_distribution"].items():
        lines.append(f"  - {cat}: {count}")
    lines.append("")

    lines.append("## 2. Outcome rates by latent state")
    lines.append("")
    lines.append("| Latent state | Count | Positive-label rate |")
    lines.append("|---|---|---|")
    for state, row in state_rates.iterrows():
        lines.append(f"| {state} | {int(row['count'])} | {round(row['positive_rate'], 4)} |")
    lines.append("")
    lines.append(f"- seasonal_sale_legitimate_returns label=0 count: {explicit_counts['seasonal_sale_legitimate_returns']['label_0_count']}")
    lines.append(f"- seasonal_sale_legitimate_returns label=1 count: {explicit_counts['seasonal_sale_legitimate_returns']['label_1_count']}")
    lines.append(f"- early_hidden_risk label=0 count: {explicit_counts['early_hidden_risk']['label_0_count']}")
    lines.append(f"- early_hidden_risk label=1 count: {explicit_counts['early_hidden_risk']['label_1_count']}")
    lines.append("")

    lines.append("## 3. Feature overlap checks")
    lines.append("")
    lines.append("Median (IQR) per latent state:")
    lines.append("")
    header = "| Feature | " + " | ".join(REQUIRED_LATENT_STATES) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(REQUIRED_LATENT_STATES))
    for feature in FEATURE_OVERLAP_FIELDS:
        cells = []
        for state in REQUIRED_LATENT_STATES:
            s = overlap_stats[feature][state]
            cells.append(f"{s['median']} ({s['q1']}-{s['q3']})")
        lines.append(f"| {feature} | " + " | ".join(cells) + " |")
    lines.append("")
    for pair, summary in overlap_summary.items():
        lines.append(f"- {pair}: {summary['overlapping_feature_count']}/{summary['total_features']} features have overlapping IQRs ({summary['overlapping_features']})")
    overlap_sufficient = _overlap_is_sufficient(overlap_summary)
    lines.append("")
    if overlap_sufficient:
        lines.append(
            "**Conclusion: both designed false-positive/false-negative pairs have at least one "
            "overlapping feature; not every feature needs to overlap (e.g. delivery_evidence_coverage "
            "is meant to differ sharply). The definitive test of whether a simple model can trivially "
            "separate all classes is the empirical baseline in Section 5, not this per-feature heuristic "
            "alone.**"
        )
    else:
        lines.append(
            "**Conclusion: at least one designed false-positive/false-negative pair has zero overlapping "
            "features on this heuristic — investigate before trusting Section 5's baseline result.**"
        )
    lines.append("")

    lines.append("## 4. Correlation / leakage screening")
    lines.append("")
    lines.append("Candidate model features (matches MODEL_CARD.md):")
    for f in CANDIDATE_MODEL_FEATURES:
        lines.append(f"- {f}")
    lines.append("")
    lines.append(f"- `{LATENT_STATE_COLUMN}` excluded from candidate features: confirmed.")
    lines.append(f"- `{LABEL_COLUMN}` excluded from candidate features: confirmed (it is the prediction target).")
    lines.append("")
    lines.append("Column-name leakage-term scan:")
    if leakage_scan["unexplained_matches"]:
        for term, cols in leakage_scan["unexplained_matches"].items():
            lines.append(f"- **UNEXPLAINED MATCH** for term '{term}': {cols}")
    else:
        lines.append("- No unexplained leakage-term matches found.")
    if leakage_scan["reviewed_not_leakage"]:
        lines.append(f"- Reviewed and confirmed not leakage (matches a term but is a legitimate prior-outcome feature): {leakage_scan['reviewed_not_leakage']}")
    lines.append("")
    lines.append("Excluded columns and reason:")
    for col, reason in EXCLUDED_COLUMNS.items():
        lines.append(f"- `{col}`: {reason}")
    lines.append("")
    lines.append("Correlation of each candidate feature with the label:")
    lines.append("")
    for feature, result in correlations.items():
        if result["type"] == "pearson_point_biserial":
            lines.append(f"- {feature}: r = {result['correlation']}")
        else:
            lines.append(f"- {feature} (positive rate by category): {result['rates']}")
    lines.append("")

    lines.append("## 5. Baseline sanity test (inspection-only Logistic Regression, no rules)")
    lines.append("")
    lines.append(f"- Train rows: {baseline_metrics['train_rows']}, Validation rows: {baseline_metrics['validation_rows']}, Test rows: {baseline_metrics['test_rows']}")
    lines.append(f"- Operating threshold (selected on validation): {baseline_metrics['operating_threshold']}")
    lines.append(f"- PR-AUC: {baseline_metrics['pr_auc']}")
    lines.append(f"- Precision: {baseline_metrics['precision']}")
    lines.append(f"- Recall: {baseline_metrics['recall']}")
    lines.append(f"- False-positive rate: {baseline_metrics['false_positive_rate']}")
    cm = baseline_metrics["confusion_matrix"]
    lines.append(f"- Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    lines.append("- Rules engine was not used in this smoke test.")
    lines.append("")

    lines.append("## 6. Near-perfect-score gate")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append(decision_reason)
    lines.append("")

    lines.append("## Limitation")
    lines.append("")
    lines.append("This report demonstrates the prototype's data-quality workflow only. It does not prove real-world chargeback-risk prediction quality; all data and outcomes are synthetic.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone 1.1 data-quality inspection.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    result = generate_report(Path(args.csv_path), Path(args.report_path))
    print(f"Decision: {result['decision']}")
    print(result["decision_reason"])
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    main()
