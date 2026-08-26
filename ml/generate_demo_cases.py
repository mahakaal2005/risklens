"""Milestone 4: select five representative held-out merchant-weeks and
generate their case packets, writing demo_data/demo_case_packets.json.

latent_state_for_demo_only and label_high_loss_next_30d are used ONLY to
select which demonstration records to use -- never passed into packet
content. build_case_packet() strips them defensively regardless.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.case_packet import build_case_packet
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN, compute_feature_frame
from ml.model_utils import ML_FEATURE_COLUMNS, combined_policy
from ml.rules_engine import DEFAULT_RULES_PATH, load_rules_config, score_merchant_week
from ml.split_data import DEFAULT_CSV_PATH, load_and_split

DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
MODEL_VERSION = "0.1.0"
DEFAULT_OUTPUT_PATH = Path("demo_data/demo_case_packets.json")


def _select_record(df: pd.DataFrame, scored: list[dict], predicate) -> tuple[dict, dict] | None:
    for i in range(len(df)):
        row = df.iloc[i]
        if predicate(row, scored[i]):
            return row.to_dict(), scored[i]
    return None


def _score_all(df: pd.DataFrame, pipeline, rules_config: dict) -> list[dict]:
    records = df.drop(columns=[LABEL_COLUMN, LATENT_STATE_COLUMN]).to_dict(orient="records")
    X = compute_feature_frame(df)[ML_FEATURE_COLUMNS]
    ml_probs = pipeline.predict_proba(X)[:, 1]

    scored = []
    for i, record in enumerate(records):
        rules_result = score_merchant_week(record, rules_config)
        scored.append({"rules_result": rules_result, "ml_probability": float(ml_probs[i])})
    return scored


def select_demo_records(df: pd.DataFrame, scored: list[dict], threshold: float) -> dict:
    selections = {}

    selections["stable_merchant"] = _select_record(
        df, scored,
        lambda row, s: row[LATENT_STATE_COLUMN] == "stable_merchant"
        and row[LABEL_COLUMN] == 0
        and s["rules_result"]["triggered_rules"] == []
        and s["ml_probability"] < 0.05,
    )

    selections["seasonal_sale_false_positive_candidate"] = _select_record(
        df, scored,
        lambda row, s: row[LATENT_STATE_COLUMN] == "seasonal_sale_legitimate_returns"
        and row[LABEL_COLUMN] == 0
        and "REFUND_RATE_SPIKE" in s["rules_result"]["triggered_rules"]
        and "CHARGEBACK_RATE_SPIKE" not in s["rules_result"]["triggered_rules"],
    )

    selections["operational_fulfilment_problem"] = _select_record(
        df, scored,
        lambda row, s: row[LATENT_STATE_COLUMN] == "operational_fulfilment_failure"
        and "CHARGEBACK_RATE_SPIKE" not in s["rules_result"]["triggered_rules"]
        and len(s["rules_result"]["triggered_rules"]) >= 2,
    )

    selections["high_risk_combined_loss_case"] = _select_record(
        df, scored,
        lambda row, s: row[LATENT_STATE_COLUMN] == "high_risk_merchant_behaviour"
        and row[LABEL_COLUMN] == 1
        and "COMBINED_LOSS_SIGNAL" in s["rules_result"]["triggered_rules"],
    )

    selections["early_hidden_risk_case"] = _select_record(
        df, scored,
        lambda row, s: row[LATENT_STATE_COLUMN] == "early_hidden_risk"
        and row[LABEL_COLUMN] == 1
        and s["ml_probability"] < threshold
        and len(s["rules_result"]["triggered_rules"]) == 0,
    )

    missing = [name for name, value in selections.items() if value is None]
    if missing:
        raise RuntimeError(f"Could not find a held-out demo record for: {missing}")

    return selections


def generate_demo_cases(
    csv_path: Path = DEFAULT_CSV_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    split = load_and_split(csv_path)
    test_df = split["test"]

    metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    model_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}.joblib"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    pipeline = joblib.load(model_path)
    threshold = metadata["selected_threshold"]
    rules_config = load_rules_config()
    rules_version = rules_config["version"]

    scored = _score_all(test_df, pipeline, rules_config)
    selections = select_demo_records(test_df, scored, threshold)

    packets = {}
    for case_name, (record, score_info) in selections.items():
        decision = combined_policy(score_info["ml_probability"], threshold, set(score_info["rules_result"]["triggered_rules"]))
        packet = build_case_packet(
            record=record,
            rules_result=score_info["rules_result"],
            ml_probability=score_info["ml_probability"],
            selected_threshold=threshold,
            model_version=metadata["model_version"],
            rules_version=rules_version,
            combined_decision=decision,
            pipeline=pipeline,
        )
        packets[case_name] = packet

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packets, f, indent=2)

    return packets


def main() -> None:
    packets = generate_demo_cases()
    for name, packet in packets.items():
        assessment = packet["assessment"]
        print(f"{name}: intensity={assessment['risk_signal_intensity']} recommendation={assessment['recommendation']} "
              f"triggered_rules={assessment['triggered_rules']}")
    print(f"\nWrote {len(packets)} demo case packets to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
