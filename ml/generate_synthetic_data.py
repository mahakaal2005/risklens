"""Synthetic merchant-week data generator for ClearRisk Recover.

One row = one merchant at one weekly prediction date. Predicts
label_high_loss_next_30d: will this merchant enter a simulated elevated
refund/chargeback-loss state in the next 30 days?

Design (per MODEL_CARD.md): a hidden latent merchant state is sampled first,
persisted week-to-week via a Markov transition matrix, and both the observed
features AND the label are drawn independently from that state's
distributions. The label is never a threshold function of the observed
refund/chargeback fields the rules engine will later check -- it is a
separate Bernoulli draw from the state's own probability, which is why a
model trained on observed features alone cannot trivially achieve
near-perfect precision/recall (see the honesty requirement in CLAUDE.md).

All output is synthetic and demonstration-only. No real merchant, customer,
or payment data is used or represented.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "0.2.0"

# v0.2.0: scaled from 220 merchants x 52 weeks to 900 x 104 weeks, and added
# two disclosed noise injections (missing data, anomaly weeks) -- see
# _inject_missing_data / _inject_anomaly_weeks below and MODEL_CARD.md.
MISSING_DATA_RATE = 0.02
MISSING_DATA_COLUMNS = [
    "delivery_evidence_coverage",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
    "previous_review_outcome",
]

ANOMALY_WEEK_RATE = 0.015
ANOMALY_REFUND_RATE_RANGE = (0.20, 0.45)
ANOMALY_VOLUME_CHANGE_RANGE = (-0.85, 3.0)

LATENT_STATES = [
    "stable_merchant",
    "seasonal_sale_legitimate_returns",
    "operational_fulfilment_failure",
    "high_risk_merchant_behaviour",
    "early_hidden_risk",
]

INITIAL_STATE_SHARES = {
    "stable_merchant": 0.70,
    "seasonal_sale_legitimate_returns": 0.10,
    "operational_fulfilment_failure": 0.08,
    "high_risk_merchant_behaviour": 0.07,
    "early_hidden_risk": 0.05,
}

# Rows sum to 1. Self-persistence dominates so a merchant's trajectory drifts
# gradually across weeks rather than resampling independently every week.
# Off-diagonal entries are calibrated (via each state's stationary share) so
# that the long-run mix of states across a multi-week timeline approximates
# INITIAL_STATE_SHARES, not just the first week's draw -- a plain "mostly
# stay, spread the rest evenly" matrix drifts far from the target mix over
# 52 weeks, since persistence compounds asymmetrically across states.
TRANSITION_MATRIX = {
    "stable_merchant": {
        "stable_merchant": 0.880,
        "seasonal_sale_legitimate_returns": 0.040,
        "operational_fulfilment_failure": 0.032,
        "high_risk_merchant_behaviour": 0.028,
        "early_hidden_risk": 0.020,
    },
    "seasonal_sale_legitimate_returns": {
        "stable_merchant": 0.350,
        "seasonal_sale_legitimate_returns": 0.550,
        "operational_fulfilment_failure": 0.040,
        "high_risk_merchant_behaviour": 0.035,
        "early_hidden_risk": 0.025,
    },
    "operational_fulfilment_failure": {
        "stable_merchant": 0.2663,
        "seasonal_sale_legitimate_returns": 0.038,
        "operational_fulfilment_failure": 0.650,
        "high_risk_merchant_behaviour": 0.0266,
        "early_hidden_risk": 0.019,
    },
    "high_risk_merchant_behaviour": {
        "stable_merchant": 0.2634,
        "seasonal_sale_legitimate_returns": 0.0376,
        "operational_fulfilment_failure": 0.0301,
        "high_risk_merchant_behaviour": 0.650,
        "early_hidden_risk": 0.0188,
    },
    "early_hidden_risk": {
        "stable_merchant": 0.2579,
        "seasonal_sale_legitimate_returns": 0.0368,
        "operational_fulfilment_failure": 0.0295,
        "high_risk_merchant_behaviour": 0.0258,
        "early_hidden_risk": 0.650,
    },
}

SEVERITY_RANK = {
    "stable_merchant": 0,
    "seasonal_sale_legitimate_returns": 0,
    "operational_fulfilment_failure": 1,
    "high_risk_merchant_behaviour": 2,
    "early_hidden_risk": 1,
}

MERCHANT_CATEGORIES = [
    "apparel",
    "electronics",
    "grocery",
    "travel",
    "digital_services",
    "food_delivery",
]

DISPUTE_REASONS = [
    "item_not_as_described",
    "not_received",
    "duplicate_charge",
    "quality_issue",
    "other",
]

REVIEW_OUTCOMES = ["none", "confirmed_risk", "false_positive", "inconclusive", "operational_issue"]

# Per-state generation parameters. Ranges deliberately overlap between
# adjacent states so no single week's snapshot trivially reveals the state.
STATE_PARAMS = {
    "stable_merchant": {
        "refund_rate": (0.015, 0.005, 0.03),
        "chargeback_rate": (0.003, 0.001, 0.008),
        "evidence_coverage": (0.92, 0.85, 0.98),
        "support_ticket_rate": (0.010, 0.002, 0.025),
        "support_resolution_hours_mean": 20.0,
        "support_resolution_hours_sigma": 0.35,
        "volume_change_mean": 0.0,
        "volume_change_sd": 0.08,
        "label_probability": 0.015,
        "dispute_weights": {"other": 0.40, "item_not_as_described": 0.20, "not_received": 0.15, "duplicate_charge": 0.15, "quality_issue": 0.10},
        "review_weights": {"none": 0.85, "false_positive": 0.05, "confirmed_risk": 0.03, "inconclusive": 0.05, "operational_issue": 0.02},
    },
    "seasonal_sale_legitimate_returns": {
        "refund_rate": (0.06, 0.04, 0.09),
        "chargeback_rate": (0.004, 0.001, 0.009),
        "evidence_coverage": (0.87, 0.80, 0.95),
        "support_ticket_rate": (0.020, 0.008, 0.035),
        "support_resolution_hours_mean": 24.0,
        "support_resolution_hours_sigma": 0.35,
        "volume_change_mean": 0.5,
        "volume_change_sd": 0.20,
        "label_probability": 0.10,
        "dispute_weights": {"item_not_as_described": 0.35, "other": 0.35, "not_received": 0.10, "duplicate_charge": 0.10, "quality_issue": 0.10},
        "review_weights": {"none": 0.80, "false_positive": 0.10, "confirmed_risk": 0.02, "inconclusive": 0.05, "operational_issue": 0.03},
    },
    "operational_fulfilment_failure": {
        "refund_rate": (0.05, 0.03, 0.08),
        "chargeback_rate": (0.012, 0.006, 0.02),
        "evidence_coverage": (0.62, 0.50, 0.75),
        "support_ticket_rate": (0.045, 0.025, 0.07),
        "support_resolution_hours_mean": 55.0,
        "support_resolution_hours_sigma": 0.4,
        "volume_change_mean": -0.05,
        "volume_change_sd": 0.15,
        "label_probability": 0.35,
        "dispute_weights": {"not_received": 0.35, "item_not_as_described": 0.30, "other": 0.15, "quality_issue": 0.10, "duplicate_charge": 0.10},
        "review_weights": {"none": 0.50, "operational_issue": 0.20, "inconclusive": 0.15, "confirmed_risk": 0.10, "false_positive": 0.05},
    },
    "high_risk_merchant_behaviour": {
        "refund_rate": (0.09, 0.06, 0.15),
        "chargeback_rate": (0.035, 0.02, 0.06),
        "evidence_coverage": (0.35, 0.20, 0.50),
        "support_ticket_rate": (0.07, 0.045, 0.10),
        "support_resolution_hours_mean": 70.0,
        "support_resolution_hours_sigma": 0.45,
        "volume_change_mean": 0.10,
        "volume_change_sd": 0.30,
        "label_probability": 0.78,
        "dispute_weights": {"duplicate_charge": 0.30, "quality_issue": 0.25, "not_received": 0.25, "item_not_as_described": 0.15, "other": 0.05},
        "review_weights": {"confirmed_risk": 0.45, "operational_issue": 0.15, "inconclusive": 0.15, "none": 0.20, "false_positive": 0.05},
    },
    "early_hidden_risk": {
        "refund_rate": (0.025, 0.015, 0.04),
        "chargeback_rate": (0.006, 0.003, 0.012),
        "evidence_coverage": (0.80, 0.70, 0.90),
        "support_ticket_rate": (0.018, 0.008, 0.03),
        "support_resolution_hours_mean": 30.0,
        "support_resolution_hours_sigma": 0.35,
        "volume_change_mean": 0.02,
        "volume_change_sd": 0.12,
        "label_probability": 0.52,
        "dispute_weights": {"item_not_as_described": 0.30, "other": 0.30, "not_received": 0.20, "duplicate_charge": 0.10, "quality_issue": 0.10},
        "review_weights": {"none": 0.75, "inconclusive": 0.10, "false_positive": 0.05, "confirmed_risk": 0.05, "operational_issue": 0.05},
    },
}

LABEL_STREAK_BONUS = 0.05
LABEL_PROBABILITY_FLOOR = 0.005
LABEL_PROBABILITY_CEILING = 0.95

DEFAULT_SEED = 42
DEFAULT_N_MERCHANTS = 900
DEFAULT_N_WEEKS = 104
DEFAULT_START_DATE = dt.date(2023, 1, 2)  # a Monday


def _sample_bounded(rng: np.random.Generator, mean: float, low: float, high: float, concentration: float = 25.0) -> float:
    rel_mean = min(max((mean - low) / (high - low), 0.01), 0.99)
    a = rel_mean * concentration
    b = (1 - rel_mean) * concentration
    return low + rng.beta(a, b) * (high - low)


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs = probs / probs.sum()
    return keys[rng.choice(len(keys), p=probs)]


def _sample_initial_state(rng: np.random.Generator) -> str:
    return _weighted_choice(rng, INITIAL_STATE_SHARES)


def _simulate_state_path(rng: np.random.Generator, initial_state: str, n_weeks: int) -> list[str]:
    path = [initial_state]
    for _ in range(n_weeks - 1):
        path.append(_weighted_choice(rng, TRANSITION_MATRIX[path[-1]]))
    return path


def _label_probability_for_week(state_path: list[str], week_index: int) -> float:
    state = state_path[week_index]
    p = STATE_PARAMS[state]["label_probability"]
    if week_index >= 2:
        current_rank = SEVERITY_RANK[state_path[week_index]]
        past_rank = SEVERITY_RANK[state_path[week_index - 2]]
        if current_rank > past_rank:
            p += LABEL_STREAK_BONUS
    return min(max(p, LABEL_PROBABILITY_FLOOR), LABEL_PROBABILITY_CEILING)


def _generate_merchant(rng: np.random.Generator, merchant_index: int, n_weeks: int, start_date: dt.date) -> list[dict]:
    merchant_id = f"merchant_demo_{merchant_index:04d}"
    merchant_category = MERCHANT_CATEGORIES[rng.integers(0, len(MERCHANT_CATEGORIES))]
    base_age_days = int(rng.integers(30, 1500))
    base_avg_ticket = float(np.exp(rng.normal(7.3, 0.4)))
    base_txn_count = float(np.exp(rng.normal(np.log(300), 0.6)))

    initial_state = _sample_initial_state(rng)
    state_path = _simulate_state_path(rng, initial_state, n_weeks)

    rows = []
    prev_volume_30d = None
    prev_refund_rate = None
    prev_chargeback_rate = None

    for week_index in range(n_weeks):
        state = state_path[week_index]
        params = STATE_PARAMS[state]
        week_start = start_date + dt.timedelta(weeks=week_index)

        refund_rate_30d = _sample_bounded(rng, *params["refund_rate"])
        chargeback_rate_30d = _sample_bounded(rng, *params["chargeback_rate"])
        evidence_coverage = _sample_bounded(rng, *params["evidence_coverage"])
        support_ticket_rate = _sample_bounded(rng, *params["support_ticket_rate"])
        resolution_hours = float(np.exp(rng.normal(np.log(params["support_resolution_hours_mean"]), params["support_resolution_hours_sigma"])))
        volume_change_noise = rng.normal(params["volume_change_mean"], params["volume_change_sd"])
        volume_change_noise = max(volume_change_noise, -0.9)

        txn_count_30d = max(1, int(round(base_txn_count * (1 + volume_change_noise) * rng.normal(1.0, 0.05))))
        volume_30d = max(0.0, txn_count_30d * base_avg_ticket * rng.normal(1.0, 0.05))

        if prev_volume_30d is None:
            volume_previous_30d = volume_30d
        else:
            volume_previous_30d = prev_volume_30d
        volume_change_30d = (volume_30d - volume_previous_30d) / volume_previous_30d if volume_previous_30d > 0 else 0.0

        refund_rate_previous_30d = prev_refund_rate if prev_refund_rate is not None else refund_rate_30d
        chargeback_rate_previous_30d = prev_chargeback_rate if prev_chargeback_rate is not None else chargeback_rate_30d

        refund_count_30d = int(round(txn_count_30d * refund_rate_30d))
        chargeback_count_30d = int(round(txn_count_30d * chargeback_rate_30d))

        label_probability = _label_probability_for_week(state_path, week_index)
        label = int(rng.random() < label_probability)

        rows.append({
            "merchant_id": merchant_id,
            "week_start": week_start.isoformat(),
            "merchant_category": merchant_category,
            "merchant_age_days": base_age_days + week_index * 7,
            "transaction_count_30d": txn_count_30d,
            "transaction_volume_30d": round(volume_30d, 2),
            "transaction_volume_previous_30d": round(volume_previous_30d, 2),
            "transaction_volume_change_30d": round(volume_change_30d, 4),
            "refund_count_30d": refund_count_30d,
            "refund_rate_30d": round(refund_rate_30d, 4),
            "refund_rate_previous_30d": round(refund_rate_previous_30d, 4),
            "refund_rate_change_30d": round(refund_rate_30d - refund_rate_previous_30d, 4),
            "chargeback_count_30d": chargeback_count_30d,
            "chargeback_rate_30d": round(chargeback_rate_30d, 4),
            "chargeback_rate_previous_30d": round(chargeback_rate_previous_30d, 4),
            "chargeback_rate_change_30d": round(chargeback_rate_30d - chargeback_rate_previous_30d, 4),
            "top_dispute_reason_category": _weighted_choice(rng, params["dispute_weights"]),
            "delivery_evidence_coverage": round(evidence_coverage, 4),
            "support_ticket_rate": round(support_ticket_rate, 4),
            "average_support_resolution_time_hours": round(resolution_hours, 2),
            "previous_review_outcome": _weighted_choice(rng, params["review_weights"]),
            "label_high_loss_next_30d": label,
            "latent_state_for_demo_only": state,
        })

        prev_volume_30d = volume_30d
        prev_refund_rate = refund_rate_30d
        prev_chargeback_rate = chargeback_rate_30d

    return rows


def _inject_missing_data(df: pd.DataFrame, rng: np.random.Generator, rate: float = MISSING_DATA_RATE) -> pd.DataFrame:
    """Randomly nulls out a small fraction of values in fields that already
    have documented missing-value behavior in ml/features.py -- so that
    behavior is actually exercised at dataset scale instead of only in
    hand-written unit tests. Independent per column; does not touch
    label_high_loss_next_30d or latent_state_for_demo_only."""
    df = df.copy()
    for column in MISSING_DATA_COLUMNS:
        mask = rng.random(len(df)) < rate
        df.loc[mask, column] = np.nan
    return df


def _inject_anomaly_weeks(df: pd.DataFrame, rng: np.random.Generator, rate: float = ANOMALY_WEEK_RATE) -> pd.DataFrame:
    """Overwrites a small fraction of rows' refund rate or transaction volume
    with an extreme one-off value drawn independently of the row's latent
    state -- a noisy week that doesn't correspond to any real state change.
    Prevents the observed features from being trivially linearly separable
    and tests whether rules/models are robust to noise, not just to the
    state signal they were designed around. Does not touch the label -- an
    anomalous week is not itself evidence of elevated loss risk.

    Every dependent raw column (counts, the *_change_30d columns) is
    recomputed after the anomalous value is set, so the row stays
    internally consistent rather than encoding a rate that disagrees with
    its own count/previous-value columns.
    """
    df = df.copy()
    n = len(df)

    refund_mask = rng.random(n) < (rate / 2)
    refund_idx = df.index[refund_mask]
    df.loc[refund_idx, "refund_rate_30d"] = rng.uniform(*ANOMALY_REFUND_RATE_RANGE, size=len(refund_idx))
    df.loc[refund_idx, "refund_count_30d"] = (df.loc[refund_idx, "transaction_count_30d"] * df.loc[refund_idx, "refund_rate_30d"]).round().astype(int)
    df.loc[refund_idx, "refund_rate_change_30d"] = (df.loc[refund_idx, "refund_rate_30d"] - df.loc[refund_idx, "refund_rate_previous_30d"]).round(4)

    volume_mask = rng.random(n) < (rate / 2)
    volume_idx = df.index[volume_mask]
    volume_change = rng.uniform(*ANOMALY_VOLUME_CHANGE_RANGE, size=len(volume_idx))
    df.loc[volume_idx, "transaction_volume_30d"] = (df.loc[volume_idx, "transaction_volume_previous_30d"] * (1 + volume_change)).clip(lower=0.0).round(2)
    df.loc[volume_idx, "transaction_volume_change_30d"] = (
        (df.loc[volume_idx, "transaction_volume_30d"] - df.loc[volume_idx, "transaction_volume_previous_30d"])
        / df.loc[volume_idx, "transaction_volume_previous_30d"].replace(0, pd.NA)
    ).fillna(0.0).round(4)

    return df


def generate_dataset(seed: int = DEFAULT_SEED, n_merchants: int = DEFAULT_N_MERCHANTS, n_weeks: int = DEFAULT_N_WEEKS, start_date: dt.date = DEFAULT_START_DATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_rows = []
    for merchant_index in range(1, n_merchants + 1):
        all_rows.extend(_generate_merchant(rng, merchant_index, n_weeks, start_date))
    df = pd.DataFrame(all_rows)
    df = _inject_missing_data(df, rng)
    df = _inject_anomaly_weeks(df, rng)
    return df


def build_metadata(df: pd.DataFrame, seed: int, n_merchants: int, n_weeks: int) -> dict:
    label_counts = df["label_high_loss_next_30d"].value_counts().to_dict()
    positive = int(label_counts.get(1, 0))
    negative = int(label_counts.get(0, 0))
    total = positive + negative
    return {
        "seed": seed,
        "generation_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "merchant_count": n_merchants,
        "row_count": int(len(df)),
        "weeks_per_merchant": n_weeks,
        "date_range": {
            "week_start_min": df["week_start"].min(),
            "week_start_max": df["week_start"].max(),
        },
        "latent_state_distribution": df["latent_state_for_demo_only"].value_counts().to_dict(),
        "label_distribution": {
            "0": negative,
            "1": positive,
            "positive_rate": round(positive / total, 4) if total else None,
        },
        "generator_version": GENERATOR_VERSION,
        "noise_injection": {
            "missing_data_rate": MISSING_DATA_RATE,
            "missing_data_columns": MISSING_DATA_COLUMNS,
            "anomaly_week_rate": ANOMALY_WEEK_RATE,
        },
        "statement": (
            "This dataset is synthetic and demonstration-only. It does not represent "
            "real merchants, real transactions, or real payment-risk outcomes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic merchant-week data for ClearRisk Recover.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-merchants", type=int, default=DEFAULT_N_MERCHANTS)
    parser.add_argument("--n-weeks", type=int, default=DEFAULT_N_WEEKS)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--output-csv", type=str, default="demo_data/synthetic_merchant_week_data.csv")
    parser.add_argument("--output-metadata", type=str, default="demo_data/synthetic_data_metadata.json")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    df = generate_dataset(seed=args.seed, n_merchants=args.n_merchants, n_weeks=args.n_weeks, start_date=start_date)
    metadata = build_metadata(df, args.seed, args.n_merchants, args.n_weeks)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    output_metadata = Path(args.output_metadata)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(df)} rows ({args.n_merchants} merchants x {args.n_weeks} weeks) to {output_csv}")
    print(f"Wrote metadata to {output_metadata}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
