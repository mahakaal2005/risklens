"""Dump per-week attention weights for a handful of held-out merchant-weeks,
so the dashboard and the demo can show what a *temporal* explanation looks
like next to the existing feature-level before/after explanation.

Scope note, deliberately narrow: this is a comparison-only artifact. It is not
wired into ml/case_packet.py, app/services/case_service.py, or anything a
merchant sees. Live case explanations remain Logistic Regression's
interpretable feature contributions -- attention weights are an illustration
of a second explanation modality, not a shipped one.

All rows are synthetic. Merchant identifiers are generator-assigned synthetic
IDs, not real merchants.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import LABEL_COLUMN
from ml.split_data import DEFAULT_CSV_PATH, load_and_split
from ml.train_trajectory_transformer import (
    DEFAULT_ARTIFACT_DIR,
    MODEL_VERSION,
    load_trained_model,
)

DEFAULT_N_EXAMPLES = 6
OUTPUT_FILENAME = "trajectory_attention_examples.json"


def build_examples(artifact_dir: Path = DEFAULT_ARTIFACT_DIR, csv_path: Path = DEFAULT_CSV_PATH,
                   n_examples: int = DEFAULT_N_EXAMPLES) -> dict | None:
    """Returns the example payload, or None if the Transformer artifact is
    absent -- same graceful-skip contract as ml/evaluate_model.py."""
    trajectory = load_trained_model(artifact_dir)
    if trajectory is None:
        return None
    model, metadata = trajectory

    split = load_and_split(csv_path)
    test_df = split["test"].reset_index(drop=True)
    # Same split-boundary fix as ml/evaluate_model.py: without this, a
    # merchant's first test weeks show attention collapsed onto the current
    # week purely because their real prior weeks sit in the validation split.
    full_history = pd.concat([split["train"], split["validation"], split["test"]], ignore_index=True)
    probabilities = model.predict(test_df, history_df=full_history)
    attention = model.attention_by_week(test_df, history_df=full_history)
    threshold = metadata["selected_threshold"]

    # Highest-scoring flagged rows: the ones a reviewer would actually open,
    # and therefore the ones an explanation has to hold up for.
    flagged = np.flatnonzero(probabilities >= threshold)
    chosen = flagged[np.argsort(-probabilities[flagged])][:n_examples]

    window = metadata["window"]
    examples = []
    for row in chosen:
        weights = attention[row]
        # Positions are relative offsets: 0 = the week being scored,
        # -k = k weeks earlier. Zero weight means that week is padding
        # (the merchant had no history that far back), not "ignored".
        by_week = [
            {"weeks_before_current": int(offset - (window - 1)), "attention": round(float(weights[offset]), 4)}
            for offset in range(window)
        ]
        examples.append({
            "merchant_id": str(test_df.loc[row, "merchant_id"]),
            "week_start": str(test_df.loc[row, "week_start"]),
            "predicted_probability": round(float(probabilities[row]), 4),
            "actual_label": int(test_df.loc[row, LABEL_COLUMN]),
            "attention_by_week": by_week,
            "peak_week_offset": int(np.argmax(weights) - (window - 1)),
        })

    return {
        "model_name": metadata["model_name"],
        "model_version": MODEL_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "window": window,
        "selected_threshold": threshold,
        "split": "held_out_test",
        "role": "explanation_illustration_only",
        "statement": (
            "Synthetic data. These attention weights illustrate a temporal explanation "
            "modality and are not used for live case scoring or shown to merchants. "
            "Live explanations come from the Logistic Regression feature contributions."
        ),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump trajectory-Transformer attention examples.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--n-examples", type=int, default=DEFAULT_N_EXAMPLES)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    payload = build_examples(artifact_dir, Path(args.csv_path), args.n_examples)
    if payload is None:
        print("[explain_trajectory_examples] Trajectory Transformer artifact not found -- nothing written. "
              "Run `python3 -m ml.train_trajectory_transformer` first.")
        return

    output_path = artifact_dir / OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[explain_trajectory_examples] wrote {len(payload['examples'])} examples to {output_path}")


if __name__ == "__main__":
    main()
