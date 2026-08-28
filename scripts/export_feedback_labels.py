"""Exports reviewer-resolved case outcomes (FALSE_POSITIVE / CONFIRMED_RISK
only) into a label-override file for the manually-triggered retraining
loop (Phase 2). See docs/PHASE_2_FEEDBACK_RETRAINING_DESIGN.md.

Run this any time after reviewing some cases, before running
`python3 -m ml.retrain_with_feedback`. Safe to re-run -- it always
reflects the current database state, overwriting the previous export.

Usage:
    python3 scripts/export_feedback_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.services.feedback_service import get_feedback_label_overrides

DEFAULT_OUTPUT_PATH = Path("ml/artifacts/feedback_label_overrides.json")


def export(output_path: Path = DEFAULT_OUTPUT_PATH) -> list[dict]:
    engine = create_db_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_scope(session_factory) as session:
        overrides = get_feedback_label_overrides(session)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    return overrides


def main() -> None:
    overrides = export()
    false_positive_count = sum(1 for o in overrides if o["final_outcome"] == "FALSE_POSITIVE")
    confirmed_risk_count = sum(1 for o in overrides if o["final_outcome"] == "CONFIRMED_RISK")
    print(f"Exported {len(overrides)} feedback label override(s) to {DEFAULT_OUTPUT_PATH}")
    print(f"  FALSE_POSITIVE (label -> 0): {false_positive_count}")
    print(f"  CONFIRMED_RISK (label -> 1): {confirmed_risk_count}")
    print("OPERATIONAL_ISSUE and INCONCLUSIVE resolutions are excluded -- their correct label is a judgment call this project does not make silently.")


if __name__ == "__main__":
    main()
