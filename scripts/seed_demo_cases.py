"""Reads the five Milestone 4 demo case packets and persists a review case
for each non-APPROVE packet. Run once against a fresh database; re-running
against an already-seeded database is safe (existing cases are reused, not
duplicated) but will not re-create already-resolved demo state.

Usage:
    python3 scripts/seed_demo_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.services.case_service import create_case_from_packet

DEMO_PACKETS_PATH = Path("demo_data/demo_case_packets.json")


def seed(packets_path: Path = DEMO_PACKETS_PATH) -> list[dict]:
    with open(packets_path, "r", encoding="utf-8") as f:
        packets = json.load(f)

    engine = create_db_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    results = []
    for case_name, packet in packets.items():
        with session_scope(session_factory) as session:
            case, audit_events = create_case_from_packet(session, packet)
            if case is None:
                results.append({"demo_case_name": case_name, "case_id": None, "status": "NOT_CREATED (APPROVE)"})
            else:
                results.append({"demo_case_name": case_name, "case_id": case.case_id, "status": case.case_status})
    return results


def main() -> None:
    results = seed()
    for r in results:
        print(f"{r['demo_case_name']}: case_id={r['case_id']} status={r['status']}")
    created = [r for r in results if r["case_id"] is not None]
    print(f"\n{len(created)} of {len(results)} demo packets became persisted review cases.")


if __name__ == "__main__":
    main()
