"""Demonstrates two full review-case workflows against the seeded demo
cases: a seasonal-sale false-positive resolution, and an operational-issue
resolution. Run scripts/seed_demo_cases.py first (against a fresh database)
so both cases exist in OPEN status.

Usage:
    python3 scripts/seed_demo_cases.py
    python3 scripts/demo_case_workflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.db.repositories import get_case
from app.services.audit_service import get_case_timeline
from app.services.case_service import apply_reviewer_action, start_review
from app.services.evidence_service import submit_evidence

DEMO_PACKETS_PATH = Path("demo_data/demo_case_packets.json")


def _print_timeline(session, case_id: str) -> None:
    timeline = get_case_timeline(session, case_id)
    for event in timeline:
        print(f"  [{event['event_sequence_number']}] {event['event_type']} (actor={event['actor_type']}:{event['actor_id']}) at {event['event_timestamp']}")


def run_seasonal_sale_workflow(session_factory, case_id: str) -> None:
    print("\n=== Seasonal-sale false-positive workflow ===")
    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        print(f"1. Loaded seeded case {case_id}, status={case.case_status}")

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "REQUEST_EVIDENCE", "Please share refund records and an explanation for the recent volume/refund increase.")
        print("2. Reviewer requested evidence.")

    with session_scope(session_factory) as session:
        submit_evidence(
            session, case_id,
            "We ran a seasonal promotional sale this period, which increased both order volume and returns. "
            "Attached are refund records and our refund policy for reference.",
            ["invoice_demo_001.pdf", "refund_policy_demo_url"],
        )
        print("3. Merchant submitted simulated refund records and volume-change explanation.")

    with session_scope(session_factory) as session:
        start_review(session, case_id)
        print("4. Reviewer began review.")

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "MARK_FALSE_POSITIVE", "Refund spike is explained by a legitimate seasonal sale; chargeback rate stayed normal. No further concern.")
        print("5. Reviewer marked FALSE_POSITIVE.")

    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        print(f"\n6. Final case state: status={case.case_status}, final_outcome={case.final_outcome}, reviewer_note={case.reviewer_note!r}")
        print("Complete ordered audit timeline:")
        _print_timeline(session, case_id)


def run_operational_issue_workflow(session_factory, case_id: str) -> None:
    print("\n=== Operational fulfilment-issue workflow ===")
    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        print(f"1. Loaded seeded case {case_id}, status={case.case_status}")

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "REQUEST_EVIDENCE", "Please share fulfilment/delivery proof and recent support ticket records.")
        print("2. Reviewer requested evidence.")

    with session_scope(session_factory) as session:
        submit_evidence(
            session, case_id,
            "We had a temporary fulfilment delay due to a warehouse system migration, which increased support "
            "tickets and slowed delivery confirmation. Attached is our delivery proof and support log summary.",
            ["delivery_proof_demo_001.pdf", "support_log_summary_demo"],
        )
        print("3. Merchant submitted simulated fulfilment and support references.")

    with session_scope(session_factory) as session:
        start_review(session, case_id)
        print("4. Reviewer began review.")

    with session_scope(session_factory) as session:
        apply_reviewer_action(session, case_id, "MARK_OPERATIONAL_ISSUE", "Confirmed a temporary operational/fulfilment issue, not fraud. Merchant has a remediation plan in place.")
        print("5. Reviewer marked OPERATIONAL_ISSUE.")

    with session_scope(session_factory) as session:
        case = get_case(session, case_id)
        print(f"\n6. Final case state: status={case.case_status}, final_outcome={case.final_outcome}, reviewer_note={case.reviewer_note!r}")
        print("Audit timeline:")
        _print_timeline(session, case_id)


def main() -> None:
    with open(DEMO_PACKETS_PATH, "r", encoding="utf-8") as f:
        packets = json.load(f)

    engine = create_db_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    seasonal_case_id = packets["seasonal_sale_false_positive_candidate"]["identification"]["case_preview_id"]
    operational_case_id = packets["operational_fulfilment_problem"]["identification"]["case_preview_id"]

    run_seasonal_sale_workflow(session_factory, seasonal_case_id)
    run_operational_issue_workflow(session_factory, operational_case_id)


if __name__ == "__main__":
    main()
