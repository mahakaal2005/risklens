# rules/

`risk_rules.yaml` is the configurable rule catalogue for ClearRisk Recover's transparent rules engine (`ml/rules_engine.py`). All numeric thresholds live in this YAML file, not in Python — changing a threshold means editing this file, not the code.

## What this is

A set of 5 transparent rule families, evaluated against a single merchant-week record's features (see `ml/features.py`):

1. `REFUND_RATE_SPIKE` — refund rate increased materially versus the merchant's own recent history.
2. `CHARGEBACK_RATE_SPIKE` — chargeback rate increased materially versus the merchant's own recent history.
3. `EVIDENCE_COVERAGE_GAP` — weak delivery/fulfilment evidence, only counted alongside a refund or chargeback increase (never on its own).
4. `SUPPORT_OPERATIONAL_STRESS` — rising support load and slower resolution, only counted alongside a refund or chargeback increase.
5. `COMBINED_LOSS_SIGNAL` — two or more of the above co-occur, which is a stronger pattern than any one rule alone.

Each rule returns evidence and a risk signal — never an enforcement action. See `docs/MILESTONE_2_RULES_AND_FEATURES.md` for the full catalogue, fixture walkthroughs, and the preliminary score/recommendation mapping.

## What this is not

- Not a fraud determination. A triggered rule means "worth a human reviewer's attention," not "confirmed loss."
- Not able to freeze funds, hold settlement, ban a merchant, terminate an account, or reject a payment. The only outputs are `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`.
- Not tuned against real merchant data — every threshold here is an illustrative, synthetic-data-only value (see the "why illustrative" section of `docs/MILESTONE_2_RULES_AND_FEATURES.md`).

## Changing a threshold

Edit the relevant rule's `thresholds` block in `risk_rules.yaml`, then re-run `python3 -m pytest tests/test_rules_engine.py -v` — the fixture tests will catch a threshold change that flips an expected trigger/non-trigger outcome.
