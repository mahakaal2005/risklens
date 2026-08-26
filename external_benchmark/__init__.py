"""External benchmark: a standalone transaction-level fraud-classification
experiment on the public IEEE-CIS Fraud Detection dataset.

This package is intentionally isolated from ClearRisk Recover's core
merchant-week model, rules engine, API, database, and dashboard -- see
docs/EXTERNAL_BENCHMARK.md and docs/EXTERNAL_BENCHMARK_DESIGN.md for the
full data boundary and non-claims. Nothing in ml/, app/, rules/, or
dashboard/ imports from this package, and this package imports nothing
from them.
"""
