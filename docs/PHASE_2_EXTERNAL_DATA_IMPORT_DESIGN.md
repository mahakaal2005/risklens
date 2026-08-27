# Phase 2 Design — External Anonymized Data Import (as-built)

**Status: implemented, on branch `phase-2-auth-design`.** Per `CLAUDE.md`'s Phase 2 roadmap: "Import anonymized CSV data from a willing merchant." No willing merchant exists yet, so this is exercised with a clearly-labeled synthetic fixture standing in for a real merchant export — never claimed to be real data. Two decisions were confirmed with the user before implementation (Section 2).

## 1. Goal

Let an already-aggregated merchant-week CSV — the shape a real merchant's own analytics export might take, as opposed to raw transaction/payment events (`ml/razorpay_adapter.py`'s job) — be validated, scored through the existing rules+model pipeline, and turned into real review cases, exactly like the synthetic demo cases already are.

This is distinct from the Razorpay-shaped adapter (`ml/razorpay_adapter.py`): that ingests raw payment/refund/dispute/settlement **events** and aggregates them into merchant-week rows. This feature ingests an **already merchant-week-shaped** file directly — the two are complementary import paths into the same downstream workflow, not duplicates.

## 2. Decisions (confirmed with the user)

1. **Exact schema match, no flexible column-name mapping in v1.** The CSV must use ClearRisk's own internal raw column names — the same schema `ml/data_validation.py` already validates for the synthetic dataset, minus the two synthetic-only columns (`label_high_loss_next_30d`, `latent_state_for_demo_only`). A real merchant's actual export would need to be reshaped to this schema first — a documented limitation, not silently handled with guesswork.
2. **Reject the entire file** on any prohibited/PII-suggestive column or the presence of a synthetic-only column. Never silently drop the offending column and import the rest.

## 3. A real gap found and fixed during implementation

The first validation pass reused `ml.data_validation.PROHIBITED_FIELD_NAMES` as-is — an **exact** column-name match set (`"email"`, `"phone_number"`, etc.). A live test with a column named `customer_email` (not `email`) **passed validation and was scored and persisted** — the exact-match check doesn't catch realistic real-world column naming variance at all. Fixed with a substring-based check (`PII_SUBSTRING_KEYWORDS`) that catches `customer_email`, `buyer_phone_number`, `cardholder_pan`, etc. — verified live that the same file is now correctly rejected with `"Column name(s) suggest PII and are rejected on suspicion: ['customer_email']"`. This is exactly the kind of finding the project's standing 5-security-checks requirement exists to catch, and it's flagged here rather than only fixed silently.

## 4. A second gap found and fixed: the API never returned the persisted per-case notice

`build_case_packet()`'s built-in `synthetic_data_notice` text ("generated from synthetic, demonstration-only data") is correct for the internal synthetic generator but would be **false** if applied to genuinely real anonymized data. The fix: override `packet["identification"]["synthetic_data_notice"]` with an import-specific message before persistence (`EXTERNAL_IMPORT_DATA_NOTICE`, naming the actual fixture-label source).

Verifying this live surfaced a **pre-existing** gap, not introduced by this feature: `app/api/routes/cases.py`'s `_to_case_summary()`/`_to_case_detail()` never mapped `case.synthetic_data_notice` (the DB column) into the response at all — every case, imported or synthetic, silently returned the generic Pydantic-default notice string (`"Local synthetic-data demonstration only."`) regardless of what was actually persisted. This was fixed as part of this feature (both functions now pass `synthetic_data_notice=case.synthetic_data_notice` explicitly), since otherwise the entire point of Section 4's fix — an accurate, source-specific notice — would have been silently defeated by an unrelated existing bug. Verified live: an imported case and a synthetic demo case now return two distinct, correct notice strings via `GET /cases/{id}`.

## 5. Pipeline

```text
demo_data/external_import_fixtures/anonymized_merchant_export_demo.csv
  (synthetic fixture standing in for a real merchant export -- 3 rows,
  3 distinct risk profiles: stable/APPROVE, high-risk/MANUAL_REVIEW_REQUIRED
  or REQUEST_EVIDENCE-shaped, operational-issue-shaped)
    |
    v
ml/external_data_import.py::validate_import_dataframe()
  -- schema match, PII/prohibited-field rejection (both exact and
     substring), synthetic-only-column rejection, merchant_id/week_start/
     range/duplicate checks -- reuses ml.data_validation's
     NON_NEGATIVE_COLUMNS/RATE_COLUMNS_0_1/PROHIBITED_FIELD_NAMES rather
     than re-deriving them
    |
    v
ml/external_data_import.py::build_mapping_report()
  -- aggregate-only mapping/data-quality report: columns found/missing,
     row count, merchant count, date range, prohibited/PII findings.
     Never includes a raw row.
    |
    v
ml/external_data_import.py::score_import_rows()
  -- reuses ml.rules_engine.score_merchant_week(), ml.model_utils.combined_policy(),
     and ml.case_packet.build_case_packet() -- the EXACT same functions the
     synthetic demo cases go through, no separate/duplicated packet-building
     logic for imported data
    |
    v
scripts/import_merchant_csv.py (DB-touching, mirrors scripts/seed_demo_cases.py)
  -- persists every non-APPROVE packet via
     app.services.case_service.create_case_from_packet() -- the same
     workflow entry point every other case in this system uses
    |
    v
ml/artifacts/external_import_report.json
  -- the full mapping/data-quality report plus cases_created/cases_approved_no_case counts
```

## 6. Live end-to-end verification performed

- The bundled 3-row fixture validated and scored correctly: 1 stable merchant → `APPROVE` (no case created), 1 high-risk merchant → `MANUAL_REVIEW_REQUIRED`, 1 operational-issue-shaped merchant → `REQUEST_EVIDENCE` — both created as real, queryable cases alongside the synthetic demo cases.
- Confirmed via the running API that the imported cases' `synthetic_data_notice` correctly differs from a synthetic demo case's notice (Section 4).
- Confirmed the `customer_email` rejection (Section 3) against the real validation path, not just a unit test.
- Confirmed a case created from an imported row is otherwise indistinguishable in the workflow from a synthetic one — same state machine, same reviewer actions, same audit trail, same evidence/SLA handling.

## 7. Known limitations (not fabricated, flagged instead)

- No flexible column-name mapping (Section 2, decision 1) — a real merchant's actual export format would need reshaping to this exact schema first.
- The PII substring check (`PII_SUBSTRING_KEYWORDS`) is a fixed, hand-curated keyword list — not exhaustive, and can both over-reject (a legitimate column containing e.g. `"card"` as a substring of an unrelated word) and under-reject (a PII field named something the list doesn't anticipate). This is a heuristic safety net, not a certified PII-detection system.
- No column-level data type coercion beyond what `pandas.read_csv` infers automatically — a numeric column stored as text with an unexpected format would surface as a validation failure rather than being silently coerced.
- No deduplication against already-imported data across multiple import runs beyond the existing `create_case_from_packet()` merchant+week uniqueness check (safe re-running, not accidental duplication).

## 8. Tests

- `tests/test_external_data_import.py` (17 tests) — schema validation (valid, missing column, exact prohibited field, PII-substring rejection, synthetic-only-column rejection, unsafe merchant_id, bad date, negative value, out-of-range rate, duplicate row), mapping report contents (including that it never contains a raw row/value), packet scoring (correct import-specific notice, no label/latent-state leakage, degraded mode without a model).
- `tests/test_import_merchant_csv.py` (3 tests) — the bundled fixture imports successfully end-to-end, an invalid CSV is rejected with zero cases created, and an imported case's notice is confirmed correct through the real FastAPI route (not just the packet dict in isolation).
