# Phase 2 Design — Feedback → Retraining Loop (as-built)

**Status: implemented, on branch `phase-2-auth-design`.** Per `CLAUDE.md`'s Phase 2 roadmap: "Add model feedback/retraining workflow — but no automatic retraining." Two decisions were confirmed with the user before implementation (Section 2).

## 1. Goal

Turn reviewer resolutions — the signal already captured every time a case is marked `FALSE_POSITIVE`, `CONFIRMED_RISK`, `OPERATIONAL_ISSUE`, or `INCONCLUSIVE` — into a real, manually-triggered retraining loop with an honest before/after comparison on the held-out test set. This closes the loop the dashboard already implies (a reviewer resolves a case) but that previously went nowhere (the resolution was never fed back into the model).

## 2. Decisions (confirmed with the user)

1. **Only `FALSE_POSITIVE` (→ label 0) and `CONFIRMED_RISK` (→ label 1) correct a training label.** `OPERATIONAL_ISSUE` and `INCONCLUSIVE` are excluded entirely — their correct label is a genuine judgment call ("was this actually the same latent risk state, just for an operational reason?") that this project does not make silently by guessing.
2. **Label corrections never touch the held-out test split.** They apply only to rows in the time-based training split. The test split stays byte-for-byte the same data across a retrain, so before/after metrics on it are a fair comparison — the entire reason a held-out set exists.

## 3. Architecture: two-step, manually triggered, no shared app/ml dependency

```text
Step 1 (touches the database):
  scripts/export_feedback_labels.py
    -> app/services/feedback_service.py::get_feedback_label_overrides()
       -- reads ReviewCase rows where final_outcome in (FALSE_POSITIVE, CONFIRMED_RISK)
    -> writes ml/artifacts/feedback_label_overrides.json

Step 2 (pure ml/, no app/ dependency -- mirrors every other ml/ module):
  python3 -m ml.retrain_with_feedback
    -> loads the ORIGINAL synthetic CSV, re-derives the same time-based
       train/validation/test split (ml/split_data.py, unchanged)
    -> applies label overrides from Step 1's JSON file, ONLY to rows that
       exist in the training split (apply_feedback_to_training_split())
    -> retrains a fresh Logistic Regression pipeline on the corrected
       training data (same preprocessing/feature set as
       ml/train_baseline_model.py), selects a threshold on validation
    -> saves the new model under a DISTINCT version
       (logistic_regression_v0.1.0-feedback1.joblib) -- never overwrites
       the original baseline artifact
    -> loads the EXISTING baseline model and evaluates BOTH baseline and
       feedback-retrained models on the IDENTICAL, untouched held-out test
       split (ml.evaluate_model.evaluate_split(), reused as-is)
    -> writes ml/artifacts/feedback_retrain_report.json with the full
       before/after comparison and an exact accounting of every override
       (applied, with its original and corrected label; or skipped, with a
       specific reason)
```

Both steps are always run manually from the command line. **Nothing in this codebase calls either step automatically** — no scheduler, no webhook, no "auto-retrain on N new resolutions" trigger.

## 4. Every override is accounted for, never silently dropped

`apply_feedback_to_training_split()` returns both `applied` (with the original label the override replaced) and `skipped` (each with a specific, human-readable reason: the week isn't in the training split, or the label already matched). The report always states how many overrides existed vs. how many actually changed something, so an operator running this can never mistake "0 applied" for "the export was empty."

## 5. A real, honest finding from end-to-end verification

Live end-to-end testing against the actual seeded demo database surfaced a genuine, worth-documenting behavior: **the demo case packets are deliberately the most recent merchant-weeks** (`docs/DATA_DICTIONARY.md`'s design), which fall in the dataset's final ~15% of weeks — the held-out **test** split, by construction. Resolving `seasonal_sale_false_positive_candidate` (FALSE_POSITIVE) and `high_risk_combined_loss_case` (CONFIRMED_RISK) and running the full loop produced **0 applied, 2 skipped** — both correctly rejected as "no matching row in the training split." This is the safety mechanism working exactly as designed, not a bug — but it does mean **the out-of-the-box demo cases alone will never move the retrained model**, since they're all in the test period. A separate live test using a hand-constructed override for an early (training-period) merchant-week confirmed the "applied" path works correctly end-to-end (1 applied, 0 skipped, a new distinctly-versioned model artifact produced) — but a single flipped label out of thousands of training rows, as expected, did not move aggregate precision/recall in a way large enough to see in this demo's scale. This is expected statistical behavior, not a defect, and is noted here so a future demo/pitch doesn't overclaim a dramatic before/after story from one or two resolved cases.

## 6. Known limitations (not fabricated, flagged instead)

- Demo review cases are clustered in the test-split time period (by the data design), so realistic reviewer feedback on the bundled demo cases will typically be skipped for retraining. A real deployment with cases spread across the full time range (or feedback accumulated over many weeks) would behave differently.
- A small number of label corrections is unlikely to visibly move aggregate metrics — this loop demonstrates the *mechanism* (safe, auditable, test-set-preserving), not a guaranteed "the model got better" story from a handful of resolutions.
- No versioning/registry beyond a fixed `-feedback1` suffix — running the retrain twice overwrites the previous feedback model file. A real iterative loop would need a proper versioning scheme (out of scope for this pass).
- No API endpoint exposes this workflow; it is CLI-only by design, matching `ml.evaluate_model`'s existing pattern and keeping "manually triggered" unambiguous (no route means no accidental automated trigger via a script that calls the API).

## 7. Tests

- `tests/test_feedback_service.py` (5 tests) — only `FALSE_POSITIVE`/`CONFIRMED_RISK` are exported, correct label mapping, empty-database case.
- `tests/test_retrain_with_feedback.py` (8 tests) — override application/skip logic (matching row, out-of-split row, already-matching label), a full retrain producing a report even with zero feedback, a missing-baseline-model error, a full retrain that applies a real override and confirms the baseline artifact is untouched, and a check that the report's language never overclaims (says "untouched," "manually triggered").
- Live end-to-end verification against the real seeded demo database and the real 220-merchant synthetic dataset (Section 5) — not just unit tests with synthetic fixtures.
