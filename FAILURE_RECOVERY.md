# Failure Recovery — two real bugs, found and fixed

Two measurement bugs were found in this project's own evaluation pipeline on
2026-09-04, after the numbers already looked plausible enough to ship. Both
are documented here as a recovery story — what looked wrong, how it was
caught, what the actual bug was, how it was fixed, and how the fix was
verified — rather than as a quiet line-item correction. Full technical detail
lives in `MODEL_CARD.md`'s "Corrected measurement errors" section; this is
the narrative version for a live demo or panel discussion.

---

## Bug 1: one model's threshold, applied to every model's probabilities

**What looked wrong.** Random Forest's headline numbers included a false
positive rate of 0.335 — three times worse than every other method — and a
seasonal-sale false-positive count of 1,020, wildly out of line with the
project's own design goal of keeping legitimate seasonal merchants out of the
review queue.

**How it was caught.** Calibration testing (`ml/calibration.py`), added to
check whether each model's stated probability could be trusted as a real
number, surfaced something odd: Random Forest's raw mean predicted
probability was 0.288 against a true base rate of 0.145 — it was stating
roughly double the real risk. That number alone didn't explain the FPR
problem, but it made Random Forest's independently-selected operating
threshold of 0.55 (versus 0.10-0.15 for every other method) look like a real
signal instead of noise worth ignoring.

**The actual bug.** `evaluate_split()` was applying Logistic Regression's
selected threshold (0.10) to *every* method's probabilities in the
comparison table — even though Random Forest had correctly selected 0.55 on
validation data, and Gradient Boosting had selected 0.15. This was never
label leakage (no test-set information reached threshold selection, and each
threshold was legitimately chosen on validation only), but it was an invalid
comparison: scoring one model's probability distribution against another
model's operating point measures the wrong thing entirely. The distortion
was worst for Random Forest specifically because its probability scale was
already inflated ~2x, which the calibration work above had just quantified.

**The fix.** `evaluate_split()` now takes a `thresholds_by_method` mapping
and records `threshold_used` for every method in the evaluation report, so
each method is always scored at its own selected operating point.

**Verified, with before/after numbers:**

| | Published (wrong threshold) | Corrected (own threshold) |
|---|---|---|
| Random Forest FPR | 0.335 | **0.112** |
| Random Forest precision | 0.323 | **0.572** |
| Random Forest recall | 0.944 | **0.881** |
| Random Forest seasonal false positives | 1,020 | **4** |
| Gradient Boosting FPR | 0.177 | **0.114** |
| Gradient Boosting precision | 0.470 | **0.568** |

PR-AUC was unaffected in either direction (0.689 / 0.693 both before and
after), since it's computed over the whole ranking and doesn't depend on a
threshold at all — which is exactly why this bug was invisible in the metric
everyone tends to check first.

---

## Bug 2: history that existed, but wasn't being read

**What looked wrong.** Nothing, at first — this one didn't show up as a bad
number. The Trajectory Transformer's held-out metrics looked reasonable
(PR-AUC 0.6837). The problem only surfaced when inspecting the model's
attention-weight explanations for its own top-confidence predictions: 3 of
the top 6 examples had an attention pattern of `[0, 0, 0, 0, 0, 0, 0, 1.0]`
— every prior week zeroed out, all attention forced onto the current week.
All three sat on the exact same date, the first week of the test split, and
all three produced the identical predicted probability, 0.8659 — a strong
sign the model was scoring one degenerate, identical-shaped input rather
than three genuinely different merchant trajectories.

**The actual bug.** `build_sequences()` only saw rows inside whichever
DataFrame it was handed. At evaluation time, that was the test split alone —
so each merchant's first 7 test weeks (about 7% of test rows) had their
trailing 8-week history window truncated at the split boundary and
zero-padded, even though the real prior weeks existed one split earlier, in
validation. This was not label leakage (only feature vectors were read, no
label ever crossed the boundary), and the effect biased recall *downward*,
the conservative direction — but it was still measuring the wrong thing: "how
well does the trajectory model work when 7% of its inputs have their history
artificially amputated," not the intended question. It specifically
handicapped the one model built to use that history, which is what made the
degenerate attention pattern a clue rather than just noise.

**The fix.** `TrajectoryModel.predict()` and `.attention_by_week()` now
accept an optional `history_df` supplying read-only trailing context across
the split boundary — features only, never labels, transformed through the
already-fit pipeline rather than re-fit on it, with de-duplication ensuring a
target row always wins over any context copy of the same week (so a row can
never serve as its own history). Zero-padding remains as the fallback only
when a merchant's real history is genuinely shorter than the window. This is
directly unit-tested in `tests/test_trajectory_transformer.py`, including an
adversarial case where context rows at or after the target week are poisoned
with a sentinel value and asserted to never leak into the model's input.

**Verified, and the metric delta is almost beside the point:**

| | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| PR-AUC | 0.6837 | 0.6879 | +0.0042 |
| Precision | 0.5689 | 0.5692 | +0.0003 |
| Recall | 0.8821 | 0.8826 | +0.0005 |
| FPR | 0.1133 | 0.1132 | −0.0001 |

Exactly 1 row out of 14,400 flipped false-positive to true-negative, and 1
flipped false-negative to true-positive. The number barely moved. What
changed is the epistemic status of the explanation artifact built on top of
it: post-fix, all 6 top attention examples show genuine 8-week context
instead of 3 of them being an artifact of missing data. Before the fix, that
attention pattern was a measurement error wearing the shape of a finding.
After it, the same shape — attention concentrated on the current week in most
high-confidence cases — is now actual evidence the model learned something,
not a bug producing a coincidentally similar-looking number.

---

## How both were caught, in one sentence

Neither bug was found by a test failing — both were found by refusing to
accept a number or an explanation that looked plausible without asking why
it looked exactly that plausible: an anomalous threshold prompted calibration
testing, and a suspiciously clean attention pattern prompted tracing the
sequence-building code back to where the data actually came from. That
habit — treating an unexplained-but-convenient number as a bug report rather
than a result — is the same discipline behind this project's near-perfect
score investigation gate (`MODEL_CARD.md`), and it's why neither of these two
bugs shipped silently.
