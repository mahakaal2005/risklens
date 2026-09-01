"""Probability calibration: are the model's stated probabilities honest?

Why this matters here specifically. A reviewer reading "0.86" reasonably assumes
roughly 86 in 100 such merchants actually deteriorate; if the true rate is 0.40,
that reading is wrong even when ranking (PR-AUC) is excellent. Discrimination and
calibration are different properties, and this project reports both rather than
letting a good PR-AUC imply a trustworthy number.

**What calibration does NOT affect here:** ml/cost_model.py is counts-based. It
derives TP/FP/FN/TN from held-out labels and multiplies those integer counts by
fixed rupee constants; the probability is used only as a cut point in
`probs >= threshold`, never as a magnitude. Because monotone calibration
preserves ordering, it preserves the set of achievable partitions, so the
minimum achievable cost is invariant to it -- verified empirically: on a
fine threshold grid, raw and isotonic-calibrated Logistic Regression minimise at
Rs18,642,800 and Rs18,641,200 respectively (a 0.009% discretisation residue).
Calibration matters for how a number is *read*, not for the cost arithmetic.

Two standard post-hoc calibrators are compared against the raw model:

- **Platt scaling** -- a logistic regression on the model's log-odds. Two
  parameters, so it cannot overfit a small calibration set, but it can only
  apply a monotonic sigmoid correction.
- **Isotonic regression** -- a free monotonic step function. Strictly more
  flexible, and correspondingly more able to overfit; with enough calibration
  data it usually wins, with too little it memorises noise.

**Methodological disclosure, stated up front because it is a real limitation:**
the calibrators are fit on the *validation* split, which this project already
uses for threshold selection. Validation therefore does double duty. The clean
alternative is a fourth dedicated calibration split, which would shrink every
other split on an already-synthetic dataset. The held-out test set is still
never touched by any fitting step, so test-set calibration metrics remain
honest -- but a calibrator fit on the same data that chose the threshold is
slightly more optimistic than one fit on genuinely fresh data, and that is
disclosed rather than hidden.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DEFAULT_N_BINS = 10
# Probabilities are clipped before the logit transform so a 0.0 or 1.0
# prediction cannot produce an infinite log-odds.
LOGIT_CLIP = 1e-6


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome.

    Lower is better. Unlike PR-AUC it is a *proper scoring rule*: it is
    minimised only by reporting the true probability, so it penalises a model
    that ranks perfectly but states overconfident numbers.
    """
    return float(np.mean((probs - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, probs: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> list[dict]:
    """Observed positive rate vs. mean predicted probability, per bin.

    A perfectly calibrated model sits on the diagonal: among rows it scored
    ~0.3, ~30% are actually positive. Empty bins are reported with a null
    observed rate rather than dropped, so a reader can see where the model
    makes no predictions at all.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize is right-open; subtracting 1 and clipping puts p == 1.0 in
    # the last bin rather than in an out-of-range bin of its own.
    bin_index = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    curve = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        curve.append({
            "bin_lower": round(float(edges[b]), 4),
            "bin_upper": round(float(edges[b + 1]), 4),
            "count": count,
            "mean_predicted_probability": round(float(probs[mask].mean()), 4) if count else None,
            "observed_positive_rate": round(float(y_true[mask].mean()), 4) if count else None,
        })
    return curve


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Count-weighted mean gap between predicted and observed rate.

    0.0 is perfect. Reported alongside Brier because Brier mixes calibration
    and discrimination together, while ECE isolates the calibration gap --
    though ECE is bin-count sensitive, so it is a summary, not a verdict.
    """
    total = 0.0
    n_rows = len(y_true)
    for entry in reliability_curve(y_true, probs, n_bins):
        if entry["count"]:
            gap = abs(entry["mean_predicted_probability"] - entry["observed_positive_rate"])
            total += entry["count"] / n_rows * gap
    return float(total)


def maximum_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Worst single-bin calibration gap.

    Included because ECE's count weighting hides bad behaviour in exactly the
    place it matters most here: the sparse high-probability bins that generate
    the escalations a reviewer acts on.
    """
    gaps = [
        abs(e["mean_predicted_probability"] - e["observed_positive_rate"])
        for e in reliability_curve(y_true, probs, n_bins)
        if e["count"]
    ]
    return float(max(gaps)) if gaps else 0.0


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(clipped / (1.0 - clipped))


class PlattCalibrator:
    """Logistic regression on the model's log-odds. Two parameters."""

    name = "platt"

    def __init__(self):
        self._model = LogisticRegression(solver="lbfgs")

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self._model.fit(_logit(probs).reshape(-1, 1), y_true)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(_logit(probs).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Free monotonic step function. More flexible, easier to overfit."""

    name = "isotonic"

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(probs, y_true)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(probs), dtype=float)


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> dict:
    return {
        "brier_score": round(brier_score(y_true, probs), 6),
        "expected_calibration_error": round(expected_calibration_error(y_true, probs, n_bins), 6),
        "maximum_calibration_error": round(maximum_calibration_error(y_true, probs, n_bins), 6),
        "mean_predicted_probability": round(float(probs.mean()), 6),
        "observed_positive_rate": round(float(y_true.mean()), 6),
        "reliability_curve": reliability_curve(y_true, probs, n_bins),
    }


def calibrate_and_evaluate(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """Fit both calibrators on validation, score all three variants on test.

    The test set is scored only; nothing is fit on it. Returns raw/platt/
    isotonic metrics plus the calibrated test probabilities, so a caller can
    re-derive an operating threshold on the calibrated scale if it wants one.
    """
    variants = {"raw": test_probs}
    for calibrator_class in (PlattCalibrator, IsotonicCalibrator):
        calibrator = calibrator_class().fit(val_probs, y_val)
        variants[calibrator.name] = calibrator.transform(test_probs)

    metrics = {name: calibration_metrics(y_test, probs, n_bins) for name, probs in variants.items()}

    # "Best" strictly by Brier, which is the proper scoring rule of the three
    # numbers reported. Named explicitly so the choice is not mistaken for a
    # composite judgement across ECE and MCE too.
    best = min(metrics, key=lambda name: metrics[name]["brier_score"])

    return {
        "metrics": metrics,
        "best_by_brier_score": best,
        "calibrated_test_probabilities": variants,
        "fit_on": "validation_split",
        "methodology_note": (
            "Calibrators are fit on the validation split, which this project also uses for "
            "threshold selection -- validation does double duty. The held-out test set is "
            "scored only, never fit on, so these test metrics are honest; but a calibrator fit "
            "on genuinely fresh data would be a stricter test."
        ),
    }
