"""Rupee cost model: turns a confusion matrix into an expected cost, selects a
cost-optimal threshold, and reports savings per 1,000 reviews.

Why this exists: precision, recall and PR-AUC do not say whether a model is
worth running. A method with worse recall can be the better operational choice
if its false positives are cheap and its misses are not. This module makes that
tradeoff explicit in rupees instead of leaving it implicit in an F2 score.

**Every rupee figure is an assumption**, loaded from rules/cost_model.yaml --
see that file's header. No public dataset exists to calibrate per-merchant
chargeback costs against, and provider cost structures are confidential. The
sensitivity analysis below is therefore not optional decoration: it is the part
that makes the result usable by someone whose costs differ from the guess.

Threshold-selection discipline is identical to ml/model_utils.py::select_threshold():
the cost-optimal threshold is chosen on **validation data only** and the
held-out test set is never consulted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ml.model_utils import THRESHOLD_GRID

DEFAULT_COST_CONFIG_PATH = Path("rules/cost_model.yaml")

# The 0.05-step THRESHOLD_GRID shared with F2 selection is too coarse to locate
# a cost minimum: the Day 2 analysis reported 0.05 (the grid floor) when the
# true optimum for Logistic Regression is 0.0605, between two grid points. This
# finer grid is used only for the cost analysis -- the F2 operating threshold
# stays on the coarse grid so the live scoring path is unchanged.
FINE_THRESHOLD_GRID = np.round(np.linspace(0.0, 1.0, 2001), 5)


def load_cost_config(path: Path = DEFAULT_COST_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    costs = config["costs_inr"]
    for key in ("review_cost", "missed_loss", "false_positive_friction"):
        if costs[key] < 0:
            raise ValueError(f"cost_model.yaml: costs_inr.{key} must be non-negative, got {costs[key]}")
    if not 0.0 <= config["recovery_rate"] <= 1.0:
        raise ValueError(f"cost_model.yaml: recovery_rate must be in [0, 1], got {config['recovery_rate']}")
    return config


def expected_cost(confusion: dict, config: dict) -> float:
    """Total expected rupee cost of one method's decisions.

    Per outcome:
      TP -- we reviewed and caught it: review cost, plus the share of the loss
            that early review does NOT avoid ((1 - recovery_rate) * missed_loss).
            Catching a case is not free and is not perfectly effective.
      FP -- we reviewed a healthy merchant: review cost + friction cost.
      FN -- we missed it entirely: the full missed loss.
      TN -- correctly left alone: zero.
    """
    costs = config["costs_inr"]
    residual_loss = (1.0 - config["recovery_rate"]) * costs["missed_loss"]

    return float(
        confusion["tp"] * (costs["review_cost"] + residual_loss)
        + confusion["fp"] * (costs["review_cost"] + costs["false_positive_friction"])
        + confusion["fn"] * costs["missed_loss"]
    )


def do_nothing_cost(confusion: dict, config: dict) -> float:
    """Cost of the honest baseline this project must beat to justify existing:
    review nobody, absorb every loss. Every actual positive (tp + fn) becomes a
    miss; no review cost is incurred because no review happens.

    Using "review nobody" rather than "review everybody" as the baseline is
    deliberate -- it is what a team without this system actually does.
    """
    actual_positives = confusion["tp"] + confusion["fn"]
    return float(actual_positives * config["costs_inr"]["missed_loss"])


def review_all_cost(confusion: dict, config: dict) -> float:
    """The other trivial policy: review every merchant-week. Included so the
    model's savings are not compared only against the weaker of the two
    do-nothing options."""
    costs = config["costs_inr"]
    residual_loss = (1.0 - config["recovery_rate"]) * costs["missed_loss"]
    actual_positives = confusion["tp"] + confusion["fn"]
    actual_negatives = confusion["tn"] + confusion["fp"]

    return float(
        actual_positives * (costs["review_cost"] + residual_loss)
        + actual_negatives * (costs["review_cost"] + costs["false_positive_friction"])
    )


def cost_metrics(confusion: dict, config: dict) -> dict:
    """Full rupee summary for one method at one operating point."""
    total = expected_cost(confusion, config)
    baseline = do_nothing_cost(confusion, config)
    review_all = review_all_cost(confusion, config)

    n_rows = sum(confusion[k] for k in ("tn", "fp", "fn", "tp"))
    n_reviews = confusion["tp"] + confusion["fp"]
    net_savings = baseline - total

    return {
        "expected_cost_inr": round(total, 2),
        "do_nothing_cost_inr": round(baseline, 2),
        "review_all_cost_inr": round(review_all, 2),
        "net_savings_vs_do_nothing_inr": round(net_savings, 2),
        "net_savings_vs_review_all_inr": round(review_all - total, 2),
        "n_reviews_generated": int(n_reviews),
        "cost_per_merchant_week_inr": round(total / n_rows, 2) if n_rows else None,
        # The headline operational number: for every 1,000 reviews an analyst
        # actually performs under this policy, how many rupees of loss does the
        # policy avoid relative to doing nothing? Undefined when a method flags
        # nobody -- reported as None rather than as a misleading zero.
        "savings_per_1000_reviews_inr": round(net_savings / n_reviews * 1000, 2) if n_reviews else None,
        "beats_do_nothing": bool(net_savings > 0),
        "beats_review_all": bool(review_all - total > 0),
    }


def confusion_at(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    return {
        "tn": int(((y_true == 0) & (pred == 0)).sum()),
        "fp": int(((y_true == 0) & (pred == 1)).sum()),
        "fn": int(((y_true == 1) & (pred == 0)).sum()),
        "tp": int(((y_true == 1) & (pred == 1)).sum()),
    }


def select_cost_optimal_threshold(y_val: np.ndarray, val_probs: np.ndarray, config: dict) -> dict:
    """Threshold minimizing expected rupee cost on **validation data only**.

    Deliberately reported alongside, not instead of, the F2-selected operating
    threshold. The cost-optimal point depends entirely on the assumed cost
    ratio; presenting it as the single right answer would disguise a guess as a
    result. ml/evaluate_model.py reports both.
    """
    candidates = []
    for t in THRESHOLD_GRID:
        confusion = confusion_at(y_val, val_probs, float(t))
        candidates.append({
            "threshold": float(t),
            "confusion_matrix": confusion,
            **cost_metrics(confusion, config),
        })

    best = min(candidates, key=lambda c: c["expected_cost_inr"])
    # When the optimum lands on the first or last grid point, the true optimum
    # may lie outside the grid entirely and the reported threshold is a floor
    # or ceiling artifact, not a located minimum. Reported rather than hidden.
    at_boundary = best["threshold"] in (float(THRESHOLD_GRID[0]), float(THRESHOLD_GRID[-1]))
    # Refine on a fine grid so the reported optimum is actually located rather
    # than pinned to a coarse grid point. Reported alongside the coarse result
    # so the two are comparable and the Day 2 numbers stay reproducible.
    fine_candidates = [
        (expected_cost(confusion_at(y_val, val_probs, float(t)), config), float(t))
        for t in FINE_THRESHOLD_GRID
    ]
    fine_cost, fine_threshold = min(fine_candidates)
    fine_confusion = confusion_at(y_val, val_probs, fine_threshold)

    return {
        "selected_threshold": best["threshold"],
        "selection_method": "minimum_expected_rupee_cost_on_validation",
        "threshold_at_grid_boundary": bool(at_boundary),
        "refined_threshold": round(fine_threshold, 5),
        "refined_expected_cost_inr": round(fine_cost, 2),
        "refined_confusion_matrix": fine_confusion,
        "refined_grid_step": round(float(FINE_THRESHOLD_GRID[1] - FINE_THRESHOLD_GRID[0]), 5),
        "refinement_note": (
            "refined_threshold is the cost minimum located on a 0.0005-step grid. The coarse "
            "selected_threshold above uses the same 0.05 grid as F2 selection and is kept for "
            "comparability; where the two differ, the coarse grid simply could not express the "
            "optimum. Neither changes the live operating threshold."
        ),
        "grid_boundary_note": (
            f"The cost-optimal threshold sits at the edge of the {THRESHOLD_GRID[0]}-{THRESHOLD_GRID[-1]} "
            f"search grid (step {round(float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]), 4)}), so the grid could "
            "not locate the true minimum: it may lie outside the grid's range OR between two grid points. "
            "Read this as 'the grid could not locate the optimum', not as a located optimum, and not as a "
            "claim about which direction the true optimum lies in."
        ) if at_boundary else None,
        "selected_candidate": best,
        "candidates": candidates,
        "cost_assumptions": config["costs_inr"],
        "recovery_rate": config["recovery_rate"],
        "assumption_notice": (
            "Every rupee figure is an unverified assumption from rules/cost_model.yaml, "
            "not a measured cost. See the sensitivity analysis before relying on this threshold."
        ),
    }


def threshold_sensitivity(y_val: np.ndarray, val_probs: np.ndarray, config: dict) -> list[dict]:
    """How the cost-optimal threshold moves as the missed-loss / review-cost
    ratio changes.

    This is the honest core of the cost analysis. The single "optimal
    threshold" above is only as good as one guessed ratio; this table lets a
    reader with their own ratio read off their own threshold, and shows how
    sharply (or not) the recommendation depends on the guess.
    """
    review_cost = config["costs_inr"]["review_cost"]
    rows = []
    for ratio in config["sensitivity_cost_ratios"]:
        variant = {
            **config,
            "costs_inr": {**config["costs_inr"], "missed_loss": review_cost * ratio},
        }
        result = select_cost_optimal_threshold(y_val, val_probs, variant)
        best = result["selected_candidate"]
        rows.append({
            "missed_loss_to_review_cost_ratio": int(ratio),
            "implied_missed_loss_inr": int(review_cost * ratio),
            "cost_optimal_threshold": result["selected_threshold"],
            "n_reviews_generated": best["n_reviews_generated"],
            "savings_per_1000_reviews_inr": best["savings_per_1000_reviews_inr"],
            "beats_do_nothing": best["beats_do_nothing"],
        })
    return rows
