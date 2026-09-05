# Model Card — RiskLens Baseline

Status: Design proposal, rescoped 2026-08-22. See `docs/RESCOPE_REVIEW.md`.

## Model name

RiskLens Logistic Regression Baseline, plus two comparison-only baselines (Random Forest, Gradient Boosting)

## Version

Logistic Regression `0.1.0` (implemented, Milestone 3 — trained and evaluated; see `docs/MILESTONE_3_MODEL_EVALUATION.md` for the full run report). Random Forest / Gradient Boosting `0.1.0` added 2026-08-28 as **evaluation-comparison-only** baselines (see "Comparison baselines" below) — neither is used for live case scoring.

## Model purpose

Estimate the probability that a merchant enters a simulated elevated refund/chargeback-loss state in the next 30 days (`label_high_loss_next_30d`), and support an explainable, human-reviewed recommendation — never an automated decision.

## Intended use

- Local demonstration
- Synthetic-data experimentation
- Explainability and false-positive/false-negative workflow demonstration
- Comparison against a transparent rules-only baseline

## Not intended for use

- Production loss/fraud prevention
- Payment approval/decline
- Merchant account termination
- Settlement holds
- KYC/AML/legal compliance decisions
- Any decision involving real customer money or access to financial services

## Training data

Synthetic, India-inspired merchant-week data generated locally from a 5-state latent merchant simulation (see below).

**Actual data characteristics (seed=42, `python3 -m ml.generate_synthetic_data`, generator `0.2.0`):**

- Date range: 2023-01-02 to 2024-12-23 (104 weeks, 2 years)
- 900 merchants, 93,600 merchant-weeks total
- Label distribution: 14.13% positive overall (`label_high_loss_next_30d`)
- Merchant-category mix (by merchant): digital_services 160, travel 159, grocery 159, apparel 154, electronics 137, food_delivery 131
- Latent-state distribution: stable_merchant 66,073 (70.6%), operational_fulfilment_failure 7,977 (8.5%), seasonal_sale_legitimate_returns 7,412 (7.9%), high_risk_merchant_behaviour 6,917 (7.4%), early_hidden_risk 5,221 (5.6%)
- Generator version: `0.2.0`, seed: `42`

**v0.2.0 change (2026-08-28):** scaled up from the original 220 merchants x 52 weeks (11,440 rows), and added two disclosed noise injections to make the dataset harder and more realistic:
- **Missing data**: a 2% independent null rate on `delivery_evidence_coverage`, `support_ticket_rate`, `average_support_resolution_time_hours`, and `previous_review_outcome` — these fields already had documented missing-value behavior in `ml/features.py` (median/most-frequent imputation), so this exercises that behavior at dataset scale instead of leaving it tested only in unit tests.
- **Anomaly weeks**: ~1.5% of rows get an extreme, state-independent spike in `refund_rate_30d` or `transaction_volume_30d` (with all dependent raw columns — counts, `*_change_30d` fields — recomputed so the row stays internally consistent). The anomaly does not affect the label; it exists purely to prevent the observed features from being trivially separable and to test whether rules/models are robust to noise that isn't a real state change.

Both rates are recorded in `demo_data/synthetic_data_metadata.json`'s `noise_injection` section, reproducibly, per seed.

## Latent-state synthetic-data design

**This section must be finalized and frozen before training, per the "Critical: Honest Synthetic Data and Evaluation" requirement.**

### Why a latent-state design, not rule-derived labels

A naive generator that sets `label_high_loss_next_30d = 1 if chargeback_rate_30d > X and refund_rate_30d > Y else 0` would make the rules engine trivially "discover" the label it was used to create — precision/recall near 1.0 would prove nothing. Instead, the label is generated from a hidden state the rules engine and model never see directly, and observed features are drawn probabilistically (with noise and cross-state overlap) from that hidden state.

### The five latent states

| State | Approx. share of merchant-weeks | Observed-feature tendency | `label_high_loss_next_30d` probability | Purpose |
|---|---|---|---|---|
| 1. Stable | ~70% | Flat volume; low refund (~1-2%) and chargeback (~0.2-0.5%) rates; high delivery-evidence coverage (0.85-0.98); low support-ticket rate | ~1-2% | Baseline population; true negatives |
| 2. Seasonal / legitimate high-return | ~10% | Volume spike; refund rate elevated (~4-8%) but chargeback rate stays normal; evidence coverage still good (0.8-0.95); support resolution reasonable | ~8-12% | **Deliberate false-positive source** — looks risky on refund rate alone but usually resolves clean |
| 3. Operational fulfilment failure | ~8% | Refund rate rising moderately; evidence coverage declining (0.5-0.75); support-ticket rate rising with slower resolution; mild chargeback increase | ~30-40% | Genuine moderate-risk population; true positives |
| 4. High-risk | ~7% | Both refund and chargeback rates elevated and rising sharply; evidence coverage low (0.2-0.5); poor support resolution; `previous_review_outcome` often `confirmed_risk` | ~70-85% | Genuine high-risk population; true positives |
| 5. Early hidden-risk | ~5% | Features close to Stable or only mildly elevated this week (subtle dispute-reason shift, small support-ticket uptick) | ~45-60% | **Deliberate false-negative source** — weak current-week signal despite real forward risk |

### State persistence and transitions

Each merchant is assigned a latent state that persists week-over-week (illustratively ~85% chance of staying, ~15% chance of moving to an adjacent-severity state), producing gradual, realistic drift (e.g. Stable → Operational fulfilment failure → High-risk over several weeks) rather than independent per-week resampling.

### Feature-distribution overlap

Observed weekly fields are drawn from state-conditional distributions (Beta for rates, log-normal/Gaussian for volumes and counts) with independent noise added, and the distributions for adjacent states are designed to overlap (e.g. Seasonal and Operational-failure refund-rate ranges overlap) so that no single week's snapshot trivially reveals the latent state.

### Label horizon

`label_high_loss_next_30d` looks ~4-5 weeks forward from `week_start`. The last ~4-5 weeks of any generated history window cannot have a fully observed label and are excluded from labeled train/validation/test data rather than defaulted to 0.

## Features

**Implemented (Milestone 3) feature list — 11 columns, produced by `ml/features.py::compute_feature_frame()`, matching `ML_FEATURE_COLUMNS` in `ml/model_utils.py`:**

- `refund_rate_change`, `chargeback_rate_change` (absolute change vs. the merchant's own prior 30-day window)
- `transaction_volume_change` (relative change)
- `refund_to_chargeback_ratio` (missing/None when chargeback rate is too close to zero to be numerically stable — see `ml/features.py`)
- `delivery_evidence_gap` (`1 - delivery_evidence_coverage`)
- `support_resolution_hours_normalized`
- `support_ticket_rate`
- `merchant_age_days`
- `transaction_count_30d`
- `merchant_category` (descriptive segmentation only, one-hot encoded at training time)
- `previous_review_outcome` (time-safe — reflects a review that concluded before the current prediction week; one-hot encoded at training time)

No direct identifiers or credentials are permitted. No per-transaction, per-device, or per-customer field is used — those do not exist in the merchant-week entity. `merchant_id`, `week_start`, `label_high_loss_next_30d`, and `latent_state_for_demo_only` are excluded from the feature matrix (verified by `tests/test_train_baseline_model.py::test_design_matrix_excludes_target_latent_id_and_date_fields`).

**Preprocessing:** a scikit-learn `ColumnTransformer` fit on the training split only — `SimpleImputer(median) + StandardScaler` for numeric features, `SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')` for categorical features. See `ml/model_utils.py::build_preprocessing_pipeline()`.

## Target

`label_high_loss_next_30d` — binary, defined via the 5-state latent simulation above. Generation parameters (state shares, probability ranges, transition matrix, feature distributions) are frozen in `ml/generate_synthetic_data.py` and documented above.

## Evaluation

**Persisted report (Milestone 7):** every number in this section is also written to `ml/artifacts/latest_evaluation_report.json` by `python3 -m ml.evaluate_model`, and served read-only (never recomputed on request) by `GET /metrics` — see `docs/MILESTONE_7_METRICS.md` and `docs/API_CONTRACT.md`. Metrics are generated **offline**, by running the evaluation pipeline explicitly; the API process never trains or re-scores a model during a request.

**Split (actual, seed=42):** strict chronological split by unique `week_start` — train: earliest 73 weeks (2023-01-02 to 2024-05-20, 65,700 rows); validation: next 15 weeks (2024-05-27 to 2024-09-02, 13,500 rows); held-out test: latest 16 weeks (2024-09-09 to 2024-12-23, 14,400 rows). No row is shuffled; no week appears in more than one split (verified by `tests/test_split_data.py`).

**Threshold selection:** grid search over [0.05, 0.95] in steps of 0.05 on **validation data only**, maximizing F-beta(2) subject to precision ≥ 0.30 and false-positive rate ≤ 0.20. Selected threshold (Logistic Regression): **0.10** (constraints were met — no fallback needed). Random Forest: **0.55**. Gradient Boosting: **0.15**. Trajectory Transformer: **0.15**. **Every method is evaluated at its own validation-selected threshold** — see "Corrected measurement errors" below for the bug where this was not the case. The held-out test set is never used for threshold selection (verified by `tests/test_train_baseline_model.py::test_held_out_data_cannot_alter_selected_threshold`).

**Held-out test results** (14,400 rows, prevalence 14.49%):

| Method | Threshold | Precision | Recall | F1 | F2 | PR-AUC | FPR | Seasonal FP | Early-hidden FN |
|---|---|---|---|---|---|---|---|---|---|
| Rules-only | — | 0.310 | 0.454 | 0.368 | 0.415 | 0.288 | 0.172 | 450 | 312 |
| Logistic Regression | 0.10 | 0.561 | 0.828 | 0.669 | 0.756 | 0.664 | 0.110 | 62 | 81 |
| Random Forest (comparison only) | 0.55 | 0.572 | 0.881 | 0.694 | 0.795 | 0.689 | 0.112 | 4 | 5 |
| Gradient Boosting (comparison only) | 0.15 | 0.568 | 0.884 | 0.691 | 0.795 | **0.693** | 0.114 | 28 | 0 |
| Trajectory Transformer (comparison only) | 0.15 | 0.569 | 0.883 | 0.692 | 0.795 | 0.688 | 0.113 | 14 | 0 |
| Combined policy (built on Logistic Regression) | 0.10 | 0.403 | 0.870 | 0.551 | 0.706 | 0.664 | 0.219 | 475 | 56 |

Rules-only has no threshold column value because it fires on its own conditions, not a probability cut point.

ROC-AUC (secondary only, not the primary metric per the imbalanced-label caution): rules-only 0.656, Logistic Regression 0.906, Random Forest 0.919, Gradient Boosting 0.921, Trajectory Transformer 0.923, combined 0.906 (held-out test).

## Comparison baselines (Random Forest, Gradient Boosting)

**Design decision: evaluation-only, never used for live scoring.** `ml/train_tree_models.py` trains a `RandomForestClassifier` and a `HistGradientBoostingClassifier` (both ship in scikit-learn — no new dependency) on the exact same preprocessing, split, and threshold-selection method as the Logistic Regression baseline, so any difference in results reflects the model, not an uneven evaluation. Neither model is loaded by `ml/case_packet.py`, `app/services/case_service.py`, or `combined_policy()` — Logistic Regression remains the sole model behind live case creation and the combined policy, per CLAUDE.md's "prefer transparent rules and interpretable models before complex models."

**Honest reading of the comparison, held-out test:**

- **Gradient Boosting** has the highest PR-AUC of any method (0.693) and the highest recall (0.884), at a false-positive rate essentially tied with Logistic Regression's (0.114 vs. 0.110) — a real, honest case that a more complex model *can* modestly outperform Logistic Regression on this synthetic held-out set, at the cost of interpretability.
- **Random Forest** is close behind (PR-AUC 0.689, recall 0.881, FPR 0.112) and produces the fewest seasonal-sale false positives of any method (4).
- **Logistic Regression** has the lowest false-positive rate (0.110), at a real cost in recall (0.828 vs. 0.881-0.884) and lower PR-AUC (0.664 vs. 0.688-0.693).

Once every model is scored at its own validation-selected threshold, the three non-linear models cluster tightly (PR-AUC 0.688-0.693, FPR 0.112-0.114) and all three beat Logistic Regression on recall by roughly 5 points for about 0.3 points of extra false-positive rate. The earlier version of this document reported a much starker tradeoff; that was a measurement bug, corrected below.

Neither tree-based model is "better" in every dimension — this is the honest, imperfect tradeoff a held-out comparison is supposed to surface, not a result to celebrate or a reason to switch the live-scoring model without a fuller interpretability/latency/monitoring evaluation (out of scope for this prototype).

## Probability calibration (added 2026-09-04)

PR-AUC measures whether the model *ranks* merchant-weeks correctly. Calibration measures whether the number it states is *true*: among rows scored 0.30, do roughly 30% actually deteriorate? These are different properties, and a model can be excellent at one and bad at the other. This matters concretely here because three downstream things treat the output as a real probability — the rupee cost model multiplies it against losses, the operating threshold is a probability cut point, and a reviewer reading "0.86" reasonably infers an 86% chance.

**Metrics** (`ml/calibration.py`): **Brier score** (a proper scoring rule — minimised only by stating the true probability, so it is the primary number), **expected calibration error** (count-weighted mean gap between stated and observed rate), and **maximum calibration error** (worst single bin). MCE is reported because ECE's count weighting hides errors in exactly the sparse high-probability bins that generate escalations.

**Methodology disclosure.** Calibrators are fit on the **validation** split, which this project already uses for threshold selection — validation does double duty. The clean alternative is a fourth dedicated calibration split, which would shrink every other split on an already-synthetic dataset. The held-out test set is scored only and never fit on, so the test metrics below are honest; but a calibrator fit on genuinely fresh data would be a stricter test. This is a real limitation, not a technicality.

### Held-out test results

| Method | Variant | Brier ↓ | ECE ↓ | MCE ↓ | Mean predicted | Observed |
|---|---|---|---|---|---|---|
| Logistic Regression | raw | 0.0762 | 0.0398 | 0.3526 | 0.1429 | 0.1449 |
| | platt | 0.0763 | 0.0392 | 0.3544 | | |
| | **isotonic** | **0.0686** | **0.0040** | 0.2720 | | |
| Random Forest | raw | 0.0971 | 0.1432 | 0.3560 | **0.2881** | 0.1449 |
| | platt | 0.0652 | 0.0082 | 0.2960 | | |
| | **isotonic** | **0.0652** | 0.0072 | 0.2625 | | |
| Gradient Boosting | **raw** | **0.0646** | 0.0105 | 0.4029 | 0.1467 | 0.1449 |
| | platt | 0.0646 | 0.0085 | 0.3366 | | |
| | isotonic | 0.0648 | 0.0079 | 0.2918 | | |
| Trajectory Transformer | raw | 0.0652 | 0.0128 | **0.0814** | 0.1354 | 0.1449 |
| | platt | 0.0649 | 0.0119 | 0.2067 | | |
| | **isotonic** | 0.0649 | 0.0078 | 0.2472 | | |

### What this found

**1. Random Forest was badly miscalibrated, and this retroactively explains the Day 1 threshold bug.** Raw Random Forest predicts a mean probability of **0.288 against a true base rate of 0.145** — it states roughly double the real risk, with an ECE of 0.143, an order of magnitude worse than every other model. This is the direct cause of the threshold-scoring bug documented below: Random Forest selected a threshold of 0.55 while the other models selected 0.10–0.15 *because its probability scale is inflated*, and applying Logistic Regression's 0.10 to that inflated scale is what produced the spurious 0.335 false-positive rate. Calibration turns an unexplained oddity ("why is RF's threshold 0.55?") into a measured property. Isotonic calibration essentially fixes it (Brier 0.097 → 0.065, ECE 0.143 → 0.007).

**2. Gradient Boosting needs no calibration.** Its raw Brier score (0.0646) is the best of any model or variant, and calibration does not improve it. `HistGradientBoostingClassifier` optimises log loss directly, so well-calibrated output is expected rather than lucky.

**3. Post-hoc calibration made the Trajectory Transformer's worst case worse.** Its raw MCE of **0.0814 is by far the best worst-bin calibration of any model** — three to five times better than everything else. Both calibrators *degrade* it (platt 0.2067, isotonic 0.2472) while improving its ECE. This is a genuine tradeoff, not a bug: the calibrators minimise average error and are free to worsen an individual bin to do it. It is also the clearest evidence in this evaluation that "apply isotonic regression" is not automatically an improvement — reported because the opposite conclusion would have been the convenient one.

This was checked for the obvious artifact: a low MCE can be manufactured by having most bins empty. It is not the case here. Every populated bin of the Transformer's raw reliability curve holds between 144 and 10,814 rows, and the largest single gap across all nine populated bins is 0.081. The good worst-bin behaviour is real, not a sparse-bin illusion.

**4. Logistic Regression's mean is nearly perfect, but it badly understates risk in the bin next to its own operating threshold.** Mean predicted 0.1429 against observed 0.1449 looks excellent, and ECE 0.0398 is respectable — but MCE is 0.3526, and the reliability curve locates it precisely:

| Bin | Rows | Mean predicted | Observed rate | Gap |
|---|---|---|---|---|
| [0.1, 0.2) | 769 | 0.126 | **0.478** | **0.353** |
| [0.2, 0.3) | 58 | 0.272 | 0.466 | 0.194 |
| [0.9, 1.0) | 153 | 0.912 | 0.810 | 0.102 |

Among the 769 merchant-weeks Logistic Regression scored between 0.10 and 0.20, **nearly half (47.8%) actually deteriorated** — the model said roughly 13%. It understates risk by a factor of ~3.8 in that band, and that band sits **immediately above its live operating threshold of 0.10**. These cases are still flagged (0.126 > 0.10), so the review queue is not missing them — but any downstream use that reads the number as a magnitude, such as a reviewer prioritising by score or a merchant-facing figure, is materially wrong in exactly the region where marginal decisions get made. (The rupee cost model is *not* such a use — see "Consequences" below.) Isotonic calibration reduces LR's MCE from 0.3526 to 0.2720 and its ECE from 0.0398 to 0.0040.

At the other end, the [0.9, 1.0) bin is *over*confident: 0.912 stated against 0.810 observed. The model is not uniformly biased in one direction, which is why a single scalar correction (Platt) barely helps it — Platt actually makes both its Brier and MCE marginally worse — while the free-form isotonic map does.

This is the case MCE was added to expose, and it would have been invisible from PR-AUC, precision, recall, or ECE alone.

### Consequences, stated honestly

- **Live scoring is unchanged.** No calibrator is applied to the live path. Logistic Regression remains the sole live-scoring model with its uncalibrated probabilities and its existing F2 threshold, because changing the live scoring scale would invalidate every number in this document and the review workflow built on it. Calibration is currently a *measurement*, not a deployment.
- **The Day 2 cost analysis is *not* affected by any of this, and that is worth stating explicitly rather than leaving ambiguous.** `ml/cost_model.py` is **counts-based**: it derives TP/FP/FN/TN from the held-out labels and multiplies those integer counts by fixed rupee constants. The predicted probability enters only as a cut point in `probs >= threshold` — never as a magnitude in a `P(risk) × loss` product. A false negative costs ₹18,000 whether the model stated 0.13 or 0.48, so the 0.10–0.20 bin error above cannot corrupt the rupee arithmetic. Verified as well as argued: because monotone calibration preserves ordering it preserves the set of achievable partitions, and on a fine threshold grid raw and isotonic-calibrated Logistic Regression minimise at ₹18,642,800 and ₹18,641,200 — a 0.009% discretisation residue.
- **A reviewer should not read the displayed probability as a literal frequency.** Given LR's MCE of 0.35, "0.86" is a severity signal and a ranking, not a calibrated 86%. The UI presents risk tiers rather than raw probabilities, which is the right call for this reason.

## Rupee cost analysis (added 2026-09-04)

**Every rupee figure in this section is an assumption, not a measured cost.** They live in `rules/cost_model.yaml` (matching the convention that policy numbers stay out of Python) and are chosen to make the cost *structure* visible on synthetic data. They are not Razorpay's costs, not any real provider's costs, and not benchmarked against published figures. No public per-merchant chargeback-cost dataset exists to calibrate against. Anyone applying this to real money must substitute their own measured costs — which is why the sensitivity table below matters more than the headline number.

### The cost matrix

| Outcome | Charged | Rationale |
|---|---|---|
| True positive | `review_cost` + `(1 − recovery_rate) × missed_loss` | Catching a case costs analyst time and is not perfectly effective |
| False positive | `review_cost` + `false_positive_friction` | Analyst time **plus** the goodwill cost of troubling a healthy merchant |
| False negative | `missed_loss` | The full unmitigated loss |
| True negative | ₹0 | — |

Assumed values: `review_cost` ₹250, `missed_loss` ₹18,000, `false_positive_friction` ₹150, `recovery_rate` 0.55.

Two design choices worth stating. **`false_positive_friction` is deliberately non-zero** — a model that treats unnecessary merchant friction as free will always over-flag, and this project's whole premise is that a wrongly-flagged merchant bears a real cost. **`recovery_rate` is 0.55, not 1.0** — setting it to 1.0 would model human review as perfectly effective and would flatter every number below; the system only ever recommends evidence requests and monitoring, never enforcement, so it cannot avoid every loss.

### Held-out test results (14,400 rows)

Both baselines are reported because each alone is misleading: "review nobody" (₹37.6M) is what a team without this system does, and "review everybody" (₹22.4M) is the trivial alternative that needs no model at all.

| Method | Operating point | Expected cost | Reviews | ₹ saved per 1,000 reviews | Beats review-everything? |
|---|---|---|---|---|---|
| Rules-only | fixed | ₹29,272,250 | 3,059 | ₹2,711,262 | **No** |
| Logistic Regression | F2 (0.10) | ₹21,441,250 | 3,079 | ₹5,237,009 | Yes |
| Combined policy | fixed | ₹21,118,000 | 4,507 | ₹3,649,434 | Yes |
| Random Forest | F2 (0.55) | ₹20,379,300 | 3,213 | ₹5,349,113 | Yes |
| Trajectory Transformer | F2 (0.15) | ₹20,348,300 | 3,236 | ₹5,320,674 | Yes |
| Gradient Boosting | F2 (0.15) | ₹20,332,600 | 3,247 | ₹5,307,484 | Yes |
| Gradient Boosting | cost-optimal (0.05) | **₹19,741,400** | 4,382 | ₹4,067,686 | Yes |

**The uncomfortable finding, stated plainly: the rules-only baseline does not beat reviewing everybody.** At ₹29.3M it costs *more* than the ₹22.4M review-everything policy. Under these assumed costs, a team running the rules engine alone would be better off reviewing every merchant-week — the rules add negative value against that baseline. Every ML method does beat it.

**The second uncomfortable finding: the margin over review-everything is thin.** The best method costs ₹19.7M against review-everything's ₹22.4M — about **12%**. Most of the value under this cost ratio comes from *reviewing a lot*, not from the model's discrimination. That is a direct consequence of the assumed 72:1 miss-to-review cost ratio: when a miss costs 72 reviews, being trigger-happy is nearly free. A team with a lower ratio would see the model's discrimination matter much more.

**Note on "₹ saved per 1,000 reviews".** It is savings versus doing nothing, divided by reviews generated. It therefore *falls* as a method flags more, since later reviews catch progressively fewer real cases. Logistic Regression at its F2 threshold has the best figure among ML methods (₹5.24M) precisely because it flags fewest. Read it as review efficiency, not total value — the two rank methods differently, and neither alone is the right answer.

### Cost-optimal threshold, and why it is reported alongside F2 rather than instead of it

The cost-optimal threshold minimizes expected rupees on **validation data only**, using the same discipline as the F2 threshold; the held-out test set is only ever scored at an already-fixed point. Under the assumed costs it is 0.05 for Logistic Regression, Gradient Boosting and the Trajectory Transformer, and 0.25 for Random Forest.

**0.05 is the bottom of the search grid, and the report flags it as such** (`cost_optimal_threshold_at_grid_boundary: true`). That threshold should be read as "the grid could not locate the optimum," not as a located optimum.

**Corrected 2026-09-04 — the flag was right but my stated reason for it was wrong.** This document originally said the true minimum "may lie below 0.05." A fine-grained sweep (0.0005 steps) shows the actual optimum for Logistic Regression is at **0.0605** — *between* the grid's 0.05 and 0.10 points, not below 0.05. The 0.05-step grid is simply too coarse to express it. The direction of the error was the opposite of what was claimed; the conclusion that the grid-boundary result should not be trusted was correct.

This is exactly why the F2 threshold remains the live operating point and the cost-optimal one is reported beside it. The cost-optimal threshold is a direct function of a guessed cost ratio; adopting it as *the* answer would disguise an assumption as a result.

### Sensitivity: how much does the answer depend on the guess?

Cost-optimal threshold for Logistic Regression as the miss-to-review cost ratio varies (validation-selected):

| missed_loss / review_cost | Implied missed loss | Cost-optimal threshold | Reviews |
|---|---|---|---|
| 10 | ₹2,500 | 0.10 | 2,768 |
| 25 | ₹6,250 | 0.10 | 2,768 |
| 50 | ₹12,500 | 0.05 | 5,935 |
| **72 (assumed)** | **₹18,000** | **0.05** | **5,935** |
| 100 | ₹25,000 | 0.05 | 5,935 |
| 200 | ₹50,000 | 0.05 | 5,935 |
| 500 | ₹125,000 | 0.05 | 5,935 |

The direction is the sanity check — as a miss gets more expensive, flagging more aggressively becomes optimal, and it does. But the table also shows the analysis is **saturated above a ratio of ~50**: every ratio from 50 to 500 pins the threshold to the grid floor. The assumed 72 sits inside that saturated region, which means the specific choice of ₹18,000 is doing less work than it appears — anything above ~₹12,500 gives the same answer. Below ratio 50 the recommendation genuinely moves. A reader whose real ratio is under 50 should use this table rather than our headline number.

### Limitations of this cost model

- Every rupee value is a guess (see above). The *structure* is the contribution here; the magnitudes are not.
- Costs are treated as constant per merchant-week. In reality a miss on a high-volume merchant costs far more than on a small one, and a volume-weighted loss model would be more faithful.
- `recovery_rate` is a single scalar. Realistically, recovery depends on how early the case was caught.
- The analysis runs on synthetic data whose label prevalence (14.5%) was chosen by the generator, and total cost scales directly with prevalence.
- No discounting, no analyst-capacity constraint, and no queueing: the model assumes every flagged case actually gets reviewed. A real team with 3,247 flagged cases and capacity for 500 would see very different economics.

## Trajectory Transformer (comparison baseline, added 2026-09-04)

**Status: comparison-only. Not used for live scoring.** Like Random Forest and Gradient Boosting, this model is evaluated on the same split, with the same preprocessing and the same threshold-selection method, and is never loaded by `ml/case_packet.py`, `app/services/case_service.py`, or `combined_policy()`. **Logistic Regression remains the sole live-scoring model, unchanged.** Added per the Day 1 upgrade-sprint plan.

**Motivation.** Every other model in this project sees one merchant-week row at a time. A merchant whose refund rate crept up for six straight weeks and one with an identical current refund rate that spiked once are indistinguishable to them, except insofar as the hand-built `*_change_30d` features capture it. This model reads the trailing 8 weeks directly.

**Architecture** (`ml/trajectory_transformer.py`, **18,113 parameters**): a 2-layer custom pre-norm self-attention encoder — written explicitly rather than assembled from `nn.TransformerEncoder`, which does not cleanly expose per-position attention weights — with `d_model=32`, `n_heads=4`, `dim_feedforward=64`, `dropout=0.1`, and a learned positional embedding over the 8-week window. Deliberately tiny: 900 merchants of synthetic data cannot support a large model, and pretending otherwise would produce an impressive-sounding architecture with a worse honest result.

Each week's input token is the **exact same** numeric + one-hot vector produced by `ml/model_utils.py::build_preprocessing_pipeline()` that Logistic Regression, Random Forest, and Gradient Boosting consume, fit on the training split only — linearly projected to `d_model`. There is no second, divergent feature path.

**Pooling and where the attention numbers come from — stated precisely, because it is easy to overclaim here.** The encoder output is combined by **masked mean-pooling** over non-padding positions, not by a learned attention-pooling query. The per-week weights reported below are therefore read from the **last encoder layer's self-attention**, averaged over query positions and over heads, with padding positions zeroed. That is a reasonable proxy for "which weeks did the model's representation draw on," but it is *not* a direct readout of each week's contribution to the final score — no such readout exists in the current architecture. Replacing mean-pooling with a learned attention-pooling query would make the weights a direct contribution readout; that is a known, unimplemented improvement, not a claim about the present model.

**New dependency: PyTorch.** `torch>=2.0,<3.0` is added to `requirements.txt` solely for this model. This is a deliberate, disclosed deviation from the otherwise scikit-learn-only ML stack, scoped to the comparison-only path — every live-scoring path remains scikit-learn. It is recorded here rather than introduced silently.

### Aggregate result

**Third on PR-AUC, statistically indistinguishable from the tree models.** PR-AUC 0.688, against Gradient Boosting's 0.693 and Random Forest's 0.689 — a spread of 0.005 across all three on a 14,400-row test set, which is well inside run-to-run noise. Precision, recall, F1, F2, and FPR are likewise near-identical (see the results table above).

**Why, mechanistically: this was the predicted outcome, not a surprise.** The synthetic generator's true process is a **low-order Markov chain** — each merchant's latent state transitions based on the current state alone, via a fixed transition matrix, with no longer-range dependency. There is consequently very little long-range temporal structure in the data for self-attention to discover. Whatever short-horizon trend signal exists is already available to a tree model through the current week's `*_change_30d` features. A sequence model cannot extract structure that the data-generating process never put there, and it would have been a red flag if it had appeared to.

The honest conclusion is that on **this** data the Transformer's differentiator is the explanation type, not accuracy. Whether it would win on real merchant data with genuine long-range dependencies is untested and unknowable from this prototype.

### Scenario-slice result (a separate finding, not a contradiction)

Third overall on the aggregate number, **joint-best on the two hardest scenarios the evaluation was specifically designed to stress**:

- **Early hidden risk** (the false-negative trap): **0 false negatives**, tied with Gradient Boosting, against Random Forest's 5 and Logistic Regression's 81. Slice size: **850 rows, of which 448 are actual positives** — "0 FN" means all 448 were caught, not that the slice was trivially small.
- **Seasonal returns** (the false-positive trap): **14 false positives**, against Gradient Boosting's 28, Logistic Regression's 62, and the rules engine's 450. Slice size: **1,128 rows, of which 108 are actual positives**. Random Forest is better still here at 4.

These are complementary to, not in tension with, the aggregate ranking. Aggregate PR-AUC averages over a test set that is 70% "stable" rows; the slice numbers say where a method's errors actually land. A model that is third by 0.005 PR-AUC but makes half as many errors on the trap scenarios is a materially different operational proposition — which is the entire reason `compute_scenario_difficulty()` exists.

### Corrected measurement errors

Two bugs were found and fixed on 2026-09-04. Both are documented here with before/after numbers rather than quietly corrected.

**1. Threshold-scoring bug (affected Random Forest and Gradient Boosting).**

`evaluate_split()` applied Logistic Regression's selected threshold of 0.10 to *every* method's probabilities, even though Random Forest had selected 0.55 and Gradient Boosting 0.15 on validation data. This was not label leakage — no test-set information reached threshold selection, and the thresholds themselves were correctly chosen on validation only — but it is an invalid comparison: applying one model's operating point to another model's probability distribution measures the wrong thing entirely.

The distortion was severe for Random Forest, whose probabilities are distributed very differently from a logistic model's. The calibration analysis above quantifies exactly why: Random Forest's raw mean predicted probability is 0.288 against a true base rate of 0.145, so its entire probability scale is inflated by roughly 2x — which is both why it selected a 0.55 threshold and why forcing 0.10 onto it was so destructive.

| | Published (wrong threshold) | Corrected (own threshold) |
|---|---|---|
| Random Forest FPR | 0.335 | **0.112** |
| Random Forest precision | 0.323 | **0.572** |
| Random Forest recall | 0.944 | **0.881** |
| Random Forest seasonal FP | 1,020 | **4** |
| Gradient Boosting FPR | 0.177 | **0.114** |
| Gradient Boosting precision | 0.470 | **0.568** |

PR-AUC was unaffected (0.689 / 0.693 in both), since it is computed over the whole probability ranking and does not depend on a threshold. **Fix:** `evaluate_split()` takes a `thresholds_by_method` mapping and records `threshold_used` for every method in the report. Corrected in `README.md`, this document, and the standalone project-overview doc.

**2. Split-boundary truncation (affected the Trajectory Transformer only).**

`build_sequences()` only saw rows inside whichever DataFrame it was handed. At evaluation time that was the test split alone, so each merchant's **first 7 test weeks (~7% of test rows)** had their trailing window truncated at the split boundary and zero-padded — even though the real prior weeks existed in the validation split immediately before.

This was **not label leakage**: only feature vectors are read, no labels crossed any boundary, and the effect biased recall *downward* — the conservative direction. But it was still an invalid measurement, because it measured "does trajectory help when 7% of rows have their history artificially cut" rather than the intended question. The truncation specifically handicapped the one model built to use that history.

**Fix:** `TrajectoryModel.predict()` and `.attention_by_week()` take an optional `history_df` supplying read-only trailing context across the split boundary. Features only, never labels; transformed through the train-fit pipeline, never re-fit; only weeks **strictly before** the target week are used, enforced by de-duplication that lets a target row win over any context copy of the same week — so a row can never serve as its own history. Zero-padding remains the fallback only when a merchant's real history is genuinely shorter than the window. Directly unit-tested in `tests/test_trajectory_transformer.py`, including an adversarial case where context rows at or after the target week are poisoned with a sentinel value and asserted never to appear.

**Metric delta: negligible.**

| | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| PR-AUC | 0.6837 | 0.6879 | +0.0042 |
| Precision | 0.5689 | 0.5692 | +0.0003 |
| Recall | 0.8821 | 0.8826 | +0.0005 |
| FPR | 0.1133 | 0.1132 | −0.0001 |

Exactly **1 row of 14,400 flipped FP→TN and 1 flipped FN→TP**. All five other methods were byte-identical before and after, confirming the change touched only the Transformer path.

**Why the fix mattered anyway.** The metric delta is inside noise, but the fix is the difference between a genuine finding and an artifact. Pre-fix, **3 of the top-6 attention examples were degenerate `[0, 0, 0, 0, 0, 0, 0, 1.0]`** — all attention on the current week, all seven prior positions padded. That pattern was *forced by missing data*, not learned. All three sat on 2024-09-09, the first week of the test split, and all three reported the identical probability 0.8659 because the model was effectively scoring one identical-shaped input.

Post-fix, **all 6 top examples have real 8-week context**, and the current-week peak that appears in 5 of 6 cases (0.33–0.48 of total attention) is now a genuine learned preference — consistent with the low-order-Markov argument above, and therefore evidence *for* that argument rather than an artifact masquerading as one. Stated plainly: the number has the same shape before and after the fix, but a completely different epistemic status. Before, it was an artifact of truncation; after, it is a finding.

### Worked examples

Two traces from `ml/artifacts/trajectory_attention_examples.json`, both at the model's 0.15 threshold. Columns are weeks before the scored week (−7 = seven weeks earlier, 0 = the week being scored).

**Example A — true positive.** `merchant_demo_0600`, week 2024-10-28, predicted probability **0.8591**, actual label **1**.

| Week offset | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 |
|---|---|---|---|---|---|---|---|---|
| Attention | 0.115 | 0.086 | 0.086 | 0.125 | 0.062 | 0.050 | 0.064 | **0.414** |

**Example B — false positive, included deliberately.** `merchant_demo_0593`, week 2024-11-25, predicted probability **0.8615**, actual label **0**.

| Week offset | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 |
|---|---|---|---|---|---|---|---|---|
| Attention | 0.118 | 0.052 | 0.092 | 0.084 | 0.085 | 0.053 | 0.067 | **0.450** |

Example B is included on purpose, and the examples artifact is deliberately **not** filtered to correct predictions only. Two of the six highest-confidence flags in the current artifact are false positives at p ≈ 0.86 — and their attention traces are essentially indistinguishable from the true positive's. That is exactly the failure mode the human-review workflow exists to catch: a high-confidence score that is still wrong, with an explanation that looks just as plausible as a correct one. It is the concrete argument for routing to a reviewer rather than acting automatically, and curating it out of the demo would remove the strongest evidence for the system's own core design decision.

### Caveat: attention is not explanation

**Attention weights are a heuristic proxy for which inputs the model weighted most, not a rigorous causal explanation of its decision.** A trace showing 0.45 on the current week does not establish that the current week *caused* the score, that the model would have scored differently had that week differed, or that a different attention distribution would have produced a different output. This is a well-established general critique of attention-as-explanation in the interpretability literature (see Jain & Wallace, "Attention is not Explanation," NAACL 2019, and the counterpoint in Wiegreffe & Pinter, "Attention is not not Explanation," EMNLP 2019) — the point is not that attention is uninformative, but that it does not carry the guarantees an explanation is normally expected to.

Compounding this specifically here: as described under "Pooling" above, these weights are read from the last encoder layer's self-attention rather than from a pooling mechanism that directly determines the output, so they are one step further removed from the score than the term "explanation" implies.

This caveat applies to both worked examples above and to **any** dashboard or demo use of `trajectory_attention_examples.json`. The artifact is labelled `"role": "explanation_illustration_only"` for this reason. Live case explanations shown to reviewers and merchants come from Logistic Regression's interpretable feature contributions, not from attention weights.

## Scenario difficulty (per latent state, held-out test)

Added 2026-08-28 to make explicit which demonstration scenarios are actually hard to catch, rather than averaging that difficulty away into one aggregate number. `ml/model_utils.py::compute_scenario_difficulty()` reports, per latent state, the label's positive rate within that state and each method's recall within that state (i.e., of the actual positives in that scenario, what fraction did the method flag).

| Scenario | Rows | Positive rate | Rules-only | Logistic Regression | Random Forest | Gradient Boosting | Trajectory Transformer |
|---|---|---|---|---|---|---|---|
| Stable | 10,057 | 1.4% | 0.072 | 0.000 | 0.000 | 0.000 | 0.000 |
| Seasonal returns (false-positive design) | 1,128 | 9.6% | 0.435 | 0.046 | 0.028 | 0.037 | 0.019 |
| Operational failure | 1,288 | 40.0% | 0.416 | 0.973 | 1.000 | 1.000 | 1.000 |
| High-risk behavior | 1,077 | 81.4% | 0.616 | 0.974 | 1.000 | 1.000 | 1.000 |
| Early hidden risk (false-negative design) | 850 | 52.7% | 0.304 | 0.819 | 0.989 | 1.000 | 1.000 |

**What this shows honestly:** the rare positive-label rows inside the nominally "Stable" state (1.4% positive rate — these are the tail-probability draws the generator's `label_probability` floor still allows) are uncatchable by every ML method (recall 0.000) and only incidentally caught by the rules engine — a genuine model limitation on the hardest, rarest scenario, not a bug being hidden. The same is nearly true of the positives buried inside the seasonal-returns trap: all four ML methods score below 0.05 recall there, meaningfully *worse* than the rules engine's 0.435, because they have learned to treat the seasonal pattern as benign. That is the intended tradeoff of the false-positive design — the seasonal FP counts in the results table are low precisely because these methods suppress that state — but it is a real cost, and it is the clearest case in this evaluation where the rules-only fallback catches something the models do not.

Conversely, Logistic Regression reaches 0.819 recall on the deliberately-hard "early hidden risk" scenario, and Gradient Boosting and the Trajectory Transformer both reach 1.000 on it.

## Baseline comparison

1. **Rules-only policy** — transparent, but weakest of the five: precision 0.31 / recall 0.45 / PR-AUC 0.29 on held-out test. Misses more than half of actual high-loss merchant-weeks and still produces 450 seasonal-sale false positives on the test set.
2. **Logistic Regression alone** — the model used for live scoring: precision 0.56 / recall 0.83 / PR-AUC 0.66, and the lowest false-positive rate (0.110) of any ML method. Direct evidence the model catches the "looks normal but isn't" pattern the rules engine is structurally unable to see, at the lowest over-flagging cost.
3. **Random Forest / Gradient Boosting / Trajectory Transformer (comparison-only)** — see "Comparison baselines" and "Trajectory Transformer" below for the full honest read: all three cluster at PR-AUC 0.688-0.693 and recall 0.881-0.884, beating Logistic Regression's 0.664 / 0.828 at a nearly identical false-positive rate (0.112-0.114 vs. 0.110). Gradient Boosting leads on PR-AUC by a margin too small to be meaningful on a 14,400-row test set.
4. **Logistic Regression plus rules combined policy** — trades precision for higher recall (0.87) by also catching what rules see independently. Produces more seasonal-sale false positives (475) than the model alone (62) — an honest trade-off: the more conservative, catch-more-for-human-review policy, appropriate when false negatives are considered costlier than an unnecessary but reversible review.

**The Logistic Regression model demonstrably adds value beyond the rules-only baseline** on every primary metric, on synthetic data, in this prototype — and the tree-based comparison shows that added model complexity buys a modest, honest improvement on some metrics (Gradient Boosting's PR-AUC/recall) at a real cost on others (FPR), not a free win.

**Model-selection note:** the selected threshold of 0.10 was chosen on validation data using an F2-oriented objective, which intentionally values recall more than precision. This threshold is appropriate only for a decision-support and review-routing prototype. **It must not be interpreted as an automatic-enforcement threshold.** The combined policy has higher recall but also a higher false-positive rate than the model-only result. It is therefore presented as a conservative case-routing policy, not as a universally superior fraud classifier.

**Method-selection guidance:**

| Use case | Recommended method | Reason |
|---|---|---|
| Live case scoring (this prototype) | Logistic Regression | Interpretable, lowest FPR among ML methods, per CLAUDE.md's model-complexity preference |
| Low-friction early-warning dashboard | Logistic Regression | Better precision and lower FPR |
| Conservative risk-operations queue | Combined policy | Higher recall; reviewer handles more cases |
| Research/comparison only | Random Forest, Gradient Boosting | Not used for live scoring; shown to make the complexity/accuracy tradeoff honest, not hidden |
| No ML available | Rules-only | Fully transparent fallback, but lower synthetic test performance |

**`class_weight` decision:** not used (`class_weight=None`). The validation-selected threshold (0.10) already satisfied both the precision ≥ 0.30 and false-positive-rate ≤ 0.20 constraints without it, so `class_weight='balanced'` was not justified per the Milestone 3 requirement to use it only when needed and to document the decision either way.

**Score-tier vs. recommendation UI rule (preserved from Milestone 2):** a high numeric risk signal does not automatically mean manual review. `ml/model_utils.py::combined_policy()` always returns `risk_signal_intensity` (Low/Medium/High), `recommendation`, and a `policy_explanation` string together, so that whenever the two diverge — e.g. high model probability with only an evidence-gap/support-stress signal and no confirmed chargeback spike — the explanation is present, not just the raw recommendation.

## Explainability

- Show rule triggers directly, with concrete before/after trend values (e.g. "chargeback rate increased from 0.4% to 2.2%").
- Show top model feature contributions.
- Convert factors into safe plain-language explanations.
- Do not expose raw coefficients or exact thresholds in merchant-facing UI.

### Explainability-quality policy: diagnostic-only features

**Resolved (added post-Milestone-4 review).** A model feature may be shown as a natural-language positive or negative risk factor only when its observed direction is plausible, stable across validation/test or sensitivity checks, and does not contradict the documented risk-policy interpretation. Otherwise it is diagnostic-only and excluded from ranked explanations. The model may continue to use it, but its direction must be documented as unvalidated.

**Diagnostic-only feature list (`ml/explain_cases.py::DIAGNOSTIC_ONLY_FEATURES`): `support_ticket_rate`.** This feature's fitted Logistic Regression coefficient is negative — higher support-ticket rate slightly *lowers* the predicted score — which contradicts `RISK_POLICY.md`'s `SUPPORT_OPERATIONAL_STRESS` rule (rising support load is documented as a risk-increasing signal) and has not been checked for stability across a sensitivity analysis. It therefore stays in the model's feature list and continues to be fit and scored on, but it is excluded from `top_model_factors` in every analyst and merchant-safe explanation (`ml/explain_cases.py::compute_top_factors()`), and its direction is documented here as unvalidated rather than presented as a natural-language risk reason.

## Near-perfect-score investigation rule

**Resolved (approved 2026-08-22).** This is the standard procedure for the held-out evaluation, not a placeholder.

If a held-out test result has PR-AUC >= 0.98, precision >= 0.98 and recall >= 0.98, or zero false positives/false negatives on a meaningful test set, mark it **"Under investigation."** Do not use it in the submission pitch until these checks pass:

- Label leakage check
- Time leakage check
- Train/validation/test split integrity check
- Entity duplication/leakage check
- Latent-state overlap check
- Verification that held-out data includes intentional false-positive seasonal-sale cases and false-negative early-hidden-risk cases
- Rules-only versus model-plus-rules baseline comparison

Document the investigation outcome in the final evaluation report (which check(s) were run, what was found, and whether the result was confirmed genuine or attributed to a bug). A result marked "Under investigation" must not be reported as a headline metric until this documentation exists.

**Actual outcome (re-run 2026-08-28 at the v0.2.0 dataset scale): gate NOT triggered for any of the five methods**, including the two tree-based comparison baselines. None reached PR-AUC ≥ 0.98, precision and recall both ≥ 0.98, or zero false positives/negatives on the held-out test set — the best PR-AUC achieved was 0.693 (Gradient Boosting), well below the 0.98 gate. All five methods are **APPROVED**, not under investigation.

**Concrete held-out test examples (Logistic Regression at threshold 0.10, v0.2.0 dataset):**

| Type | merchant_id | week_start | refund_rate_30d | prior refund rate | chargeback_rate_30d | evidence coverage | model probability |
|---|---|---|---|---|---|---|---|
| Seasonal false positive | merchant_demo_0003 | 2024-11-04 | 6.05% | 5.67% | 0.41% | 86.3% | 0.107 |
| Seasonal false positive | merchant_demo_0011 | 2024-12-16 | 7.15% | 1.22% | 0.37% | 85.2% | 0.106 |
| Seasonal false positive | merchant_demo_0021 | 2024-12-16 | 6.88% | 1.36% | 0.43% | 83.6% | 0.111 |
| Early-hidden-risk false negative | merchant_demo_0009 | 2024-10-28 | 2.28% | 5.41% | 0.48% | 82.9% | 0.095 |
| Early-hidden-risk false negative | merchant_demo_0013 | 2024-09-30 | 2.30% | 6.87% | 0.53% | 80.0% | 0.090 |

The false positives are exactly the intended pattern: elevated refund rate with low chargeback rate and healthy evidence coverage, correctly resembling a legitimate return spike, yet the merchant did enter a simulated high-loss state within 30 days — a genuinely hard case, not a modeling error. The false negatives show the model's probability (0.08) sitting just below the 0.10 threshold for merchants whose observed signals are mild — exactly the "early hidden risk" pattern the latent-state design intends to be only partially catchable.

## Limitations

- Synthetic labels and distributions cannot establish real-world loss/fraud performance.
- Model cannot access real gateway signals, bank intelligence, device intelligence, or genuine dispute evidence.
- Potential fairness and bias concerns cannot be fully evaluated with synthetic data.
- Model performance may still be inflated if the latent-state generator's feature distributions are insufficiently overlapping between states — this must be checked, not assumed away.

## Monitoring plan

For the MVP, generate basic reports for:

- Missingness by feature
- Merchant-category mix
- Risk-score distribution
- Drift proxy between training and newer synthetic data
- Reviewer outcome distribution
- False-positive and false-negative rate over time

## Human oversight

High-risk outputs only create review recommendations. Human reviewers must record the outcome (clear case, mark false positive, request evidence, mark operational issue, escalate to compliance, or mark inconclusive) before any downstream action occurs outside the MVP boundary.
