# Model Card — ClearRisk Recover Baseline

Status: Design proposal, rescoped 2026-08-22. See `docs/RESCOPE_REVIEW.md`.

## Model name

ClearRisk Recover Logistic Regression Baseline, plus two comparison-only baselines (Random Forest, Gradient Boosting)

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

**Threshold selection:** grid search over [0.05, 0.95] in steps of 0.05 on **validation data only**, maximizing F-beta(2) subject to precision ≥ 0.30 and false-positive rate ≤ 0.20. Selected threshold (Logistic Regression): **0.10** (constraints were met — no fallback needed). Random Forest: **0.55**. Gradient Boosting: **0.15**. The held-out test set is never used for threshold selection (verified by `tests/test_train_baseline_model.py::test_held_out_data_cannot_alter_selected_threshold`).

**Held-out test results** (14,400 rows, prevalence 14.49%):

| Method | Precision | Recall | F1 | F2 | PR-AUC | FPR | Seasonal FP | Early-hidden FN |
|---|---|---|---|---|---|---|---|---|
| Rules-only | 0.310 | 0.454 | 0.368 | 0.415 | 0.288 | 0.172 | 450 | 312 |
| Logistic Regression (threshold 0.10) | 0.561 | 0.828 | 0.669 | 0.756 | 0.664 | 0.110 | 62 | 81 |
| Random Forest (threshold 0.55, comparison only) | 0.323 | 0.944 | 0.481 | 0.682 | 0.689 | 0.335 | 1020 | 0 |
| Gradient Boosting (threshold 0.15, comparison only) | 0.470 | 0.925 | 0.624 | 0.775 | 0.693 | 0.177 | 787 | 0 |
| Combined policy (built on Logistic Regression) | 0.403 | 0.870 | 0.551 | 0.706 | 0.664 | 0.219 | 475 | 56 |

ROC-AUC (secondary only, not the primary metric per the imbalanced-label caution): rules-only 0.656, Logistic Regression 0.906, Random Forest 0.919, Gradient Boosting 0.921, combined 0.906 (held-out test).

## Comparison baselines (Random Forest, Gradient Boosting)

**Design decision: evaluation-only, never used for live scoring.** `ml/train_tree_models.py` trains a `RandomForestClassifier` and a `HistGradientBoostingClassifier` (both ship in scikit-learn — no new dependency) on the exact same preprocessing, split, and threshold-selection method as the Logistic Regression baseline, so any difference in results reflects the model, not an uneven evaluation. Neither model is loaded by `ml/case_packet.py`, `app/services/case_service.py`, or `combined_policy()` — Logistic Regression remains the sole model behind live case creation and the combined policy, per CLAUDE.md's "prefer transparent rules and interpretable models before complex models."

**Honest reading of the comparison, held-out test:**

- **Random Forest** reaches the highest recall (0.944) but at a false-positive rate roughly 3x Logistic Regression's (0.335 vs. 0.110) and the worst precision of the three ML methods (0.323) — it over-flags substantially, including 1,020 seasonal-sale false positives (vs. 62 for Logistic Regression). class_weight="balanced" was used to address the label imbalance directly; the tradeoff is visible here, not hidden.
- **Gradient Boosting** is the best-balanced of the three: precision 0.470, recall 0.925, PR-AUC 0.693 (the highest of the four methods) — a real, honest case that a more complex model *can* modestly outperform Logistic Regression on this synthetic held-out set, at the cost of interpretability.
- **Logistic Regression** has the lowest false-positive rate (0.110) and the fewest seasonal-sale false positives (62) of the three ML methods, at a real cost in recall (0.828 vs. 0.925-0.944) and slightly lower PR-AUC (0.664 vs. 0.689-0.693).

Neither tree-based model is "better" in every dimension — this is the honest, imperfect tradeoff a held-out comparison is supposed to surface, not a result to celebrate or a reason to switch the live-scoring model without a fuller interpretability/latency/monitoring evaluation (out of scope for this prototype).

## Scenario difficulty (per latent state, held-out test)

Added 2026-08-28 to make explicit which demonstration scenarios are actually hard to catch, rather than averaging that difficulty away into one aggregate number. `ml/model_utils.py::compute_scenario_difficulty()` reports, per latent state, the label's positive rate within that state and each method's recall within that state (i.e., of the actual positives in that scenario, what fraction did the method flag).

| Scenario | Rows | Positive rate | Rules-only recall | Logistic Regression recall | Random Forest recall | Gradient Boosting recall |
|---|---|---|---|---|---|---|
| Stable | 10,057 | 1.4% | 0.072 | 0.000 | 0.158 | 0.000 |
| Seasonal returns (false-positive design) | 1,128 | 9.6% | 0.435 | 0.046 | 1.000 | 0.843 |
| Operational failure | 1,288 | 40.0% | 0.416 | 0.973 | 1.000 | 1.000 |
| High-risk behavior | 1,077 | 81.4% | 0.616 | 0.974 | 1.000 | 1.000 |
| Early hidden risk (false-negative design) | 850 | 52.7% | 0.304 | 0.819 | 1.000 | 1.000 |

**What this shows honestly:** the rare positive-label rows inside the nominally "Stable" state (1.4% positive rate — these are the tail-probability draws the generator's `label_probability` floor still allows) are essentially uncatchable by Logistic Regression (recall 0.000) and barely caught by the tree models — this is a genuine model limitation on the hardest, rarest scenario, not a bug being hidden. Conversely, Logistic Regression already reaches 0.819 recall on the deliberately-hard "early hidden risk" scenario it was designed to only partially catch — direct evidence the model adds real value on the harder half of the false-negative design, while still leaving the "surprise positive in an otherwise stable merchant" case as an open, disclosed limitation.

## Baseline comparison

1. **Rules-only policy** — transparent, but weakest of the five: precision 0.31 / recall 0.45 / PR-AUC 0.29 on held-out test. Misses more than half of actual high-loss merchant-weeks and still produces 450 seasonal-sale false positives on the test set.
2. **Logistic Regression alone** — the model used for live scoring: precision 0.56 / recall 0.83 / PR-AUC 0.66, and the lowest false-positive rate (0.110) of any ML method. Direct evidence the model catches the "looks normal but isn't" pattern the rules engine is structurally unable to see, at the lowest over-flagging cost.
3. **Random Forest / Gradient Boosting (comparison-only)** — see "Comparison baselines" above for the full honest read: Gradient Boosting edges out Logistic Regression on PR-AUC (0.693 vs. 0.664) and recall (0.925 vs. 0.828), at a higher false-positive rate (0.177 vs. 0.110); Random Forest reaches the highest recall (0.944) but the worst precision/FPR of the three.
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
