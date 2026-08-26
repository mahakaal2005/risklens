# Model Card — ClearRisk Recover Baseline

Status: Design proposal, rescoped 2026-08-22. See `docs/RESCOPE_REVIEW.md`.

## Model name

ClearRisk Recover Logistic Regression Baseline

## Version

`0.1.0` (implemented, Milestone 3 — trained and evaluated; see `docs/MILESTONE_3_MODEL_EVALUATION.md` for the full run report)

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

**Actual data characteristics (seed=42, `python3 ml/generate_synthetic_data.py`):**

- Date range: 2025-01-06 to 2025-12-29 (52 weeks)
- 220 merchants, 11,440 merchant-weeks total
- Label distribution: 14.28% positive overall (`label_high_loss_next_30d`); per-state rates in the table below
- Merchant-category mix: apparel 40, digital_services 42, electronics 37, food_delivery 35, grocery 33, travel 33 (by merchant)
- Latent-state distribution: stable_merchant 8012 (70.0%), operational_fulfilment_failure 1061 (9.3%), seasonal_sale_legitimate_returns 897 (7.8%), high_risk_merchant_behaviour 864 (7.6%), early_hidden_risk 606 (5.3%)
- Generator version: `0.1.0`, seed: `42`

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

**Split (actual, seed=42):** strict chronological split by unique `week_start` — train: earliest 36 weeks (2025-01-06 to 2025-09-08, 7,920 rows); validation: next 8 weeks (2025-09-15 to 2025-11-03, 1,760 rows); held-out test: latest 8 weeks (2025-11-10 to 2025-12-29, 1,760 rows). No row is shuffled; no week appears in more than one split (verified by `tests/test_split_data.py`).

**Threshold selection:** grid search over [0.05, 0.95] in steps of 0.05 on **validation data only**, maximizing F-beta(2) subject to precision ≥ 0.30 and false-positive rate ≤ 0.20. Selected threshold: **0.10** (constraints were met — no fallback needed). The held-out test set is never used for threshold selection (verified by `tests/test_train_baseline_model.py::test_held_out_data_cannot_alter_selected_threshold`).

**Validation results** (1,760 rows, prevalence 15.45%):

| Method | Precision | Recall | F1 | F2 | PR-AUC | FPR |
|---|---|---|---|---|---|---|
| Rules-only | 0.328 | 0.456 | 0.382 | 0.423 | 0.317 | 0.171 |
| Logistic Regression (threshold 0.10) | 0.575 | 0.890 | 0.698 | 0.802 | 0.739 | 0.120 |
| Combined policy | 0.423 | 0.912 | 0.577 | 0.740 | 0.739 | 0.228 |

**Held-out test results** (1,760 rows, prevalence 12.84%):

| Method | Precision | Recall | F1 | F2 | PR-AUC | FPR | Seasonal FP | Early-hidden FN |
|---|---|---|---|---|---|---|---|---|
| Rules-only | 0.269 | 0.425 | 0.329 | 0.381 | 0.246 | 0.170 | 53 | 35 |
| Logistic Regression (threshold 0.10) | 0.529 | 0.863 | 0.656 | 0.766 | 0.653 | 0.113 | 30 | 4 |
| Combined policy | 0.375 | 0.881 | 0.526 | 0.693 | 0.653 | 0.216 | 71 | 2 |

ROC-AUC (secondary only, not the primary metric per the imbalanced-label caution): rules-only 0.640, Logistic Regression 0.901, combined 0.901 (held-out test).

## Baseline comparison

1. **Rules-only policy** — transparent, but weakest of the three: precision 0.27 / recall 0.42 / PR-AUC 0.25 on held-out test. Misses more than half of actual high-loss merchant-weeks (recall 0.42) and still produces 261 false positives on the test set (53 of them the designed seasonal-sale case).
2. **Logistic Regression alone** — the strongest single method: precision 0.53 / recall 0.86 / PR-AUC 0.65. Nearly triples the rules-only PR-AUC and cuts early-hidden-risk false negatives from 35 (rules-only) to 4 — direct evidence the model catches the "looks normal but isn't" pattern the rules engine is structurally unable to see.
3. **Logistic Regression plus rules combined policy** — trades precision for even higher recall (0.88) by also catching what rules see independently (e.g. any `CHARGEBACK_RATE_SPIKE`/`COMBINED_LOSS_SIGNAL` trigger routes to `MANUAL_REVIEW_REQUIRED` regardless of the model's probability). Produces more seasonal-sale false positives (71) than the model alone (30) — an honest trade-off, not a flaw: it is the more conservative, catch-more-for-human-review policy, appropriate when false negatives (missed real loss) are considered costlier than false positives (an unnecessary but reversible review).

**The Logistic Regression model demonstrably adds value beyond the rules-only baseline** on every primary metric, on synthetic data, in this prototype.

**Model-selection note:** the selected threshold of 0.10 was chosen on validation data using an F2-oriented objective, which intentionally values recall more than precision. This threshold is appropriate only for a decision-support and review-routing prototype. **It must not be interpreted as an automatic-enforcement threshold.** The combined policy has higher recall but also a higher false-positive rate than the model-only result. It is therefore presented as a conservative case-routing policy, not as a universally superior fraud classifier.

**Method-selection guidance:**

| Use case | Recommended method | Reason |
|---|---|---|
| Low-friction early-warning dashboard | Logistic Regression | Better precision and lower FPR |
| Conservative risk-operations queue | Combined policy | Higher recall; reviewer handles more cases |
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

**Milestone 3 actual outcome: gate NOT triggered.** None of rules-only, Logistic Regression, or the combined policy reached PR-AUC ≥ 0.98, precision and recall both ≥ 0.98, or zero false positives/negatives on the held-out test set (see the Evaluation section above for exact figures — the best PR-AUC achieved was 0.653, well below the 0.98 gate). All three methods are **APPROVED**, not under investigation. See `docs/MILESTONE_3_MODEL_EVALUATION.md` Section 11 for the full gate check output.

**Concrete held-out test examples (Logistic Regression at threshold 0.10):**

| Type | merchant_id | week_start | refund_rate_30d | prior refund rate | chargeback_rate_30d | evidence coverage | model probability |
|---|---|---|---|---|---|---|---|
| Seasonal false positive | merchant_demo_0018 | 2025-11-17 | 6.16% | 5.95% | 0.30% | 87.7% | 0.119 |
| Seasonal false positive | merchant_demo_0023 | 2025-11-24 | 6.00% | 1.83% | 0.29% | 84.0% | 0.128 |
| Seasonal false positive | merchant_demo_0026 | 2025-12-01 | 5.45% | 6.10% | 0.33% | 83.3% | 0.144 |
| Early-hidden-risk false negative | merchant_demo_0007 | 2025-12-08 | 2.61% | 2.61% | 0.55% | 84.6% | 0.081 |
| Early-hidden-risk false negative | merchant_demo_0032 | 2025-12-15 | 2.73% | 2.52% | 0.74% | 84.6% | 0.084 |

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
