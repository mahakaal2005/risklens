# Milestone 1 Data Quality Report — ClearRisk Recover

Generated: 2026-08-23T14:01:10.671069+00:00

Status: **APPROVED FOR MILESTONE 2**

Smoke-test model is moderately useful but imperfect, as expected.

This is a Milestone 1.1 verification report, not the Milestone 7 model. The Logistic Regression below is an inspection-only smoke test with no rules engine involved.

## 1. Dataset overview

- Row count: 11440
- Merchant count: 220
- Week count: 52 (2025-01-06 to 2025-12-29)
- Target-label positive rate: 0.1428
- Latent-state distribution:
  - stable_merchant: 8012
  - operational_fulfilment_failure: 1061
  - seasonal_sale_legitimate_returns: 897
  - high_risk_merchant_behaviour: 864
  - early_hidden_risk: 606
- Merchant-category distribution (by merchant, not row):
  - digital_services: 42
  - apparel: 40
  - electronics: 37
  - food_delivery: 35
  - travel: 33
  - grocery: 33

## 2. Outcome rates by latent state

| Latent state | Count | Positive-label rate |
|---|---|---|
| stable_merchant | 8012 | 0.0132 |
| seasonal_sale_legitimate_returns | 897 | 0.0936 |
| operational_fulfilment_failure | 1061 | 0.3987 |
| high_risk_merchant_behaviour | 864 | 0.816 |
| early_hidden_risk | 606 | 0.5215 |

- seasonal_sale_legitimate_returns label=0 count: 813
- seasonal_sale_legitimate_returns label=1 count: 84
- early_hidden_risk label=0 count: 290
- early_hidden_risk label=1 count: 316

## 3. Feature overlap checks

Median (IQR) per latent state:

| Feature | stable_merchant | seasonal_sale_legitimate_returns | operational_fulfilment_failure | high_risk_merchant_behaviour | early_hidden_risk |
|---|---|---|---|---|---|
| refund_rate_30d | 0.0149 (0.0133-0.0166) | 0.0599 (0.0566-0.0635) | 0.0502 (0.0468-0.0536) | 0.0892 (0.084-0.0947) | 0.025 (0.0233-0.0267) |
| refund_rate_change_30d | -0.0005 (-0.0036-0.0019) | 0.0065 (-0.0018-0.042) | 0.0027 (-0.0031-0.0278) | 0.008 (-0.0034-0.0621) | 0.0007 (-0.0023-0.005) |
| chargeback_rate_30d | 0.003 (0.0026-0.0034) | 0.004 (0.0035-0.0045) | 0.012 (0.011-0.0129) | 0.0348 (0.0324-0.0373) | 0.006 (0.0054-0.0066) |
| chargeback_rate_change_30d | -0.0001 (-0.0008-0.0005) | 0.0002 (-0.0007-0.001) | 0.001 (-0.0007-0.0076) | 0.0037 (-0.0013-0.0286) | 0.0004 (-0.0006-0.0018) |
| delivery_evidence_coverage | 0.9203 (0.9116-0.929) | 0.8695 (0.86-0.8795) | 0.6211 (0.6032-0.6377) | 0.3503 (0.3292-0.371) | 0.7994 (0.7856-0.8124) |
| support_ticket_rate | 0.0099 (0.0085-0.0114) | 0.0201 (0.0181-0.0218) | 0.045 (0.0421-0.0482) | 0.0699 (0.0662-0.0738) | 0.0178 (0.0163-0.0195) |
| average_support_resolution_time_hours | 20.045 (15.81-25.6) | 23.9 (18.45-30.42) | 55.21 (42.37-72.5) | 69.26 (51.675-92.675) | 29.755 (23.3725-37.6) |
| transaction_volume_change_30d | -0.003 (-0.1063-0.1014) | 0.1839 (-0.013-0.4702) | -0.0167 (-0.182-0.1553) | 0.0252 (-0.2098-0.3008) | 0.0 (-0.1334-0.1295) |

- stable_merchant_vs_early_hidden_risk: 4/8 features have overlapping IQRs (['refund_rate_change_30d', 'chargeback_rate_change_30d', 'average_support_resolution_time_hours', 'transaction_volume_change_30d'])
- seasonal_sale_legitimate_returns_vs_operational_fulfilment_failure: 3/8 features have overlapping IQRs (['refund_rate_change_30d', 'chargeback_rate_change_30d', 'transaction_volume_change_30d'])

**Conclusion: both designed false-positive/false-negative pairs have at least one overlapping feature; not every feature needs to overlap (e.g. delivery_evidence_coverage is meant to differ sharply). The definitive test of whether a simple model can trivially separate all classes is the empirical baseline in Section 5, not this per-feature heuristic alone.**

## 4. Correlation / leakage screening

Candidate model features (matches MODEL_CARD.md):
- refund_rate_30d
- refund_rate_change_30d
- chargeback_rate_30d
- chargeback_rate_change_30d
- transaction_volume_change_30d
- delivery_evidence_coverage
- support_ticket_rate
- average_support_resolution_time_hours
- merchant_age_days
- merchant_category
- previous_review_outcome

- `latent_state_for_demo_only` excluded from candidate features: confirmed.
- `label_high_loss_next_30d` excluded from candidate features: confirmed (it is the prediction target).

Column-name leakage-term scan:
- No unexplained leakage-term matches found.
- Reviewed and confirmed not leakage (matches a term but is a legitimate prior-outcome feature): ['previous_review_outcome']

Excluded columns and reason:
- `merchant_id`: identifier, not a predictive feature
- `week_start`: identifier/time key, used for splitting, not fed to the model directly
- `transaction_count_30d`: raw count; the model uses the derived rate/change fields instead
- `transaction_volume_30d`: raw volume; the model uses transaction_volume_change_30d instead
- `transaction_volume_previous_30d`: raw prior volume; superseded by transaction_volume_change_30d
- `refund_count_30d`: raw count; the model uses refund_rate_30d / refund_rate_change_30d instead
- `refund_rate_previous_30d`: superseded by refund_rate_change_30d
- `chargeback_count_30d`: raw count; the model uses chargeback_rate_30d / chargeback_rate_change_30d instead
- `chargeback_rate_previous_30d`: superseded by chargeback_rate_change_30d
- `top_dispute_reason_category`: not in MODEL_CARD.md's frozen candidate feature list
- `latent_state_for_demo_only`: hidden ground-truth generation state; not observable in a real deployment, would make evaluation circular
- `label_high_loss_next_30d`: the prediction target itself

Correlation of each candidate feature with the label:

- refund_rate_30d: r = 0.5697
- refund_rate_change_30d: r = 0.2361
- chargeback_rate_30d: r = 0.6262
- chargeback_rate_change_30d: r = 0.271
- transaction_volume_change_30d: r = 0.0281
- delivery_evidence_coverage: r = -0.6584
- support_ticket_rate: r = 0.6355
- average_support_resolution_time_hours: r = 0.4966
- merchant_age_days: r = -0.0143
- merchant_category (positive rate by category): {'apparel': 0.1346, 'digital_services': 0.1516, 'electronics': 0.1414, 'food_delivery': 0.1385, 'grocery': 0.148, 'travel': 0.1428}
- previous_review_outcome (positive rate by category): {'confirmed_risk': 0.4564, 'false_positive': 0.1193, 'inconclusive': 0.2639, 'none': 0.0886, 'operational_issue': 0.3821}

## 5. Baseline sanity test (inspection-only Logistic Regression, no rules)

- Train rows: 7920, Validation rows: 1760, Test rows: 1760
- Operating threshold (selected on validation): 0.1
- PR-AUC: 0.6464
- Precision: 0.5215
- Recall: 0.8584
- False-positive rate: 0.116
- Confusion matrix: TN=1356, FP=178, FN=32, TP=194
- Rules engine was not used in this smoke test.

## 6. Near-perfect-score gate

**APPROVED FOR MILESTONE 2**

Smoke-test model is moderately useful but imperfect, as expected.

## Limitation

This report demonstrates the prototype's data-quality workflow only. It does not prove real-world chargeback-risk prediction quality; all data and outcomes are synthetic.
