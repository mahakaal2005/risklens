 # Research Notes — RiskLens

 Status: Design proposal, rescoped 2026-08-22 to a single flagship scenario (merchant
 refund/chargeback loss risk). See `docs/RESCOPE_REVIEW.md`.

 ## Research question

 How can an explainable, merchant-week-level refund/chargeback loss-risk review workflow
 reduce harm from false-positive merchant flags while still catching genuine loss risk?

 ## Verified facts

 - Payment aggregators in India are regulated by RBI and must conduct customer/merchan
 - Payment gateways may use holds, reserves, KYC/compliance reviews, and other control
 - Fraud/risk systems must balance fraud detection with false-positive harm: incorrect

 ## Company-reported/public-provider material

 - Razorpay public guidance describes account freezes/settlement holds in relation to
 - Public review-site complaints can indicate recurring user pain, but individual comp

 ## Design proposal

 This MVP does not attempt to replicate a payment gateway's hidden fraud engine. It de

 - One bounded, transparent risk scenario: merchant refund/chargeback loss risk.
 - Rules plus interpretable scoring, with the label generated from a hidden latent-state
   simulation rather than directly from the rules' own thresholds.
 - Safe explanation (with concrete before/after trend values) and evidence request.
 - Human review and merchant appeal.
 - Auditability and false-positive/false-negative measurement.
 ## Assumptions

 - A future integration partner could supply permissioned transaction, refund, chargeb
 - High-impact actions would remain in an authorized payment-gateway workflow, not ins
 - Merchant-facing explanations can safely reveal broad reason categories and evidence

 ## Unknown / not publicly verified

 - Exact internal risk thresholds and permanent-termination rules of any specific paym
 - Exact availability of merchant-level risk-score APIs, analyst workflows, or policy
 - Whether a given public complaint reflects fraud, operational error, policy violatio

 ## What we can safely claim

 RiskLens is a synthetic-data prototype for explainable merchant refund/chargeback
 loss-risk review. It

 ## What still needs verification

 - Specific integration options with a payment aggregator.
 - Legal/privacy/security requirements for any real deployment.
 - Model effectiveness using representative, permissioned, real-world data.
 - Operational thresholds, reviewer SLAs, and policy requirements for a production par

