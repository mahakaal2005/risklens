 # Open-Source Foundations and Research References

 Status: Design proposal, rescoped 2026-08-22. The project narrowed from an earlier
 four-scenario concept to one flagship scenario (merchant refund/chargeback loss risk).
 The reference-repo research from that earlier pass (verified via GitHub/search, Aug
 2026) still applies — it was concept/license research, not scenario-specific — and is
 kept below unchanged except where a note calls out the narrower scope. See
 `docs/RESCOPE_REVIEW.md` for the rescope decision.

## Purpose

This project will not copy or fork an entire fraud-detection product.

We will use selected open-source projects as:

- Architecture references
- Threat-model references
- Synthetic-data references
- Explainability/monitoring libraries
- Optional future-production references

Every dependency or borrowed idea must be checked for license compatibility, maintenance, security, and fit before use.

---

## Decision summary

### Use directly in MVP

- scikit-learn — interpretable baseline model
- SHAP — optional model explanation
- Evidently — optional data quality / drift report
- FastAPI, Streamlit, SQLite, Pandas — local MVP application stack

### Use as reference only

- Jube
- MITRE Fight Fraud Framework (F3)
- Santander `gen-fraud-graph`
- Databricks Telco Fraud Prevention
- IBM TabFormer

### Consider later, not in MVP

- Feast
- MLflow
- Graph database / graph ML
- GNNs
- Kafka / streaming infrastructure
- Kubernetes / microservices

---

## Repository comparison

| Project | Credibility / maintainer | License | How this project uses it | Do not use it for | Decision |
|---|---|---|---|---|---|
| [Jube](https://github.com/jube-home/aml-fraud-transaction-monitoring) | Open-source AML and fraud transaction-monitoring project | AGPL-3.0-or-later | Study alert → case → investigator → resolution flow; rule management; analyst workbench; case/audit concepts | Copying code into this project; using it as a drop-in payment-gateway solution; assuming it supports Indian payment rails | Reference only |
| [MITRE Fight Fraud Framework (F3)](https://github.com/center-for-threat-informed-defense/fight-fraud-framework) | MITRE Center for Threat-Informed Defense | Apache-2.0 | Naming/documentation-taxonomy reference for the one flagship scenario (merchant refund/chargeback loss) | ML training, application UI, payment gateway integration, actual decision engine, or a claim of F3 coverage | Reference only |
| [Santander AI: gen-fraud-graph](https://github.com/SantanderAI/gen-fraud-graph) | Banco Santander AI Lab | Apache-2.0 | Borrow the *principle* of deliberately injecting known-answer synthetic patterns rather than pure random noise, adapted to a flat, non-graph merchant-week generator | A full dashboard, direct UPI dataset, graph database/GNN, or first MVP model | Reference only for MVP |
| [IBM TabFormer](https://github.com/IBM/TabFormer) | IBM research project | Verify repository license before any use | Later research into transaction-history / sequential tabular modelling | Initial model, baseline scoring, or merchant explanation; it is too complex and less transparent for MVP | Future research only |
| [Databricks Telco Fraud Prevention](https://github.com/databricks-industry-solutions/fraud_prevention_in_telco) | Databricks industry solution | Verify repository license before any code use | Study analyst-workbench layout, risk-engine tables, investigation workflow; the raw→enriched-features→rules→score→workbench separation | Copying telecom-specific features, claiming payment-gateway coverage, or introducing Databricks dependency in MVP | Reference only |
| [Feast](https://github.com/feast-dev/feast) | Open-source feature-store project | Apache-2.0 | Later use if production system needs consistent offline/online features | Hackathon/local MVP; do not introduce extra infrastructure yet | Deferred |
| [Evidently](https://github.com/evidentlyai/evidently) | Open-source ML evaluation/monitoring project | Apache-2.0 | Optional report for missing values, feature drift proxy, score drift, and data-quality checks | Loss/fraud model training or risk decisioning | Optional MVP dependency |
| [MLflow](https://github.com/mlflow/mlflow) | Open-source ML lifecycle platform | Apache-2.0 | Later experiment tracking, model metadata, artifact registry, and reproducibility | Required MVP dependency; avoid extra setup until core flow works | Deferred |
| [SHAP](https://github.com/shap/shap) | Widely used model-explainability library | MIT | Optional local feature attribution for analyst view | Complete merchant explanation; it must be combined with rules and safe plain language | Optional MVP dependency |

---

## Licensing rules

1. Do not copy Jube source code into this project without reviewing AGPL-3.0-or-later obligations.
2. Treat every external repository as reference material unless a specific file and license have been reviewed.
3. Preserve required copyright notices and license notices for any copied code.
4. Prefer writing a small original implementation for the MVP.
5. Before adding a Python dependency, verify:
   - license
   - maintained release history
   - security advisories
   - compatibility with the local Python version
   - whether it is actually necessary

---

## What we borrow conceptually

### From Jube

Conceptual workflow only:

```text
Event → Alert → Case → Analyst review → Evidence → Resolution → Audit history
```

MVP mapping:

```text
Synthetic merchant-week record
→ Risk assessment
→ Review case
→ Analyst action
→ Merchant appeal/evidence
→ Resolution
→ Application audit event
```

### From MITRE F3

Use the framework's disciplined naming approach for the one flagship scenario:

- Refund or chargeback loss trend (the sole Phase 1 scenario)

Earlier drafts of this document also listed "unusual transaction behaviour," "possible account takeover," and "merchant-profile mismatch" as F3-inspired scenario names — these are no longer part of the Phase 1 scope (see `docs/RESCOPE_REVIEW.md`) and are Future work only, not implemented scenarios.

Do not pretend that this MVP has full MITRE F3 coverage.

### From Santander gen-fraud-graph

Use the principle that fraud/loss patterns can be simulated safely by deliberately injecting known-answer examples rather than relying purely on random noise:

- A small set of deterministic, labeled example merchant-weeks per outcome type (true positive, false positive, false negative, true negative) for testability and demonstration.
- Repeat/persistent merchant behaviour over time (state persistence across merchant-weeks), not one-shot independent sampling.

Explicitly not borrowed: shared customer/device tokens, connected merchant/customer graphs, or mule-ring patterns — these solve a different problem (multi-party collusion) than the one flagship scenario (single-merchant refund/chargeback trend), and would require a graph database this MVP does not use.

For MVP version 1, use a simple flat CSV/SQLite dataset, not a graph database.

### From Databricks fraud-prevention solution

Study the separation of:

```text
Raw event
→ Enriched features
→ Rules / risk signals
→ Risk score
→ Analyst workbench
→ Investigation outcome
```

Translate telecom concepts into merchant refund/chargeback-risk concepts. Do not copy telecom-specific labels such as SIM swap, call forwarding, impossible travel, or cell-hop detection.

### From SHAP

Use SHAP only to help the analyst understand model output.

The final explanation must combine:

```text
1. Triggered business rules, with concrete before/after trend values
2. Top model feature contributions
3. Safe plain-language explanation
4. Recommended next reviewer action
```

---

## Why the MVP does not use complex tools

| Tool / approach | Why not in version 1 |
|---|---|
| GNN / graph neural network | Requires graph data, complex evaluation, and reduces explainability; the one flagship scenario is a single-merchant trend, not a multi-party pattern |
| TabFormer | Research-heavy and unnecessary before a transparent baseline works |
| Feast | Feature store overhead is unnecessary for local batch data |
| MLflow | Useful, but not required until experiments become difficult to track |
| Kafka | No real-time event stream exists in this local prototype |
| Kubernetes | No deployment-scale need |
| Microservices | Adds complexity and slows learning |
| Real gateway integration | No approved access, real data, or safe testing environment |

---

## Mandatory quality check before adopting anything

For every repository, package, or code sample considered:

1. Confirm maintainer/organization.
2. Check license.
3. Check recent maintenance activity.
4. Read the README and installation path.
5. Inspect tests and CI, if present.
6. Identify whether it uses synthetic or real data.
7. Identify security/privacy risks.
8. Record exactly what will be reused:
   - concept only
   - documentation pattern
   - dependency
   - small code fragment with license compliance
9. Record what will not be reused.
10. Do not adopt it if it expands the MVP scope without clear value.

---

## Final MVP stack decision

```text
Application:
- Python
- FastAPI
- Streamlit
- SQLite

Data and ML:
- Pandas
- NumPy
- scikit-learn
- PyYAML
- Joblib

Optional:
- SHAP for analyst explanations
- Evidently for a basic monitoring/data-quality report

Reference materials:
- Jube
- MITRE F3
- Santander gen-fraud-graph
- Databricks Telco Fraud Prevention
- IBM TabFormer
```

The MVP will be built primarily with original, small, understandable code. External repositories guide architecture and research; they are not proof that the MVP is production-ready.
