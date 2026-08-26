# Rescope Review — ClearRisk Recover

Status: Design proposal. No application code created or changed as part of this
review — per instruction, this document (and the other documentation updates
listed in "Documentation Updates Required") is the deliverable, and application
code (`app/`, `ml/*.py`, `rules/risk_rules.yaml`, `tests/`) waits for explicit
approval.

This document records the rescope from the original four-scenario "AI Risk
Manager" to the single-flagship "ClearRisk Recover" (merchant refund and
chargeback loss risk), why each change was made, and the concrete design this
narrower scope implies.

---

## 1. What was reviewed

All root documentation (`CLAUDE.md`, `PRD.md`, `ARCHITECTURE.md`,
`DATA_DICTIONARY.md`, `RISK_POLICY.md`, `MODEL_CARD.md`, `SECURITY.md`,
`RESEARCH.md`, `README.md`), `docs/OPEN_SOURCE_FOUNDATIONS.md.txt`, and the
code artifacts already on disk (`rules/risk_rules.yaml`,
`ml/data_validation.py`, `ml/__init__.py`) were reviewed against the new scope
decision.

---

## 2. Old requirements removed, changed, or moved to future work

### Removed as standalone Phase 1 scenarios (moved to Future work / optional context only)

| Old requirement | Disposition |
|---|---|
| R1 — Unusual transaction amount detector | **Removed as a scored scenario.** Transaction-level amount anomalies are no longer part of the hero use case. May be mentioned once, under Future work, as a possible Phase 2+ signal. |
| R2 — New device + customer-dispute detector | **Removed.** This was an account-takeover-flavored scenario, a different loss class from refund/chargeback loss. Moved to Future work only. |
| R4 — Merchant profile mismatch detector | **Removed as a standalone scenario.** A category-mismatch signal may still exist as one *supporting feature* inside the merchant-week record (not a separate case type or rule family with its own recommendation path). |
| Generic transaction-level fraud scoring | **Removed.** No per-transaction risk_score/risk_tier/case in Phase 1. |
| Payment-level account-takeover detection | **Removed.** Not part of the refund/chargeback loss class. |
| Gateway connector | **Stays out of scope**, and additionally now removed from the pitch/demo narrative entirely (previously it was "out of scope" but still discussed in ARCHITECTURE.md's component table; now it should not appear in the hero pitch at all). |
| Live webhook integration | **Removed from Phase 1 narrative.** Previously implied as a Phase 2 item ("gateway exports/webhook-shaped mock events"); now explicitly deferred, not mentioned in the core demo. |
| Phase 3 production plans in the main pitch/demo | **Removed from pitch/demo.** Phase 3 still exists as a documented future phase (see `docs/PROJECT_SKILLS.md`-adjacent memory / `ARCHITECTURE.md`), but must not appear in `docs/SUBMISSION_PITCH.md` or `docs/DEMO_SCRIPT.md`. |

### Changed (not removed, but redefined)

| Old shape | New shape | Why |
|---|---|---|
| Primary entity: `transaction` (per-event), with a separate `merchant` dataset joined ad hoc for R3/R4 | Primary scored entity: **`merchant_week`** (one row per merchant per week) | The flagship loss class (refund/chargeback trend) is inherently a merchant-level, time-windowed phenomenon, not a per-event one. Per-transaction records may still exist as an internal generation detail that rolls up into merchant-week aggregates, but they are not the scored/cased entity. |
| `R3_REFUND_SPIKE` / `R3_CHARGEBACK_SPIKE` rules, evaluated at arbitrary transaction-scoring time against merchant's `_current_30d` vs `_prior_30d` fields | Same underlying idea, now the **only** rule family, evaluated once per merchant-week using that week's `_30d` vs `_previous_30d` fields | Directly matches the new Core Data Model; no change in spirit, only in cadence and exclusivity. |
| Risk score/tier existed on both `Transaction` and `Merchant` conceptually (per `ARCHITECTURE.md` Section 5 core entities: `Transaction`, `RiskAssessment` implied per-transaction) | Single `risk_score` / `risk_tier` per **merchant-week** | Removes the dual-entity scoring ambiguity; one scored thing, one case type. |
| `case.entity_type` enum: `transaction/merchant` | `case.entity_type`: effectively always `merchant_week` (merchant_id + week_start) | Matches the single flagship entity. |
| Label: `label_risk_outcome` (`clean/risk/false_positive`) on transactions; no explicit forward-looking merchant label existed | One explicit label: **`label_high_loss_next_30d`** (binary, forward 30-day horizon), defined via latent-state simulation, documented in `DATA_DICTIONARY.md` and `MODEL_CARD.md` | Old label was retrospective/ambiguous and transaction-scoped; new label is a precise, forward-looking, merchant-scoped prediction target — matching "Primary model task" in the new spec. |
| Explanation text: generic bullet list of triggered rule names/categories | Explanation must state **concrete before/after trend values** (e.g., "chargeback rate increased from 0.4% to 2.2%") | Matches "Core Risk Explanations" requirement; makes the explanation demonstrably grounded in the actual data, not just a rule-name label. |
| Recommendations included `STEP_UP_VERIFICATION_RECOMMENDED` (tied to the now-removed device/dispute scenario) | Recommendation enum narrows to: `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE` | `STEP_UP_VERIFICATION_RECOMMENDED` only made sense for an account-takeover-flavored scenario (verify the customer/device), which no longer exists in Phase 1. **This is flagged as a design decision, not silently assumed** — see Open Question 1 below. |
| Reviewer outcomes: `CONFIRMED_RISK / FALSE_POSITIVE / INCONCLUSIVE / OPERATIONAL_ISSUE` | Reviewer actions: `Clear case / Mark false positive / Request evidence / Mark operational issue / Escalate to compliance / Mark inconclusive` | New spec's action list is action-oriented (verbs) rather than outcome-oriented (nouns) and adds `Request evidence` and `Escalate to compliance` as explicit reviewer actions rather than only case-status transitions. Both lists cover the same underlying states; renamed/expanded per the new spec verbatim. |

### Stale code artifacts already on disk (not touched yet — flagged only)

These were written before the rescope decision and encode the **old** four-scenario, transaction-level design. They are not deleted or edited in this pass (per "wait for approval before creating or changing application code"), but they will need a full rewrite once code work resumes:

- `rules/risk_rules.yaml` — encodes `R1_HIGH_VALUE_ANOMALY`, `R2_NEW_DEVICE_DISPUTES`, `R3_REFUND_SPIKE`, `R3_CHARGEBACK_SPIKE`, `R4_PROFILE_MISMATCH` as transaction-level rules with a `STEP_UP_VERIFICATION_RECOMMENDED` path. Needs a full rewrite to a merchant-week-only rule set (refund/chargeback rules only, category-mismatch as a feature not a rule, no device/amount rules).
- `ml/data_validation.py` — Pydantic schemas (`MerchantRecord`, `TransactionRecord`) shaped for the old transaction+merchant dataset. Needs a new `MerchantWeekRecord` schema matching the Core Data Model fields below.
- `ml/__init__.py` — empty, no change needed.

---

## 3. Revised acceptance criteria

### FR-1: Synthetic data (revised)

- The system shall generate synthetic **merchant-week** records only as the scored/cased entity. Per-transaction records may exist internally as a generation detail but are not separately validated, scored, or cased.
- Each merchant-week record includes: `merchant_id`, `week_start`, `merchant_category`, `merchant_age_days`, `transaction_count_30d`, `transaction_volume_30d`, `transaction_volume_previous_30d`, `transaction_volume_change_30d`, `refund_count_30d`, `refund_rate_30d`, `refund_rate_previous_30d`, `refund_rate_change_30d`, `chargeback_count_30d`, `chargeback_rate_30d`, `chargeback_rate_previous_30d`, `chargeback_rate_change_30d`, `top_dispute_reason_category`, `delivery_evidence_coverage`, `support_ticket_rate`, `average_support_resolution_time_hours`, `previous_review_outcome`, `label_high_loss_next_30d`.
- Records are generated from 5 latent merchant states (Section 4) with probabilistic, overlapping, noisy observed features.
- **Acceptance test:** `label_high_loss_next_30d` must NOT be recoverable as a deterministic function of the same fields/thresholds the rules engine checks (verified by confirming the rules-only baseline's precision/recall on held-out data is measurably imperfect — see Open Question 2).

### FR-2: Risk scenario (revised, singular)

- One scenario family only: **merchant refund/chargeback loss risk**, forward-looking 30-day horizon.
- **Acceptance:** the held-out test set demonstrably contains at least one true positive, one false positive (seasonal/legitimate high-return case), one false negative (early hidden-risk case), and one true negative.

### FR-3: Rules engine (revised)

- Rules operate on merchant-week aggregate fields only: refund-rate trend, chargeback-rate trend, delivery-evidence coverage, support-ticket-rate trend, previous review outcome.
- Same structural requirements as before: configurable YAML, unit-tested, each triggered rule returns `rule_id`, `severity`, `reason_category`, `safe_explanation`.
- No rule operates on a single transaction, a device token, or a customer dispute count — those fields no longer exist in the scored entity.

### FR-4: ML risk scoring (revised)

- Logistic Regression predicts `label_high_loss_next_30d` probability per merchant-week.
- Implementation order (per "Model and Evaluation"): (1) rules-only baseline, (2) Logistic Regression baseline, (3) rules + Logistic Regression combined policy — each reported separately before combining.
- Strict time-based split: earliest weeks → train, middle weeks → validation/threshold selection, latest weeks → held-out test. Thresholds are frozen before touching the held-out set.

### FR-5: Decision policy (revised)

- Allowed recommendations narrow to: `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`.
- No automatic freeze, hold, ban, termination, or payment rejection — recommendation only, matching the "Recommended Actions" section verbatim.

### FR-6: Explainability (revised)

- Explanation text must include concrete before/after values for at least the two trend fields that drove the flag (e.g., refund rate and chargeback rate, or delivery-evidence coverage and support resolution time), not just rule names.
- Merchant-facing view omits exact thresholds, raw model coefficients, and any fraud-evasion-sensitive logic (unchanged principle from the original scope).

### FR-7: Analyst review (revised)

- Reviewer view must show: risk score/tier, main risk reasons, rule triggers, refund/chargeback trend (chart or table), volume trend, delivery-evidence coverage, support-ticket rate and resolution-time trend, previous review outcomes, suggested evidence checklist, audit timeline.
- Reviewer actions: Clear case, Mark false positive, Request evidence, Mark operational issue, Escalate to compliance, Mark inconclusive. Each requires a note and writes an audit event.

### FR-8: Merchant appeal simulation (revised)

- Merchant sees safe reason category + evidence checklist, enters free-text explanation, and submits simulated evidence references (fake filenames/URLs, e.g. `invoice_demo_001.pdf`) — no real file upload.
- Submission updates case status and audit timeline (unchanged principle).

### FR-9: Audit log (largely unchanged)

- `entity_type` enum simplifies: the case/audit subject is effectively always a merchant-week; transaction, if it appears at all, is an internal generation detail, not an audited entity type in its own right.

### FR-10: Evaluation dashboard (revised)

- Must report: precision, recall, PR-AUC, false-positive rate, confusion matrix, precision/recall at the selected operating threshold, rules-only vs. rules+ML comparison, metrics by merchant state and merchant category (sample-size permitting), at least 3 concrete false-positive examples, at least 2 concrete false-negative examples.
- Must display verbatim: *"Synthetic-data metrics demonstrate the prototype workflow only. They do not prove real-world payment fraud or chargeback performance."*
- **A near-perfect score (e.g. precision/recall both > ~0.97) is treated as a bug signal, not a result to report** — the evaluation must show the investigation outcome if this occurs (see Open Question 2).

---

## 4. Synthetic latent-state generator — detailed design (proposal)

**Design proposal, not yet implemented.**

### 4.1 Latent states

Each merchant is assigned a latent state per week, drawn with **persistence** (a merchant tends to stay in its current state week-over-week, occasionally transitioning to an adjacent state) rather than resampled independently every week — this produces realistic gradual drift instead of noisy week-to-week teleporting.

| State | Approx. share of merchant-weeks | Observed-feature tendency | `label_high_loss_next_30d` probability |
|---|---|---|---|
| 1. Stable | ~70% | Flat volume, low refund (~1-2%) and chargeback (~0.2-0.5%) rates, high delivery-evidence coverage (0.85-0.98), low support-ticket rate | ~1-2% (occasional noise) |
| 2. Seasonal / legitimate high-return | ~10% | Volume spike, refund rate elevated (~4-8%) but chargeback rate stays normal, evidence coverage still good (0.8-0.95), support resolution reasonable | ~8-12% — **deliberately creates false-positive opportunities** |
| 3. Operational fulfilment failure | ~8% | Refund rate rising moderately, evidence coverage declining (0.5-0.75), support-ticket rate rising with slower resolution, mild chargeback increase | ~30-40% |
| 4. High-risk | ~7% | Both refund and chargeback rates elevated and rising sharply, evidence coverage low (0.2-0.5), poor support resolution, `previous_review_outcome` often `confirmed_risk`/`escalated` | ~70-85% |
| 5. Early hidden-risk | ~5% | Features look close to Stable or only mildly elevated this week (subtle dispute-reason shift, small support-ticket uptick); current-week signal is deliberately weak | ~45-60% — **deliberately creates false-negative opportunities** |

State transition uses a simple Markov-style matrix biased toward self-persistence (illustratively ~85% stay, ~15% move to an adjacent-severity state), so a merchant's trajectory drifts gradually (e.g. Stable → Operational failure → High-risk over several weeks) rather than jumping arbitrarily.

### 4.2 Observed feature generation

For each state, observed weekly fields are drawn from state-conditional distributions (Beta for rates, log-normal or Gaussian for volumes/counts) **with added independent noise**, and the distributions are designed to **overlap** between adjacent states (e.g. Seasonal and Operational-failure refund-rate ranges overlap) so states are not trivially separable from a single week's snapshot alone — this is what makes the eventual model's job non-trivial and the metrics meaningful.

### 4.3 Label generation

`label_high_loss_next_30d` is drawn as `Bernoulli(p)` where `p` is the state-conditional probability in the table above (optionally nudged slightly by how many consecutive weeks the merchant has been drifting toward a more severe state). It is **not** computed as `if chargeback_rate > threshold and refund_rate > threshold: label = 1` — there is no direct deterministic path from the rules engine's threshold fields to the label. This is the specific mechanism that avoids the "bad example" circularity called out in the new spec.

---

## 5. Time-based data-generation timeline and label horizon

- **History length (proposal):** 52-104 weeks (1-2 years) per merchant, across ~150-300 synthetic merchants — enough merchant-weeks for a meaningful three-way time split and enough positive labels across all 5 latent states, especially the rarer states (4 and 5, ~7% and ~5% of weeks respectively).
- **Week granularity:** `week_start` as a Monday-aligned date; deterministic via `SYNTHETIC_DATA_SEED`.
- **Label horizon:** `label_high_loss_next_30d` looks forward ~4-5 weeks from `week_start`. This means the **last ~4-5 weeks of the generated timeline cannot have a fully-observed label** (their forward window extends past the end of available history). These trailing weeks must be explicitly excluded from labeled train/validation/test data (or marked with a null/unknown label) — they must not be silently defaulted to `label = 0`, which would bias the held-out test set.
- **Three-way time-based split** (by global `week_start`, not per-merchant shuffling, to avoid leakage across the boundary):
  - Training: earliest ~60% of labeled merchant-weeks
  - Validation (threshold selection): next ~20%
  - Held-out test: latest ~20%, untouched until final evaluation, per "Do not tune thresholds using held-out test data."

---

## 6. MVP pages and user flows

### Streamlit pages (proposal)

1. **Merchant-week risk feed** — list/queue of merchant-weeks with risk tier, score, and recommendation; filterable by tier, merchant category, and (for internal/demo use) latent state. Analyst landing page.
2. **Merchant-week risk detail page** — full explanation with before/after trend values, rule triggers, refund/chargeback trend chart, evidence checklist, case action panel.
3. **Human review case page** — case queue + case detail + reviewer action form (note required for every action).
4. **Merchant appeal view** — safe reason category, evidence checklist, free-text explanation field, simulated evidence-reference submission.
5. **Audit timeline view** — chronological event log per case.
6. **Risk-manager metrics dashboard** — precision/recall/PR-AUC/confusion matrix/false-positive rate at the selected operating threshold, rules-only vs. rules+ML comparison, false-positive/false-negative example gallery, train/validation/test period display, and the synthetic-data limitation banner (verbatim text, always visible).

### User flows

**Flow A — Merchant-week review:**
`Synthetic merchant-week generated → rules + model score it → policy recommends action → medium/high risk creates case → analyst reviews reasons and trends → analyst resolves, requests evidence, or escalates → audit log records every step`

**Flow B — Merchant appeal:**
`Open case → merchant sees safe reason category and evidence checklist → merchant submits appeal text and evidence references → case returns to analyst queue → analyst records final outcome → audit timeline updates`

**Flow C — Risk-manager evaluation:**
`Load held-out metrics → inspect false-positive and false-negative examples → compare rules-only vs. rules+ML → review threshold effects → document recommended policy changes`

---

## 7. Revised folder structure

```text
CLAUDE.md
PRD.md
ARCHITECTURE.md
RESEARCH.md
DATA_DICTIONARY.md
MODEL_CARD.md
SECURITY.md
RISK_POLICY.md
README.md
requirements.txt
.env.example
app/
  main.py
  api/
  db/
  models/
  schemas/
  services/
dashboard/
  streamlit_app.py
ml/
  generate_synthetic_data.py   # merchant-week latent-state generator (rewrite pending)
  features.py                  # merchant-week feature engineering / transaction→weekly rollup
  train.py
  evaluate.py
  explain.py
  artifacts/
rules/
  risk_rules.yaml              # merchant-week refund/chargeback rules only (rewrite pending)
demo_data/
docs/
  OPEN_SOURCE_FOUNDATIONS.md
  SUBMISSION_PITCH.md
  DEMO_SCRIPT.md
  RESCOPE_REVIEW.md            # this file
tests/
```

No structural change from the original tree — the rescope changes what lives inside `generate_synthetic_data.py`, `features.py`, and `risk_rules.yaml`, not the folder layout itself.

---

## 8. Decisions (resolved 2026-08-22)

All four open questions below were resolved by explicit user approval. This section is kept as a record of what was asked and what was decided — it is no longer open.

1. **`STEP_UP_VERIFICATION_RECOMMENDED` removal — RESOLVED: remove entirely.** The Phase 1 recommendation enum is final at five values: `APPROVE`, `ALLOW_WITH_MONITORING`, `REQUEST_EVIDENCE`, `MANUAL_REVIEW_REQUIRED`, `ESCALATE_TO_COMPLIANCE`. Applied across `CLAUDE.md`, `PRD.md`, `ARCHITECTURE.md`, `RISK_POLICY.md`. No alias/fallback use of the removed value exists anywhere.
2. **"Near-perfect score" investigation procedure — RESOLVED: documented checklist adopted.** `MODEL_CARD.md` now contains the "Near-perfect-score investigation rule": any held-out result with PR-AUC >= 0.98, precision >= 0.98 and recall >= 0.98, or zero false positives/negatives is marked "Under investigation" and excluded from the submission pitch until label leakage, time leakage, split-integrity, entity-duplication, and latent-state-overlap checks pass, held-out data is confirmed to include intentional false-positive/false-negative cases, and a rules-only vs. rules+ML comparison is documented. The investigation outcome must be recorded in the final evaluation report.
3. **Merchant category mismatch — RESOLVED: dropped completely from Phase 1.** Declared-vs-observed category mismatch is not a feature, rule, scenario, or label input. `merchant_category` (a single field, no "observed" counterpart) is retained only for descriptive segmentation and optional segmented metrics (`DATA_DICTIONARY.md`, `MODEL_CARD.md`). Profile/KYC mismatch is mentioned only under Future Work.
4. **Project naming — RESOLVED: rename product/docs now; physical folder rename deferred.** Product name is `ClearRisk Recover`; full name is "ClearRisk Recover — Explainable Refund & Chargeback Risk Review"; tagline is "Detect early loss. Explain the risk. Review fairly." All documentation and product-facing text now uses this naming. The repository directory itself (currently `TracePay`) was **not** physically renamed in this pass — renaming the active working directory mid-session risked breaking session paths and open IDE tabs, so per the user's choice this is deferred to a manual `mv TracePay clearrisk-recover` done outside the session, whenever convenient. Once renamed, no further doc changes are needed — nothing in the docs hardcodes the old folder name.
