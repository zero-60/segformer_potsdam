# SXJM — LLM Training Resource Allocation Modeling

Code for the mathematical-modeling project on data quality evaluation, quality-conflict analysis, and domain-mixture modeling for large-language-model training.

## Problem 1 pipeline

The current experimental pipeline is:

1. **Round 1 — Quality-signal EDA**
   - Audit 22 quality signals.
   - Analyze scalar/list-valued features.
   - Check PCA structure and A1 vs. A2/A3 distribution shifts.

2. **Round 2 — Conflict-aware quality score**
   - Convert quality signals to a common higher-is-better direction.
   - Construct CRITIC-weighted base quality score.
   - Measure cross-group quality conflict.
   - Apply conflict correction and aggregate sample scores to domain-level quality.
   - Validate A1 sample results against the full A2/A3 extensions.

3. **Round 3 — Compositional RegMix baseline**
   - Treat the 17-domain mixture as compositional data.
   - Apply zero smoothing and ILR transformation.
   - Train mixture-to-Loss regression models on A4+A5.
   - Evaluate on A6–A11 without retraining.
   - Use Spearman correlation to study cross-scale mixture-ranking transfer.

4. **Round 4 — Quality-augmented RegMix**
   - Map quality-domain scores to the 17 RegMix domains using A16.
   - Compare M0/M1/M2 quality ablations.
   - Audit 10B/70B extrapolation using A12–A15.

5. **Round 5 — Final domain-effect analysis**
   - Re-audit A16 mapping.
   - Re-run the quality ablation after mapping correction.
   - Estimate local compositional domain effects and pairwise interactions.
   - Search for candidate mixtures only inside the empirical training support.

## Main models

Baseline:

\[
M_0:\quad \hat L=f(\operatorname{ILR}(\mathbf p))
\]

Quality-augmented:

\[
Q_{\mathrm{mix}}=\sum_j p_jQ_j,
\qquad
M_1:\quad \hat L=f(\operatorname{ILR}(\mathbf p),Q_{\mathrm{mix}})
\]

Interaction extension:

\[
M_2:\quad
\hat L=f(\operatorname{ILR}(\mathbf p),Q_{\mathrm{mix}},
\operatorname{ILR}(\mathbf p)Q_{\mathrm{mix}})
\]

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Competition attachments are expected locally under:

```text
real_attachments/
└── A_data_value/
    ├── domain_mapping_guide.csv
    └── regmix_tables/
```

Raw competition attachments and generated experiment outputs are intentionally excluded from Git by `.gitignore`.

## Running Round 5

```bash
python problem1_round5_final_domain_effects_SXJM_FIXED.py
```

Round 5 expects the Round 2 quality outputs to exist locally under `round2_outputs/`.

## Reproducibility notes

- Random seeds are fixed in the scripts.
- A4+A5 are used for RegMix fitting/model selection.
- A6–A11 are treated as external real test sets.
- A12–A15 are used only for extrapolation-robustness analysis and are not treated as real 10B/70B validation experiments.
- Mixture proportions satisfy a simplex constraint, so ILR coordinates and simplex-preserving perturbations are used instead of interpreting raw independent coefficients.
- Candidate mixtures are constrained to the empirical training support in ILR space; they should not be interpreted as proven global optima.

## Repository note

The repository contains modeling code only. Before publishing competition data or derived outputs, check the competition's data-distribution and submission rules.
