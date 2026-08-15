# BullionVault Multi-Horizon Research — Stage 6

## Purpose

Stage 6 is a newly versioned candidate-family expansion after the valid negative Stage-3 result.

This document and `src/metal_predictor/multi_horizon/stage6_preregistration.py` are committed **before** any Stage-6 candidate is fitted or scored. The locked historical-test blocks are still sealed.

## What remains unchanged

- Immutable BullionVault Chart Export HLC source files and Stage-1 SHA provenance.
- The 12-feature causal HLC feature graph from Stage 2.
- Exact next-source-bar log-return target.
- Expanding four-fold walk-forward plan with a one-bar purge.
- Independent selection per horizon; no cross-horizon pooling.
- No fill, interpolation, resampling, fabricated Open, or calendar/timezone-derived features.
- 1d remains `DATA_PENDING`; live 1d forward bars are not converted into fabricated historical training data.
- The formal future holdout, the frozen 52-feature live model, and Shadow62 are untouched.

## New preregistered candidate families

1. `train_median_return_v2` — train-only median return, a robust absolute-error-optimal learned constant.
2. `ridge_alpha_1000_v2` — strongly shrunk linear model intended to stay close to the random-walk baseline.
3. `random_forest_shallow_v2` — fixed shallow forest with large leaves and no tuning search.
4. `hist_gradient_boosting_shallow_v2` — fixed low-capacity boosting under absolute-error loss.

Every hyperparameter is frozen in the preregistration. No grid search, Bayesian search, Optuna, manual result-driven edits, or horizon-specific tuning is allowed.

## Development gate

The random-walk zero-return baseline remains the benchmark. A candidate must pass every predeclared check:

- lower development OOF MAE than random walk;
- lower MAE in at least 3 of 4 folds;
- paired block-bootstrap 95% MAE-improvement CI lower bound greater than zero;
- bootstrap probability of candidate MAE improvement at least 0.99;
- directional-accuracy delta no worse than -0.02.

The bootstrap uses 5,000 resamples, block length 4, and seed `20260815`.

If no candidate passes, Stage 6 retains `random_walk_zero_return` and **does not** create or deploy a candidate forecast model.

## Historical confirmation firewall

Stage-6 development is forbidden from reading, predicting, scoring, reporting, or tuning against the existing locked historical-test blocks.

Only if a candidate passes the development gate may that single selected winner be authorized for a separate one-shot historical confirmation. The confirmation must occur in a later commit/run. Failure cannot be followed by post-test tuning.

## Production firewall

This preregistration does not change the production forecast routes, does not create model artifacts, and does not promote a model.

Guardrails remain:

`edge_status=NOT_PROVEN`, `research_only=true`, no BUY/SELL, no execution, no automatic promotion, no mutation of the frozen 52-feature graph, and no mutation of Shadow62.
