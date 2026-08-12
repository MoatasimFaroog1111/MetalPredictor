# Platinum + Palladium cross-asset research

This component is a **candidate research path only**. It does not change the frozen live Silver models, the frozen 52-feature graph, the future holdout, or any BUY/SELL/execution setting.

## Scientific question

Does information from the Platinum (`XPT.CMD/USD`) and Palladium (`XPD.CMD/USD`) markets improve the next exact-hour Silver return forecast beyond the frozen Silver-only feature set?

The feature family is pre-registered as `precious-metals-cross-asset-v1` before any comparison result is observed.

## Source policy

Historical auxiliary candles are requested from Dukascopy Trading Tools historical-prices API with:

- `timeFrame=1hour`
- `dayStartTime=UTC`
- `offerSide=B` (Bid)
- exact provider timestamps only
- no forward-fill
- no backward-fill
- no interpolation
- no nearest/as-of timestamp matching
- conflicting duplicate hours fail closed

Dukascopy lists Platinum as `XPT.CMD/USD` and Palladium as `XPD.CMD/USD`, with one CFD corresponding to one troy ounce. Source USD/troy-ounce OHLC is normalized to USD/kg using the project's existing `TROY_OZ_PER_KG` constant.

The API credential is read only from `DUKASCOPY_API_KEY`. Never commit the key.

A successful source parse is **not** model-readiness. Sparse/truncated provider coverage must pass the pre-registered development coverage gate below before any estimator is fitted.

## Pre-registered candidate features

For each of XPT and XPD:

- exact-current availability
- candle range and candle body
- log metal/Silver price ratio
- exact 1h, 6h and 24h returns + availability flags
- relative return versus Silver at 1h, 6h and 24h
- metal/Silver ratio changes at 6h and 24h
- 24h realized volatility
- Silver/metal rolling correlation at 24h and 72h

Joint metal-complex features:

- both-metals-current availability
- log XPT/XPD ratio
- 1h XPT/XPD ratio change
- mean XPT/XPD 1h return
- XPT/XPD return dispersion
- XPT/XPD directional breadth
- XPT minus XPD 1h return spread

Total candidate features: **43**.

Changing these windows or definitions requires a new feature-version identifier and a new experiment; v1 is immutable.

## Pre-registered provider-coverage gate

The coverage rule is fixed before observing model-comparison results. The enhanced development dataset must satisfy all of the following:

- each metal exact-current coverage over full development: at least 50%
- each metal exact-current coverage in every purged training fold: at least 40%
- each metal exact-current coverage in every validation fold: at least 60%
- joint XPT+XPD exact-current coverage in every validation fold: at least 50%
- joint exact-current rows in every purged training fold: at least 2,000
- joint exact-current rows in every validation fold: at least 500

If any gate fails, the experiment stops before fitting the baseline or enhanced model. Missing provider observations remain missing; the gate prevents an experiment that is effectively dominated by imputation from being labeled valid.

## Evaluation protocol

`compare_precious_metals_features.py` compares:

- A: frozen Silver-only feature set
- B: Silver + pre-registered XPT/XPD v1 candidates

using the same frozen `ridge_alpha_100` estimator recipe, paired Purged Walk-Forward folds, and paired block bootstrap. The loader opens original Train + Validation only. The old Test and future holdout are forbidden for feature selection.

A candidate result does not alter the production model automatically. Promotion requires the existing predeclared evidence rule and a separate explicit model-freeze decision.

## Run order

1. Configure `DUKASCOPY_API_KEY` as a repository secret.
2. Run `Cross-Asset Platinum Palladium Research` manually.
3. Inspect source acquisition artifacts.
4. Require the pre-registered provider-coverage gate to pass.
5. Inspect the paired development-only comparison report.
6. Do not score or inspect the future holdout as part of this experiment.
