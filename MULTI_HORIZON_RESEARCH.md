# BullionVault Multi-Horizon Research — Stage 2

## Scope

The isolated BullionVault multi-horizon research component now has causal per-horizon
datasets and a preregistered model-selection protocol for 4h, 12h, 2d, and 30d.
The direct 1d horizon remains `DATA_PENDING`.

Stage 2 does **not** fit candidate models, read locked historical-test performance,
publish forecasts, create forecast UI pages, or change the production Silver model.

Reserved later UI routes remain:

- `/forecast/4h`
- `/forecast/12h`
- `/forecast/1d`
- `/forecast/2d`
- `/forecast/30d`

## Historical provenance remains immutable

The original BullionVault chart exports remain byte-for-byte under
`research_data/bullionvault_horizons/raw/`. Stage 2 consumes the Stage-1 manifest and
rechecks each source SHA256 before building a dataset.

No Open is fabricated. There is no resampling, forward fill, backward fill,
interpolation, nearest-time alignment, or timezone inference. The newest source row
is excluded when Stage 1 marked it `POTENTIALLY_INCOMPLETE`.

Direct source coverage currently produces:

- 4h: 172 causal labeled rows after 6-row warmup and one unlabeled tail row.
- 12h: 172 causal labeled rows.
- 2d: 172 causal labeled rows.
- 30d: 232 causal labeled rows.
- 1d: blocked; no direct 86,400-second source export has been supplied.

These small samples are the reason Stage 2 preregisters simple regularized models
instead of deep learning.

## Timestamp policy

Source timestamps retain semantics `UNVERIFIED_EXPORT_CLIENT_TIMEZONE`.
They are used only to preserve ordering and verify that each target is exactly one
registered source interval after its feature row.

No hour-of-day, weekday, month, session, or other calendar feature is allowed while
the timezone remains unverified. Historical chart exports must not be silently joined
to UTC live data.

## Target

The frozen Stage-2 target version is:

`next-source-bar-log-return-v1`

For feature row `t`:

`target_log_return = log(close[t+1] / close[t])`

The target timestamp must be exactly one registered source interval after the feature
timestamp. A later predicted price can be reconstructed as:

`predicted_close = current_close * exp(predicted_log_return)`

The random-walk benchmark predicts zero log return, which is equivalent to predicting
that the next close equals the current close.

## Causal feature set

Feature-set version:

`bullionvault-hlc-causal-features-v1`

The same 12 compact HLC-only features are defined separately for each horizon:

- close log returns over 1, 2, 3, and 6 source bars;
- current HLC range percentage;
- current close location inside the High/Low range;
- rolling 3- and 6-bar mean and volatility of the one-bar return;
- rolling 3- and 6-bar mean HLC range percentage.

Maximum lookback is six source bars. All features at row `t` depend only on source
observations at or before `t`. A future-perturbation test proves that modifying future
HLC rows cannot alter prior features.

Feature fingerprint:

`5ad621a8b432f874566b115887200e0008a8fe5e4bba207a689894b5de242043`

## Preregistered comparison

The locked preregistration is stored at:

`research_data/bullionvault_horizons/stage2_preregistration.json`

Fingerprint:

`fcf19e14ef55932093cd5406034700469b1e04723ac3d11b6c543345cb33b1d6`

Each horizon is evaluated independently. Cross-horizon pooling is not allowed.

Baseline:

`random_walk_zero_return`

Candidate registry is fixed before any Stage-3 metrics are computed:

1. `ridge_alpha_10`
2. `huber_v1`
3. `elastic_net_v1`

Every candidate uses `StandardScaler` fit on the training slice only. There is no
development-time hyperparameter search in this preregistration.

## Walk-forward policy

The historical sample is split deterministically:

- final contiguous historical test: 20%, with at least 30 rows;
- development model selection: first 80%;
- four expanding walk-forward validation folds;
- one source bar purged immediately before each validation fold;
- at least 60 training rows and 15 validation rows per fold.

For the current 172-row horizons, the locked historical test contains 35 rows.
For 30d with 232 rows, the locked historical test contains 47 rows.

Stage 2 designates these test rows but does not read performance metrics from them.

## Development selection gate

Primary metric is MAE of `target_log_return`.

A candidate must satisfy all preregistered development gates:

- beat the random-walk OOF MAE;
- beat it in at least 3 of 4 walk-forward folds;
- paired block-bootstrap 95% MAE-improvement CI lower bound above zero;
- paired block-bootstrap probability of lower MAE at least
  `0.9833333333333333`;
- directional-accuracy delta no worse than -2 percentage points.

Bootstrap configuration is fixed at 5,000 iterations, block length four rows, and
seed `20260813`. Bootstrap probability is descriptive evidence, not a classical
p-value.

If no candidate passes, the random-walk baseline remains the result. If multiple
candidates pass, the largest development OOF MAE improvement wins; the fixed
tie-break order is Ridge, Huber, then ElasticNet.

## Locked historical confirmation

Only after the feature set and one winning candidate for a horizon are locked may
the designated historical test be read once.

The confirmation gate requires positive MAE improvement, positive 95% CI lower
bound, bootstrap probability at least 0.95, and no directional-accuracy degradation
greater than two percentage points.

Any feature or hyperparameter change after reading that test invalidates the
candidate version and requires a new version plus new forward-shadow evidence.

## Separation from authenticated BullionVault live data

Railway credentials continue to support the existing read-only BullionVault
authenticated market adapter and current Bid/Ask collection. Those forward snapshots
are a separate provenance stream.

Stage 2 does not backfill historical chart bars from live quotes and does not silently
merge historical client-timezone exports with UTC live data. A later stage may create
append-only forward bars with explicit source semantics.

## Guardrails

All non-negotiable project safety invariants remain unchanged:

- `edge_status = NOT_PROVEN`
- `research_only = true`
- `buy_sell_enabled = false`
- `execution_enabled = false`
- `live_model_mutated = false`
- `frozen_52_feature_graph_mutated = false`
- `future_holdout_read = false`
- `shadow62_mutated = false`

The existing production 52-feature H1 predictor, formal future holdout, Shadow62
protocol, and BullionVault microstructure collector remain isolated.

## Next stage

Stage 3 may execute the preregistered **development-only** walk-forward comparison
for Random Walk, Ridge, Huber, and ElasticNet independently for 4h, 12h, 2d, and 30d.

The locked historical-test blocks must remain unread until a winner for that horizon
has been selected and frozen from development evidence. The 1d horizon remains
blocked until a direct daily dataset is supplied or a separately preregistered
alternative provenance path is approved.
