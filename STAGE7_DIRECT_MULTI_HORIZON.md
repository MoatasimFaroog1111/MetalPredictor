# Stage 7 — Direct Multi-Horizon Research

Stage 7 is an isolated research study. It does not modify the frozen live one-hour model, the formal future holdout, Shadow62, BullionVault collection, execution controls, or production forecast routes.

## Source and provenance

The study uses the repository's canonical five-year H1 silver dataset:

- `XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet`
- instrument: XAG/USD
- unit: USD/kg
- preregistered Git blob: `9b95fcc5aa2679208c6b5c44c830ce6b1eaa5829`

The executable Stage-7 runner reads the parquet bytes exactly once, verifies that immutable byte snapshot against the preregistered Git-blob SHA-1, and supplies the same verified in-memory snapshot to every horizon. A modified source fails closed at capture time, and a later filesystem change cannot alter the bytes consumed by a running study. CI performs the source check independently and also reruns whenever the canonical source or any direct Stage-7 feature/statistics dependency changes.

BullionVault forward bars and microstructure snapshots are a different provenance domain and are not merged into Stage-7 training data.

## Direct targets

Five independent horizons are evaluated from the H1 clock:

- 4h: `log(close[t+4h] / close[t])`
- 12h: `log(close[t+12h] / close[t])`
- 1d: `log(close[t+24h] / close[t])`
- 2d: `log(close[t+48h] / close[t])`
- 30d: `log(close[t+720h] / close[t])`

A target exists only when the exact UTC timestamp `t+h` exists in the canonical source. Nearest-neighbor matching, as-of joins, interpolation, forward fill, and fabricated bars are forbidden.

## Causal features

Stage 7 reuses the existing six canonical feature components without changing them:

`PriceActionFeatures`, `MomentumFeatures`, `VolatilityFeatures`, `TrendFeatures`, `TemporalFeatures`, and `QualityFeatures`.

Their combined graph is asserted to contain exactly 52 unique features. Missing exact-clock lag features remain missing and any required imputation is fitted inside the training fold only.

## Locked development protocol

The latest 20% feature-time region is reserved as a Stage-7 historical test. During Stage 7, that partition is metadata-only: no test metric, test prediction, fit, bootstrap, or model-selection operation is authorized.

Development uses four chronological expanding validation folds. Before each validation boundary, training rows are purged until every retained training label timestamp is strictly earlier than the first validation feature timestamp. Purge and recorded embargo are both equal to the target horizon: 4, 12, 24, 48, or 720 hours.

Because training is expanding and only uses observations before validation, no post-validation future rows can enter the training set. The horizon-sized embargo is recorded explicitly for auditability.

## Preregistered candidates

The candidate recipes are frozen before results are observed:

- Ridge, alpha 100, train-only median imputation and scaling.
- HistGradientBoosting with absolute-error loss and fixed shallow regularization.
- ExtraTrees with fixed shallow/regularized parameters.

The benchmark is `random_walk_zero_return`, equivalent to forecasting no log-price change over the horizon.

There is no hyperparameter search and no result-driven candidate editing in Stage 7.

## Development gate

MAE of the direct horizon log return is primary. A candidate must beat the random-walk OOF MAE, beat it in at least 3 of 4 folds, have a positive lower bound of the preregistered 95% paired moving-block bootstrap interval, achieve at least 0.99 bootstrap probability of lower MAE, and avoid a directional-accuracy deterioration worse than 2 percentage points.

If no candidate passes every gate, the study retains the random-walk baseline and the locked historical test remains sealed. Passing development does not authorize deployment; it only makes a separately approved historical confirmation stage scientifically eligible.

## Safety

Stage 7 remains `NOT_PROVEN`, research-only, with BUY/SELL and execution disabled. It writes a development report only. It does not produce a deployable candidate model artifact and cannot promote anything to the live service automatically.
