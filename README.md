# MetalPredictor — Silver Training Data Pipeline

Leakage-safe preparation of hourly XAG/USD data for supervised forecasting.

## Component architecture

- `data`: canonical loading and raw-data validation
- `features`: independent causal feature components
- `targets`: future-label construction only
- `splitting`: chronological purged Train/Validation/Test
- `leakage`: fail-fast leakage checks
- `artifacts`: output serialization
- `pipeline`: orchestration only

The implementation follows SOLID principles: components have one responsibility, depend on contracts, and can be replaced independently.

## Forecast target

The supervised target is the next **exact** hourly bar:

- `target_log_return_1h`
- `target_close_usd_per_kg`
- `target_timestamp_utc`

Rows whose next source record is not exactly one hour later receive no target and are excluded. This prevents weekend/session gaps from being mislabeled as one-hour forecasts.

## Causal features

Features use only information available at or before completed bar `t`:

- candle geometry
- 1/3/6/12/24/72/168-hour returns and momentum
- RSI
- realized volatility
- ATR
- SMA/EMA-relative trend
- cyclic UTC hour/weekday
- previous-bar time gap
- current source-hour quality indicator

No centered rolling windows, backfills, future interpolation, or negative feature shifts are used.

## Leakage protection

The pipeline validates chronology and OHLC, builds targets in a separate target component, removes incomplete samples, applies a chronological 70/15/15 split with purged boundaries, verifies label isolation across splits, rejects target/future-like feature names, and rejects NaN/Inf in the final feature matrix.

Scaling is intentionally model-specific and must be fit on Train only, then applied to Validation and Test.

## Run

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python build_training_dataset.py
```

Outputs:

```text
data/processed/
├── train.parquet
├── validation.parquet
├── test.parquet
├── feature_manifest.json
├── split_manifest.json
└── data_quality_report.json
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```
