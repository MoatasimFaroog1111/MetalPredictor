# BullionVault Multi-Horizon Research — Stage 1

## Scope

This component prepares isolated research tracks for Silver forecasts at 4h, 12h,
1d, 2d, and 30d horizons. Stage 1 is data provenance and contracts only. It does
not train a model, publish a forecast, change the production 52-feature model, or
read any sealed holdout performance.

Routes reserved for later UI work:

- `/forecast/4h`
- `/forecast/12h`
- `/forecast/1d`
- `/forecast/2d`
- `/forecast/30d`

## Historical bootstrap

The four supplied BullionVault chart exports are stored byte-for-byte under
`research_data/bullionvault_horizons/raw/`. Their SHA256 digests and audit
statistics are locked in `manifest.json`.

The exports contain High/Low/Close in USD/kg and USD/troy-ounce. They do not
contain Open. Stage 1 never fabricates Open and never resamples, interpolates,
forward-fills, backward-fills, nearest-matches, or otherwise invents bars.

The provided intervals are:

- 4h: 14,400 seconds, 180 raw rows.
- 12h: 43,200 seconds, 180 raw rows.
- 2d: 172,800 seconds, 180 raw rows.
- 30d: 2,592,000 seconds, 240 raw rows.
- 1d: 86,400 seconds is intentionally `DATA_PENDING`; no direct daily file was supplied.

The newest row in each supplied export is conservatively marked
`POTENTIALLY_INCOMPLETE` when the filename export timestamp falls inside that
bar interval. The research loader excludes that row by default.

## Timestamp policy

BullionVault chart timestamps are retained as source-local, timezone-naive values
with semantics `UNVERIFIED_EXPORT_CLIENT_TIMEZONE`. No UTC conversion is guessed.
Until timezone semantics are verified, these historical exports must not be joined
to UTC live data by clock time and must not generate hour-of-day/calendar features.

## Authenticated BullionVault live source

Railway credentials already feed the existing read-only BullionVault market
adapter. Stage 1 does not add any order, cancel, account-mutation, or execution
method. A later stage may aggregate authenticated read-only Bid/Ask snapshots into
append-only forward bars, but those bars will be a separate provenance stream from
the historical chart exports and will not be silently mixed with them.

## Scientific guardrails

The multi-horizon component is research-only:

- `edge_status = NOT_PROVEN`
- `research_only = true`
- `buy_sell_enabled = false`
- `execution_enabled = false`
- `live_model_mutated = false`
- `frozen_52_feature_graph_mutated = false`

The existing live H1 predictor, formal future holdout, Shadow62 protocol, and
BullionVault microstructure collector are not modified by Stage 1.

## Next stage

Stage 2 may build causal per-horizon training datasets and preregister model
selection gates. Each horizon receives its own artifact and walk-forward process.
The missing direct 1d dataset remains blocked rather than being fabricated.
