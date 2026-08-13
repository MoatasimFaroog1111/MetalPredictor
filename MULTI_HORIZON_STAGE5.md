# BullionVault Multi-Horizon Research — Stage 5

Stage 5 exposes research pages for the fixed 4h, 12h, 1d, 2d, and 30d forward-bar horizons. It does not create a new predictive model and it does not change the existing live H1 model, Shadow62, or either sealed holdout.

## Data admission

A completed forward bar is admissible for the Stage-5 page only when all of these checks pass:

- coverage is at least 90%;
- every stored snapshot is `AUTHENTICATED_READ_ONLY`;
- every stored snapshot has freshness `CURRENT_GUI_SOURCE`;
- at least two real observations exist;
- observed snapshot count does not exceed the expected cadence count.

The latest completed bar must itself pass. Stage 5 does not silently fall back to an older admitted bar when the newest completed bar fails the gate.

There is no forward fill, backward fill, interpolation, synthetic candle, historical Chart fallback, or silent provenance mixing.

## Baseline policy

The only value published by Stage 5 is `random_walk_zero_return`:

`predicted next close midpoint = latest admitted close midpoint`

For 4h, 12h, 2d, and 30d, this is the baseline retained by the locked Stage-3 development result after none of Ridge, Huber, or ElasticNet passed the preregistered gate. Historical confirmation was therefore not authorized.

The 1d track had no direct historical Stage-3 dataset. Its random-walk value is explicitly labeled `BASELINE_REFERENCE_ONLY`, not a historically selected predictive model.

No prediction interval is fabricated. A page shows `COLLECTING_EVIDENCE` instead of a numeric baseline whenever the latest completed forward bar fails admission or no completed bar exists yet.

## Routes

Pages:

- `/forecast/4h`
- `/forecast/12h`
- `/forecast/1d`
- `/forecast/2d`
- `/forecast/30d`

Research API:

- `/api/v1/research/multi-horizon-forecast/status`
- `/api/v1/research/multi-horizon-forecast/{horizon}`

The status route intentionally omits numeric forecast values and exposes only operational state and evidence quality.

## Safety

The invariant state remains:

- `edge_status=NOT_PROVEN`
- `research_only=true`
- execution disabled
- automatic promotion disabled
- production 52-feature graph unchanged
- Shadow62 unchanged
- historical Chart data not merged into forward bars
- no interim holdout performance metric is computed by Stage 5
