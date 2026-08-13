# XPT/XPD 62-Feature Shadow Research Protocol

## Purpose

`Shadow62` evaluates the historically confirmed XPT/XPD candidate as a parallel research process without changing the production Silver forecast. The live model remains the frozen 52-feature system. The shadow candidate is `ridge_alpha_100` over the same 52 Silver features plus exactly 10 locked Platinum/Palladium features.

Candidate ID: `xpt-xpd-candle-shape-own-returns-v1`.

Locked auxiliary features:

- XPT/XPD candle range and candle body percentage.
- XPT/XPD exact-clock log returns at 1h, 6h, and 24h.

No rejected or redundant feature family from the earlier 43-feature experiment is restored.

## Causality and source semantics

The auxiliary source is the keyless Dukascopy public H1 Bid history for `XPT.CMD/USD` and `XPD.CMD/USD`. H1 timestamps mean `BAR_START_UTC`; values are usable only after the corresponding bar has closed. Alignment is exact UTC timestamp equality only. Source gaps remain missing. There is no forward fill, backward fill, interpolation, nearest-time match, or synthetic candle generation.

A shadow prediction for feature bar `t` is admissible only before the close of its target H1 bar could already be known. Missed observations are not reconstructed retrospectively.

## Sealed forward window

- Freeze ID: `xpt-xpd-shadow-62-v1-20260814`
- First feature bar: `2026-08-14T00:00:00Z`
- End exclusive: `2027-02-10T00:00:00Z`
- Earliest final score: `2027-02-10T02:00:00Z`
- Fixed duration: 180 days
- Minimum exact-hour outcomes: 2500

The database stores immutable shadow predictions and realized closes in separate append-only tables. Public API access exposes operational counts and timestamps only; prediction values, outcome values, and performance metrics remain sealed during the active holdout.

The final scorer is intentionally not implemented before the end of the fixed window. There are no interim MAE, directional-accuracy, correlation, strategy, or promotion metrics.

## Runtime isolation

The extension is disabled by default and installed around the existing FastAPI lifespan rather than modifying `LivePredictionEngine` or the frozen 52-feature graph.

Runtime variables:

```text
SHADOW62_ENABLED=false
SHADOW62_DELAY_MINUTES=8
SHADOW62_DB_PATH=runtime/xpt_xpd_shadow62.sqlite3
```

Operational status endpoint:

```text
GET /api/v1/research/shadow62/status
```

## Non-negotiable guardrails

`edge_status=NOT_PROVEN`, `research_only=true`, no automatic live promotion, no modification of the frozen 52-feature live graph, and no trading/execution path is introduced by this component.
