# BullionVault Forward Multi-Horizon Data Factory — Stage 4

Stage 4 adds an isolated append-only data factory for BullionVault Silver. It consumes only snapshots already observed and stored by the existing read-only microstructure collector; it makes no additional BullionVault API requests.

## Horizons

Direct UTC fixed-duration bars are built for 4h (14,400s), 12h (43,200s), 1d (86,400s), 2d (172,800s), and 30d (2,592,000s). Buckets are anchored to Unix epoch. The 30d track is a fixed 30-day interval, not a calendar month.

For each completed bucket, midpoint is `(best_bid + best_ask) / 2`. Open/High/Low/Close are calculated only from actually observed midpoint snapshots inside `[bucket_start, bucket_end)`. Bid/Ask endpoints, spread statistics, sample count, expected count, coverage, access-mode counts, freshness counts, and first/last sample timestamps are preserved.

No Open is fabricated. There is no forward fill, backward fill, interpolation, nearest-time matching, synthetic flat candle, or historical Chart backfill. Fewer than two observed snapshots produces an explicit immutable gap record instead of a bar.

Historical BullionVault Chart CSV data remains a separate provenance stream. Authenticated, public-cached, or mixed forward observations are never silently relabeled.

## Persistence and API

The factory writes to a separate append-only SQLite database with one immutable assessment per horizon/bucket. Research routes are:

- `/api/v1/research/forward-bars/status`
- `/api/v1/research/forward-bars/latest?horizon=4h`
- `/api/v1/research/forward-bars/history?horizon=4h&limit=100`

## Railway variables

No new secret is required:

```env
BULLIONVAULT_FORWARD_BARS_ENABLED=true
BULLIONVAULT_FORWARD_BARS_DB_PATH=/data/bullionvault_forward_bars.sqlite3
BULLIONVAULT_FORWARD_BARS_MATERIALIZATION_INTERVAL_SECONDS=60
BULLIONVAULT_FORWARD_BARS_CLOSE_DELAY_SECONDS=120
BULLIONVAULT_FORWARD_BARS_MAX_BUCKETS_PER_CYCLE=512
```

The existing microstructure collector should remain enabled for continuous evidence.

Safety remains unchanged: `edge_status=NOT_PROVEN`, `research_only=true`, BUY/SELL disabled, execution disabled, no live-model mutation, no frozen-52-feature mutation, no Shadow62 mutation, and no automatic promotion.
