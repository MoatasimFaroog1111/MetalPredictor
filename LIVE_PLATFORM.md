# Silver AI Forecast — Live Platform

This layer turns the research repository into an operational **research-only** forecasting service without changing `baseline-v1` or the frozen Stage-7 future holdout.

## What runs in production

- **Baseline:** frozen `ridge_alpha_100`
- **Research challenger:** frozen `ridge_alpha_10`
- **Features:** the exact existing 52 causal Silver features
- **Target:** next exact-hour XAG/USD log return
- **UI:** same-origin responsive PWA for desktop/mobile
- **API:** FastAPI
- **Storage:** SQLite by default behind a repository interface
- **Primary operational market feed:** Gold API XAG completed-hour OHLC
- **Optional fallback market feed:** Twelve Data XAG/USD M1, conservatively aggregated to H1
- **Optional notifications:** Telegram Bot API
- **Trading execution:** disabled

The live service never fits a model, tunes a parameter, scores the frozen future holdout, or emits BUY/SELL commands.

## Required deployment files

Use `Dockerfile.live`. The image contains the frozen model payloads, frozen historical feature context, the PWA, and the API service.

Persist `/data` when running the container. The default production database is `/data/live_predictions.sqlite3`.

### v1 scaling boundary

Run **exactly one application replica/process** for v1. SQLite and the in-process hourly scheduler are deliberately a single-instance deployment design. The Docker command starts one Uvicorn process.

Do not horizontally scale this image by adding replicas. When scale-out is required, keep the same API/inference contracts but replace `SQLiteForecastRepository` with a shared PostgreSQL implementation and move hourly scheduling to a single external scheduler/worker. This prevents duplicate collectors and split-brain local databases.

## Environment variables

Never commit real values. Store them in the hosting platform's secret manager.

| Variable | Required | Purpose |
|---|---|---|
| `LIVE_ADMIN_TOKEN` | Yes for writes/admin | Protects ingestion and admin endpoints |
| `LIVE_MARKET_PROVIDER` | No | `auto` (default), `goldapi`, or `twelvedata` |
| `GOLD_API_KEY` | For Gold API automatic collection | Authenticates XAG OHLC requests |
| `GOLD_API_SYMBOL` | No | Defaults to `XAG` |
| `TWELVEDATA_API_KEY` | Only for Twelve Data fallback | XAG/USD M1 operational feed |
| `TWELVEDATA_SYMBOL` | No | Defaults to `XAG/USD` |
| `LIVE_AUTO_COLLECT` | No | Set `true` to run the internal hourly collector |
| `LIVE_COLLECTION_DELAY_MINUTES` | No | Minutes after each UTC hour before collection; default `5` |
| `TELEGRAM_BOT_TOKEN` | Only for Telegram | Bot token from BotFather |
| `TELEGRAM_WEBHOOK_SECRET` | Only for Telegram commands | Secret checked on Telegram webhook requests |
| `TELEGRAM_ALLOWED_CHAT_IDS` | Only for Telegram | Comma-separated allowlist |
| `PUBLIC_BASE_URL` | Only for Telegram webhook | Public HTTPS origin, no trailing slash |
| `LIVE_DB_PATH` | No | Runtime database path |
| `PORT` | No | HTTP port; default `8000` |

In `auto` mode, Gold API is preferred when `GOLD_API_KEY` is present. Otherwise Twelve Data is selected when `TWELVEDATA_API_KEY` is present.

See `.env.live.example` for names only.

## Start locally

Install `requirements-live.txt`, install the project package, set `LIVE_ADMIN_TOKEN`, then run:

```bash
python run_live.py
```

Open `http://localhost:8000/` for the PWA and `/docs` for the API schema.

## Automatic hourly forecasts

Recommended production configuration:

```text
LIVE_MARKET_PROVIDER=goldapi
GOLD_API_KEY=<secret>
GOLD_API_SYMBOL=XAG
LIVE_AUTO_COLLECT=true
LIVE_COLLECTION_DELAY_MINUTES=5
```

The service runs at the configured minute after each UTC hour and requests the previous completed hour.

### Gold API free-tier behavior

Gold API exposes XAG OHLC directly. The adapter requests an exact completed UTC-hour range and converts the provider's precious-metal USD/troy-ounce values to the model's USD/kg convention.

The free Gold API plan limits OHLC/history calls. To stay inside that operational boundary, a multi-hour catch-up request intentionally fetches only the **newest requested completed hour**. Missing hours from downtime remain explicit timestamp gaps. They are not forward-filled, interpolated, or replaced with synthetic bars.

This is safe for the frozen runtime contract because the exact-clock feature graph already represents missing lags as `NaN`, and the sealed frozen Ridge payload reproduces the fitted training-time median imputer and missing-value indicators. A restart can therefore resume with the newest completed hour without manufacturing historical observations.

The Gold API bar is labeled `source_provider=GoldAPI`, `source_symbol=XAG`, `market_type=spot_quote`, and `quality_flag=PROVIDER_AGGREGATED_H1`. It is an operational cross-feed and is not assumed identical to HistData spot-bid training data, so resulting forecasts expose `source_compatible_with_training=false`.

### Twelve Data fallback

When `LIVE_MARKET_PROVIDER=twelvedata`, or when `auto` has no Gold API key but has a Twelve Data key, Twelve Data M1 is requested in bounded blocks of at most 72 hours. The returned minute rows pass through the existing `ConservativeH1Aggregator` to construct H1 bars.

Twelve Data is also treated as an operational cross-feed, not as a replacement for the frozen training dataset.

## Gap and forecast semantics

Backfilled or resumed market bars are feature context only. The service deliberately does **not** manufacture retroactive “live forecasts” for all missing historical hours. Only the requested latest completed hour may materialize a forecast. This keeps the live forecast history auditable.

Market closures and source gaps are never forward-filled. Exact-clock lag features can therefore be `NaN` after normal market closures, provider gaps, or downtime. This is intentional and matches the sealed runtime contract. Live inference preserves those `NaN` values and rejects infinite feature values; it does not invent prices or interpolate exact-clock history. If the requested latest H1 bar itself is unavailable, no forecast is created.

A protected manual collection trigger is available after deployment:

```text
POST /api/v1/admin/collect
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

With no timestamp it collects through the previous completed UTC hour. An explicit timezone-aware `hour_start_utc` can be supplied to cap collection at an earlier hour.

## Manual H1 ingestion

Automatic collection is optional. A trusted upstream source can send completed H1 bars sequentially to:

```text
POST /api/v1/market/silver/hourly
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

The request is idempotent. Re-sending the identical bar is accepted without duplicating it. Re-sending a changed bar at the same timestamp raises a revision conflict rather than silently rewriting history. Expected missing exact-clock lag features are handled only by the frozen model's sealed training-time imputer; the service never forward-fills them.

## Telegram

Create a bot with BotFather, store the token only as `TELEGRAM_BOT_TOKEN`, and allowlist the intended chat IDs in `TELEGRAM_ALLOWED_CHAT_IDS`.

Also configure:

```text
TELEGRAM_WEBHOOK_SECRET=<strong random secret>
PUBLIC_BASE_URL=https://your-public-domain
```

After deployment, call once with the admin header:

```text
POST /api/v1/admin/telegram/configure-webhook
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

The backend registers:

```text
https://your-public-domain/api/v1/telegram/webhook
```

with Telegram's secret-token protection. Only allowlisted chats are acted on.

Supported bot commands:

- `/latest` — latest frozen-model forecast
- `/status` — service/model status
- `/start` or `/help` — command help

Newly materialized forecasts are pushed to the allowlisted chats when Telegram notifications are enabled. Notification delivery is best-effort: a temporary Telegram outage does not undo or invalidate an already persisted forecast.

## Main API

```text
GET  /api/v1/health
GET  /api/v1/status
GET  /api/v1/model/status
GET  /api/v1/forecast/latest
GET  /api/v1/forecast/history
GET  /api/v1/market/silver/recent
POST /api/v1/market/silver/hourly
POST /api/v1/admin/collect
POST /api/v1/admin/telegram/configure-webhook
POST /api/v1/telegram/webhook
```

Admin endpoints require `X-Admin-Token`. API responses use `Cache-Control: no-store`; the PWA receives restrictive security headers and a same-origin Content Security Policy.

## Interpretation boundary

The dashboard reports `UP`, `DOWN`, predicted return, predicted USD/kg price, data quality, source, model version, and history. It deliberately reports:

```text
edge_status = NOT_PROVEN
research_only = true
buy_sell_enabled = false
```

until the predeclared future evidence standard is actually met. The live platform must not be used to alter the frozen Stage-7 protocol or inspect its interim performance.
