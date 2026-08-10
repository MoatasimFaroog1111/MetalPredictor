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
- **Optional market feed:** Twelve Data XAG/USD M1, conservatively aggregated to H1
- **Optional notifications:** Telegram Bot API
- **Trading execution:** disabled

The live service never fits a model, tunes a parameter, scores the frozen future holdout, or emits BUY/SELL commands.

## Required deployment files

Use `Dockerfile.live`. The image contains the frozen model payloads, frozen historical feature context, the PWA, and the API service.

Persist `/data` when running the container. The default production database is `/data/live_predictions.sqlite3`.

## Environment variables

Never commit real values. Store them in the hosting platform's secret manager.

| Variable | Required | Purpose |
|---|---|---|
| `LIVE_ADMIN_TOKEN` | Yes for writes/admin | Protects ingestion and admin endpoints |
| `TWELVEDATA_API_KEY` | Only for automatic market collection | XAG/USD M1 operational feed |
| `TWELVEDATA_SYMBOL` | No | Defaults to `XAG/USD` |
| `LIVE_AUTO_COLLECT` | No | Set `true` to run the internal hourly catch-up/collector |
| `LIVE_COLLECTION_DELAY_MINUTES` | No | Minutes after each UTC hour before collection; default `5` |
| `TELEGRAM_BOT_TOKEN` | Only for Telegram | Bot token from BotFather |
| `TELEGRAM_WEBHOOK_SECRET` | Only for Telegram commands | Secret checked on Telegram webhook requests |
| `TELEGRAM_ALLOWED_CHAT_IDS` | Only for Telegram | Comma-separated allowlist |
| `PUBLIC_BASE_URL` | Only for Telegram webhook | Public HTTPS origin, no trailing slash |
| `LIVE_DB_PATH` | No | Runtime database path |
| `PORT` | No | HTTP port; default `8000` |

See `.env.live.example` for names only.

## Start locally

Install `requirements-live.txt`, install the project package, set `LIVE_ADMIN_TOKEN`, then run:

```bash
python run_live.py
```

Open `http://localhost:8000/` for the PWA and `/docs` for the API schema.

## Automatic hourly forecasts and startup catch-up

With these secrets configured:

```text
TWELVEDATA_API_KEY=<secret>
LIVE_AUTO_COLLECT=true
LIVE_COLLECTION_DELAY_MINUTES=5
```

the service runs at the configured minute after each UTC hour. Before trying to forecast, it detects the latest H1 bar already persisted and requests every missing completed hour from that point through the previous completed hour.

The frozen historical feature context currently ends before the live service starts. Therefore startup catch-up is mandatory for causal continuity: the service **does not** jump from the frozen history directly to today's latest hour.

Twelve Data M1 catch-up is requested in bounded blocks of at most 72 hours. At one-minute resolution this is at most 4,320 possible data points per request, below the provider's documented 5,000-point ceiling. `start_date` and `end_date` bound each request; the returned M1 rows then pass through the existing `ConservativeH1Aggregator`.

Backfilled H1 bars are feature context only. The service deliberately does **not** manufacture retroactive “live forecasts” for all missing historical hours. After catch-up, only the requested latest completed hour may materialize a forecast. This keeps the live forecast history auditable.

Market closures and source gaps are not forward-filled. If the requested latest hour does not exist, no forecast is created. If the exact frozen 52-feature vector is incomplete, inference fails closed with `LIVE_FEATURES_INCOMPLETE` and waits for a later eligible hour rather than imputing data.

The Twelve Data feed is explicitly marked as an operational cross-feed. It is **not assumed identical** to the HistData spot-bid training feed, so forecasts produced from it expose `source_compatible_with_training=false`.

A protected manual catch-up trigger is available after deployment:

```text
POST /api/v1/admin/collect
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

With no timestamp it catches up through the previous completed UTC hour. An explicit timezone-aware `hour_start_utc` can be supplied to cap the catch-up at an earlier hour.

## Manual H1 ingestion

Automatic collection is optional. A trusted upstream source can send completed H1 bars sequentially to:

```text
POST /api/v1/market/silver/hourly
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

The request is idempotent. Re-sending the identical bar is accepted without duplicating it. Re-sending a changed bar at the same timestamp raises a revision conflict rather than silently rewriting history. If earlier context is missing, the bar can still be persisted but forecast materialization fails closed until the causal 52-feature vector is complete.

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

Newly materialized forecasts are also pushed to the allowlisted chats when Telegram notifications are enabled.

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
