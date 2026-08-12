# Railway deployment — Silver AI Forecast

This deployment is **research-only**. Predictive edge remains `NOT_PROVEN`; BUY/SELL execution stays disabled.

## Repository configuration

`railway.json` is the deployment source of truth for build/deploy settings:

- builder: `DOCKERFILE`
- Dockerfile: `Dockerfile.live`
- health check: `/api/v1/health`
- restart: `ON_FAILURE`, max 10 retries
- no start-command override; the Docker image `CMD` remains authoritative
- watch patterns deliberately exclude `forward_holdout/observations.csv`, `forward_holdout/predictions.csv`, and `forward_holdout/collection_state.json`, so append-only holdout ledger updates do not redeploy the web service

## Railway service settings

Use the private `MetalPredictor` repository and `main` branch. Keep exactly one replica for v1.

Attach a persistent Railway Volume at:

```text
/data
```

The Docker image defaults the SQLite database to:

```text
/data/live_predictions.sqlite3
```

The container starts with only the privileges needed to prepare ownership of the mounted runtime directory, then immediately drops to `appuser` (UID `10001`) before starting FastAPI. Do **not** set `RAILWAY_RUN_UID=0`; the application process is intentionally non-root. The volume/privilege-drop path is covered by the sanitized public Docker CI mirror.

## Required variables

Set real values only in Railway Variables/Secrets; never commit them:

```text
LIVE_ADMIN_TOKEN=<strong-random-secret>
LIVE_MARKET_PROVIDER=goldapi
GOLD_API_KEY=<secret>
GOLD_API_SYMBOL=XAG
LIVE_AUTO_COLLECT=true
LIVE_COLLECTION_DELAY_MINUTES=5
LIVE_DB_PATH=/data/live_predictions.sqlite3
```

`LIVE_MARKET_PROVIDER=auto` is also supported. In `auto` mode, a configured `GOLD_API_KEY` is preferred; if no Gold API key exists, the service can fall back to Twelve Data when `TWELVEDATA_API_KEY` is configured.

Optional Twelve Data fallback:

```text
TWELVEDATA_API_KEY=<secret>
TWELVEDATA_SYMBOL=XAG/USD
```

For Telegram also set:

```text
TELEGRAM_BOT_TOKEN=<secret>
TELEGRAM_WEBHOOK_SECRET=<strong-random-secret>
TELEGRAM_ALLOWED_CHAT_IDS=<comma-separated-chat-ids>
PUBLIC_BASE_URL=https://<railway-public-domain>
```

`PORT` is read dynamically by `run_live.py`; do not hard-code an external Railway port.

## Gold API free-tier operational behavior

Gold API provides the completed-hour OHLC adapter used by the live service. The free plan limits OHLC/history requests, so a catch-up range intentionally requests only the **newest requested completed hour**. Older missing hours remain explicit gaps rather than being fabricated or repeatedly requested until the rate limit is exhausted.

The frozen feature graph already treats missing exact-clock lags as missing values, and the sealed frozen Ridge payload applies the training-time fitted imputer/missing indicators. Therefore a restart after downtime can resume with the newest completed hour and still produce a research-only forecast while exposing the source as an operational cross-feed.

Once the service is running continuously, the normal path is one completed Gold API H1 OHLC request after each UTC hour.

## First deployment verification

After Railway reports the deployment healthy, verify these public read-only endpoints:

```text
GET /api/v1/health
GET /api/v1/status
GET /api/v1/model/status
```

Expected market-source state with Gold API configured:

```text
provider = GoldAPI
symbol = XAG
mode = PROVIDER_H1_OHLC_LATEST_ONLY
```

Expected safety state:

```text
edge_status = NOT_PROVEN
buy_sell_enabled = false
research_only = true
```

Then trigger the protected collection once if automatic collection has not already run:

```text
POST /api/v1/admin/collect
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

No source gap is forward-filled or interpolated. Exact-clock NaNs remain NaN until the sealed frozen model applies its fitted training-time imputer/missing indicators.

## Telegram activation

After `PUBLIC_BASE_URL` points to the live HTTPS domain, register the webhook once:

```text
POST /api/v1/admin/telegram/configure-webhook
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

Only allowlisted chat IDs are processed.
