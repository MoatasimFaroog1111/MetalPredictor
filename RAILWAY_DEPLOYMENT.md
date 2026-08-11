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
TWELVEDATA_API_KEY=<secret>
TWELVEDATA_SYMBOL=XAG/USD
LIVE_AUTO_COLLECT=true
LIVE_COLLECTION_DELAY_MINUTES=5
LIVE_DB_PATH=/data/live_predictions.sqlite3
```

For Telegram also set:

```text
TELEGRAM_BOT_TOKEN=<secret>
TELEGRAM_WEBHOOK_SECRET=<strong-random-secret>
TELEGRAM_ALLOWED_CHAT_IDS=<comma-separated-chat-ids>
PUBLIC_BASE_URL=https://<railway-public-domain>
```

`PORT` is read dynamically by `run_live.py`; do not hard-code an external Railway port.

## First deployment verification

After Railway reports the deployment healthy, verify these public read-only endpoints:

```text
GET /api/v1/health
GET /api/v1/status
GET /api/v1/model/status
```

Expected safety state:

```text
edge_status = NOT_PROVEN
buy_sell_enabled = false
research_only = true
```

Then trigger the protected catch-up once if automatic collection has not already run:

```text
POST /api/v1/admin/collect
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

The catch-up fills missing causal H1 context but does not manufacture retroactive live forecasts. Exact-clock NaNs caused by market closures remain NaN until the sealed frozen model applies its fitted training-time imputer/missing indicators.

## Telegram activation

After `PUBLIC_BASE_URL` points to the live HTTPS domain, register the webhook once:

```text
POST /api/v1/admin/telegram/configure-webhook
X-Admin-Token: <LIVE_ADMIN_TOKEN>
```

Only allowlisted chat IDs are processed.
