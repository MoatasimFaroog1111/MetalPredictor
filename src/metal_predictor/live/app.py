from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from metal_predictor.live.catchup import LiveMarketCatchUpService
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.inference import LiveForecastOrchestrator, LivePredictionEngine
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.notifications import TelegramForecastPublisher
from metal_predictor.live.repository import SQLiteForecastRepository
from metal_predictor.live.scheduler import HourlyCollectionScheduler
from metal_predictor.live.settings import LiveSettings


class HourlyBarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime
    open_usd_per_kg: float = Field(gt=0)
    high_usd_per_kg: float = Field(gt=0)
    low_usd_per_kg: float = Field(gt=0)
    close_usd_per_kg: float = Field(gt=0)
    minute_count: int = Field(ge=1, le=60)
    quality_flag: str = Field(min_length=1, max_length=64)
    source_provider: str = Field(default="Manual", min_length=1, max_length=64)
    source_symbol: str = Field(default="XAGUSD", min_length=1, max_length=64)
    market_type: str = Field(default="manual_input", min_length=1, max_length=64)

    def to_contract(self) -> HourlySilverBar:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must include a timezone.")
        return HourlySilverBar(
            timestamp_utc=self.timestamp_utc.astimezone(timezone.utc),
            open_usd_per_kg=self.open_usd_per_kg,
            high_usd_per_kg=self.high_usd_per_kg,
            low_usd_per_kg=self.low_usd_per_kg,
            close_usd_per_kg=self.close_usd_per_kg,
            minute_count=self.minute_count,
            quality_flag=self.quality_flag,
            source_provider=self.source_provider,
            source_symbol=self.source_symbol,
            market_type=self.market_type,
        )


def create_app(settings: LiveSettings | None = None) -> FastAPI:
    config = settings or LiveSettings.from_environment()
    root = config.repository_root.resolve()
    db_path = config.database_path
    if not db_path.is_absolute():
        db_path = root / db_path

    repository = SQLiteForecastRepository(db_path)
    engine = LivePredictionEngine(root)
    notifier = (
        TelegramForecastPublisher(config.telegram_bot_token, config.telegram_allowed_chat_ids)
        if config.telegram_enabled
        else None
    )
    orchestrator = LiveForecastOrchestrator(repository, engine, notifier)
    source = (
        TwelveDataSilverMinuteSource(config.twelvedata_api_key, config.twelvedata_symbol)
        if config.market_source_enabled
        else None
    )
    catchup = (
        LiveMarketCatchUpService(source, repository, engine, orchestrator)
        if source is not None
        else None
    )
    scheduler = (
        HourlyCollectionScheduler(
            catchup,
            delay_minutes=config.collection_delay_minutes,
        )
        if config.auto_collection_enabled and catchup is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        scheduler_task: asyncio.Task[None] | None = None
        if scheduler is not None:
            scheduler_task = asyncio.create_task(
                scheduler.run_forever(),
                name="silver-hourly-catch-up",
            )
        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await scheduler_task

    app = FastAPI(
        title="Silver AI Forecast API",
        version="1.0.0",
        description=(
            "Operational research-only XAG/USD hourly forecasts using frozen causal models. "
            "BUY/SELL execution is intentionally disabled because predictive edge is not proven."
        ),
        lifespan=lifespan,
    )
    app.state.settings = config
    app.state.repository = repository
    app.state.engine = engine
    app.state.orchestrator = orchestrator
    app.state.market_source = source
    app.state.catchup = catchup
    app.state.telegram = notifier
    app.state.scheduler = scheduler

    static_dir = root / "live_web"
    if not static_dir.exists():
        raise FileNotFoundError(f"Live web directory not found: {static_dir}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        path = request.url.path
        if path == "/" or path == "/sw.js" or path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; manifest-src 'self'; worker-src 'self'"
            )
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)

    def require_admin(
        supplied: str | None = Depends(admin_header),
    ) -> None:
        expected = config.admin_token
        if not expected:
            raise HTTPException(status_code=503, detail="LIVE_ADMIN_TOKEN is not configured.")
        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Invalid admin token.")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker() -> FileResponse:
        return FileResponse(
            static_dir / "sw.js",
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "silver-ai-live",
            "edge_status": "NOT_PROVEN",
            "buy_sell_enabled": False,
        }

    @app.get("/api/v1/status")
    def status() -> dict[str, object]:
        latest = orchestrator.latest()
        recent_bars = repository.recent_bars(limit=1)
        return {
            "service": "silver-ai-live",
            "model": engine.model_status(),
            "market_source": {
                "configured": config.market_source_enabled,
                "provider": "TwelveData" if config.market_source_enabled else None,
                "symbol": config.twelvedata_symbol if config.market_source_enabled else None,
                "mode": "M1_TO_H1_CONSERVATIVE_AGGREGATION" if config.market_source_enabled else None,
            },
            "automatic_collection": {
                "enabled": config.auto_collection_enabled,
                "catch_up_enabled": catchup is not None,
                "delay_minutes_after_hour": config.collection_delay_minutes,
            },
            "telegram": {
                "notifications_enabled": config.telegram_enabled,
                "webhook_ready": config.telegram_webhook_enabled,
                "allowed_chat_count": len(config.telegram_allowed_chat_ids),
            },
            "latest_live_bar_timestamp_utc": (
                recent_bars[-1].timestamp_utc.isoformat() if recent_bars else None
            ),
            "latest_forecast_timestamp_utc": (
                latest.feature_timestamp_utc.isoformat() if latest else None
            ),
        }

    @app.get("/api/v1/model/status")
    def model_status() -> dict[str, object]:
        return engine.model_status()

    @app.get("/api/v1/forecast/latest")
    def latest_forecast() -> dict[str, object]:
        snapshot = orchestrator.latest()
        if snapshot is None:
            raise HTTPException(status_code=404, detail="No live forecast has been materialized yet.")
        return snapshot.as_dict()

    @app.get("/api/v1/forecast/history")
    def forecast_history(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict[str, object]]:
        return [item.as_dict() for item in repository.forecast_history(limit=limit)]

    @app.get("/api/v1/market/silver/recent")
    def recent_market(limit: Annotated[int, Query(ge=1, le=1000)] = 168) -> list[dict[str, object]]:
        return [item.as_dict() for item in repository.recent_bars(limit=limit)]

    @app.post(
        "/api/v1/market/silver/hourly",
        dependencies=[Depends(require_admin)],
    )
    def ingest_hourly(payload: HourlyBarRequest) -> dict[str, object]:
        try:
            snapshot, bar_created, forecast_created = orchestrator.ingest_and_forecast(
                payload.to_contract()
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "bar_created": bar_created,
            "forecast_created": forecast_created,
            "forecast": snapshot.as_dict(),
        }

    @app.post(
        "/api/v1/admin/collect",
        dependencies=[Depends(require_admin)],
    )
    def collect_market_through(
        hour_start_utc: datetime | None = None,
    ) -> dict[str, object]:
        if catchup is None:
            raise HTTPException(
                status_code=503,
                detail="TWELVEDATA_API_KEY is not configured; automatic collection is disabled.",
            )
        target = hour_start_utc or orchestrator.previous_completed_hour()
        if target.tzinfo is None:
            raise HTTPException(status_code=422, detail="hour_start_utc must include a timezone.")
        try:
            result = catchup.catch_up(target.astimezone(timezone.utc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"catch_up": result.as_dict()}

    @app.post(
        "/api/v1/admin/telegram/configure-webhook",
        dependencies=[Depends(require_admin)],
    )
    def configure_telegram_webhook() -> dict[str, object]:
        if notifier is None or not config.telegram_webhook_enabled:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Telegram requires TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_CHAT_IDS, "
                    "TELEGRAM_WEBHOOK_SECRET, and PUBLIC_BASE_URL."
                ),
            )
        try:
            return notifier.configure_webhook(
                config.public_base_url,
                config.telegram_webhook_secret,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/telegram/webhook", include_in_schema=False)
    def telegram_webhook(
        update: dict[str, Any],
        x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, bool]:
        if notifier is None or not config.telegram_webhook_secret:
            raise HTTPException(status_code=503, detail="Telegram webhook is not configured.")
        supplied = x_telegram_bot_api_secret_token or ""
        if not secrets.compare_digest(supplied, config.telegram_webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret.")

        message = update.get("message")
        if not isinstance(message, dict):
            return {"ok": True}
        chat = message.get("chat")
        if not isinstance(chat, dict) or "id" not in chat:
            return {"ok": True}
        chat_id = str(chat["id"])
        if chat_id not in config.telegram_allowed_chat_ids:
            return {"ok": True}
        text = str(message.get("text", "")).strip().split(maxsplit=1)[0].lower()
        if text in {"/start", "/help"}:
            notifier.send_text(
                chat_id,
                "🥈 <b>Silver AI Forecast</b>\n\n"
                "الأوامر:\n/latest — آخر توقع\n/status — حالة النظام\n\n"
                "⚠️ Research only. لا توجد إشارات BUY/SELL.",
            )
        elif text == "/latest":
            snapshot = orchestrator.latest()
            notifier.send_text(
                chat_id,
                notifier.format_forecast(snapshot)
                if snapshot
                else "لا يوجد Forecast حي حتى الآن.",
            )
        elif text == "/status":
            notifier.send_text(
                chat_id,
                f"✅ النظام يعمل\nBaseline: {engine.baseline_model_name}\n"
                f"Challenger: {engine.challenger_model_name}\n"
                "Edge: NOT_PROVEN\nBUY/SELL: DISABLED",
            )
        return {"ok": True}

    return app
