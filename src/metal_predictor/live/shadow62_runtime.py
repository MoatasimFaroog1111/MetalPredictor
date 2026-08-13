from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from metal_predictor.precious_metals.dukascopy_public_source import (
    DukascopyPublicHistoricalMetalSource,
)
from metal_predictor.shadow62.api import create_shadow62_research_router
from metal_predictor.shadow62.engine import Shadow62InferenceEngine
from metal_predictor.shadow62.repository import SQLiteShadowRepository
from metal_predictor.shadow62.scheduler import Shadow62Scheduler
from metal_predictor.shadow62.service import Shadow62Service


@dataclass(frozen=True)
class Shadow62Runtime:
    repository: SQLiteShadowRepository
    engine: Shadow62InferenceEngine | None
    service: Shadow62Service | None
    scheduler: Shadow62Scheduler | None


def build_shadow62_runtime(app: FastAPI) -> Shadow62Runtime:
    """Compose the shadow runtime from already-created live app dependencies.

    This adapter depends only on app.state.settings and app.state.repository. It does
    not alter LivePredictionEngine, LiveForecastOrchestrator, or their forecast routes.
    """

    config = app.state.settings
    root = Path(config.repository_root).resolve()
    db_path = Path(config.shadow62_database_path)
    if not db_path.is_absolute():
        db_path = root / db_path
    repository = SQLiteShadowRepository(db_path)

    if not config.shadow62_enabled:
        return Shadow62Runtime(repository, None, None, None)

    engine = Shadow62InferenceEngine(root)
    source = DukascopyPublicHistoricalMetalSource()
    service = Shadow62Service(app.state.repository, repository, engine, source)
    scheduler = Shadow62Scheduler(service, delay_minutes=config.shadow62_delay_minutes)
    return Shadow62Runtime(repository, engine, service, scheduler)


def install_shadow62_runtime(app: FastAPI) -> FastAPI:
    """Attach research-only routes and a parallel lifespan task to a live app."""

    runtime = build_shadow62_runtime(app)
    config = app.state.settings
    app.state.shadow62_repository = runtime.repository
    app.state.shadow62_engine = runtime.engine
    app.state.shadow62_service = runtime.service
    app.state.shadow62_scheduler = runtime.scheduler
    app.include_router(
        create_shadow62_research_router(
            runtime.repository,
            runtime.service,
            enabled=config.shadow62_enabled,
            delay_minutes=config.shadow62_delay_minutes,
        )
    )

    if runtime.scheduler is None:
        return app

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(application: FastAPI):
        async with original_lifespan(application):
            task = asyncio.create_task(
                runtime.scheduler.run_forever(),
                name="xpt-xpd-shadow62-research",
            )
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = combined_lifespan
    return app
