from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI

from metal_predictor.forward_bars.api import create_forward_bar_research_router
from metal_predictor.forward_bars.factory import ForwardBarFactory
from metal_predictor.forward_bars.repository import SQLiteForwardBarRepository
from metal_predictor.forward_bars.scheduler import ForwardBarMaterializationScheduler
from metal_predictor.forward_bars.settings import ForwardBarSettings
from metal_predictor.forward_bars.source import SQLiteMicrostructureQuoteSampleSource


def install_forward_bar_runtime(app: FastAPI) -> FastAPI:
    live = app.state.settings
    cfg = ForwardBarSettings.from_environment()
    root = Path(live.repository_root).resolve()
    db = Path(cfg.database_path)
    if not db.is_absolute():
        db = root / db

    repository = SQLiteForwardBarRepository(db)
    factory = ForwardBarFactory(
        SQLiteMicrostructureQuoteSampleSource(
            Path(app.state.microstructure_repository.database_path)
        ),
        repository,
        security_id=live.bullionvault_security_id,
        currency=live.bullionvault_currency,
        source_cadence_seconds=live.bullionvault_microstructure_interval_seconds,
        close_delay_seconds=cfg.close_delay_seconds,
        max_buckets_per_cycle=cfg.max_buckets_per_cycle,
    )
    scheduler = (
        ForwardBarMaterializationScheduler(
            factory, interval_seconds=cfg.materialization_interval_seconds
        )
        if cfg.enabled
        else None
    )
    app.state.forward_bar_repository = repository
    app.state.forward_bar_factory = factory
    app.state.forward_bar_scheduler = scheduler
    app.include_router(
        create_forward_bar_research_router(
            repository,
            enabled=cfg.enabled,
            source_collection_enabled=live.bullionvault_microstructure_enabled,
            materialization_interval_seconds=cfg.materialization_interval_seconds,
            close_delay_seconds=cfg.close_delay_seconds,
            source_cadence_seconds=live.bullionvault_microstructure_interval_seconds,
            security_id=live.bullionvault_security_id,
            currency=live.bullionvault_currency,
        )
    )
    if scheduler is None:
        return app

    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with original(application):
            task = asyncio.create_task(
                scheduler.run_forever(), name="bullionvault-forward-bars"
            )
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = lifespan
    return app
