from __future__ import annotations

from metal_predictor.multi_horizon.contracts import DatasetState, ForecastHorizonSpec


HORIZONS: tuple[ForecastHorizonSpec, ...] = (
    ForecastHorizonSpec(
        key="4h",
        label="4 Hours",
        interval_seconds=14_400,
        route="/forecast/4h",
        dataset_state=DatasetState.READY,
    ),
    ForecastHorizonSpec(
        key="12h",
        label="12 Hours",
        interval_seconds=43_200,
        route="/forecast/12h",
        dataset_state=DatasetState.READY,
    ),
    ForecastHorizonSpec(
        key="1d",
        label="1 Day",
        interval_seconds=86_400,
        route="/forecast/1d",
        dataset_state=DatasetState.DATA_PENDING,
    ),
    ForecastHorizonSpec(
        key="2d",
        label="2 Days",
        interval_seconds=172_800,
        route="/forecast/2d",
        dataset_state=DatasetState.READY,
    ),
    ForecastHorizonSpec(
        key="30d",
        label="30 Days",
        interval_seconds=2_592_000,
        route="/forecast/30d",
        dataset_state=DatasetState.READY,
    ),
)

_BY_KEY = {spec.key: spec for spec in HORIZONS}


def get_horizon(key: str) -> ForecastHorizonSpec:
    try:
        return _BY_KEY[key.strip().lower()]
    except KeyError:
        raise KeyError(f"Unknown forecast horizon: {key!r}.") from None
