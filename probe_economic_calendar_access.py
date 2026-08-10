from __future__ import annotations

import json

from metal_predictor.economic_calendar_source import TradingEconomicsCalendarClient


def run() -> dict[str, object]:
    frame, report = TradingEconomicsCalendarClient().fetch_us_calendar(
        "2024-01-01", "2024-01-31", importance=3
    )
    sample_columns = [
        column for column in (
            "release_utc", "Event", "Category", "Actual", "Forecast",
            "ActualValue", "ForecastValue", "Unit", "Ticker", "Importance",
        )
        if column in frame.columns
    ]
    sample = frame.loc[:, sample_columns].head(20).copy()
    if "release_utc" in sample.columns:
        sample["release_utc"] = sample["release_utc"].astype("string")
    return {
        "status": "PASS",
        "access": report.as_dict(),
        "columns": list(frame.columns),
        "sample": sample.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
