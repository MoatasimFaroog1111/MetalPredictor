from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import (
    DownloadWindow,
    GenericAsciiM1Parser,
    HistDataArchiveDownloader,
    InstrumentSpec,
)


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
RAW_DIR = Path("data/raw/histdata/xauusd")
OUTPUT_DIR = Path("data/market")
GOLD_PARQUET = OUTPUT_DIR / "XAUUSD_H1_USD_PER_KG.parquet"
GOLD_CSV = OUTPUT_DIR / "XAUUSD_H1_USD_PER_KG.csv"
QUALITY_REPORT = OUTPUT_DIR / "XAUUSD_H1_quality_report.json"


def silver_window(path: Path = SILVER_PATH) -> DownloadWindow:
    timestamps = pd.read_parquet(path, columns=["timestamp_utc"])["timestamp_utc"]
    timestamps = pd.to_datetime(timestamps, utc=True, errors="raise")
    return DownloadWindow(start_utc=timestamps.min(), end_utc=timestamps.max())


def build() -> dict[str, object]:
    window = silver_window()
    instrument = InstrumentSpec(asset="XAU", pair="xauusd", source_symbol="XAUUSD")
    downloader = HistDataArchiveDownloader()
    parser = GenericAsciiM1Parser()
    aggregator = ConservativeH1Aggregator()

    archives = downloader.download(instrument, window, RAW_DIR)
    minutes = parser.parse(archives)
    gold, quality = aggregator.aggregate(minutes, instrument, window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_PARQUET, index=False)
    gold.to_csv(GOLD_CSV, index=False)
    report = {
        "status": "PASS",
        "instrument": instrument.__dict__,
        "requested_window": {
            "start_utc": pd.Timestamp(window.start_utc).isoformat(),
            "end_utc": pd.Timestamp(window.end_utc).isoformat(),
        },
        "source_policy": {
            "format": "HistData Generic ASCII M1",
            "quote_side": "Bid OHLC",
            "source_timezone": "fixed EST UTC-05:00",
            "utc_conversion": "timestamp_utc = source_timestamp + 5 hours",
            "forward_fill": False,
            "conflicting_or_ambiguous_source_hours": "excluded",
        },
        "archives": [path.name for path in archives],
        "quality": quality.as_dict(),
        "outputs": [str(GOLD_PARQUET), str(GOLD_CSV)],
    }
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
