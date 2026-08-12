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
RAW_ROOT = Path("data/raw/histdata")
OUTPUT_DIR = Path("data/market")

INSTRUMENTS = (
    InstrumentSpec(asset="XPT", pair="xptusd", source_symbol="XPTUSD"),
    InstrumentSpec(asset="XPD", pair="xpdusd", source_symbol="XPDUSD"),
)


def silver_window(path: Path = SILVER_PATH) -> DownloadWindow:
    timestamps = pd.read_parquet(path, columns=["timestamp_utc"])["timestamp_utc"]
    timestamps = pd.to_datetime(timestamps, utc=True, errors="raise")
    return DownloadWindow(start_utc=timestamps.min(), end_utc=timestamps.max())


def build() -> dict[str, object]:
    window = silver_window()
    downloader = HistDataArchiveDownloader()
    parser = GenericAsciiM1Parser()
    aggregator = ConservativeH1Aggregator()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instruments: list[dict[str, object]] = []
    for instrument in INSTRUMENTS:
        raw_dir = RAW_ROOT / instrument.pair.lower()
        archives = downloader.download(instrument, window, raw_dir)
        minutes = parser.parse(archives)
        hourly, quality = aggregator.aggregate(minutes, instrument, window)

        parquet_path = OUTPUT_DIR / f"{instrument.source_symbol}_H1_USD_PER_KG.parquet"
        csv_path = OUTPUT_DIR / f"{instrument.source_symbol}_H1_USD_PER_KG.csv"
        quality_path = OUTPUT_DIR / f"{instrument.source_symbol}_H1_quality_report.json"
        hourly.to_parquet(parquet_path, index=False)
        hourly.to_csv(csv_path, index=False)

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
                "interpolation": False,
                "conflicting_or_ambiguous_source_hours": "excluded",
            },
            "archives": [path.name for path in archives],
            "quality": quality.as_dict(),
            "outputs": [str(parquet_path), str(csv_path)],
        }
        quality_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        instruments.append(report)

    combined = {
        "status": "PASS",
        "research_only": True,
        "model_mutated": False,
        "frozen_feature_graph_mutated": False,
        "instruments": instruments,
    }
    (OUTPUT_DIR / "XPT_XPD_H1_build_report.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    return combined


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
