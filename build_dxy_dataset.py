from __future__ import annotations

import json
from pathlib import Path

from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import GenericAsciiM1Parser, HistDataArchiveDownloader, InstrumentSpec
from metal_predictor.price_normalization import IdentityIndexNormalizer
from metal_predictor.reference_window import ParquetTimestampWindowProvider


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
RAW_DIR = Path("data/raw/histdata/udxusd")
OUTPUT_DIR = Path("data/market")
DXY_PARQUET = OUTPUT_DIR / "UDXUSD_H1_INDEX.parquet"
DXY_CSV = OUTPUT_DIR / "UDXUSD_H1_INDEX.csv"
QUALITY_REPORT = OUTPUT_DIR / "UDXUSD_H1_quality_report.json"


def build() -> dict[str, object]:
    window = ParquetTimestampWindowProvider().get(SILVER_PATH)
    instrument = InstrumentSpec(asset="DXY", pair="udxusd", source_symbol="UDXUSD")
    downloader = HistDataArchiveDownloader()
    parser = GenericAsciiM1Parser()
    aggregator = ConservativeH1Aggregator(IdentityIndexNormalizer())

    archives = downloader.download(instrument, window, RAW_DIR)
    minutes = parser.parse(archives)
    dxy, quality = aggregator.aggregate(minutes, instrument, window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dxy.to_parquet(DXY_PARQUET, index=False)
    dxy.to_csv(DXY_CSV, index=False)
    report = {
        "status": "PASS",
        "instrument": instrument.__dict__,
        "requested_window": {
            "start_utc": str(window.start_utc),
            "end_utc": str(window.end_utc),
        },
        "source_policy": {
            "format": "HistData Generic ASCII M1",
            "quote_side": "Bid OHLC",
            "source_timezone": "fixed EST UTC-05:00",
            "utc_conversion": "timestamp_utc = source_timestamp + 5 hours",
            "value_unit": "index_points",
            "forward_fill": False,
            "conflicting_or_ambiguous_source_hours": "excluded",
        },
        "archives": [path.name for path in archives],
        "quality": quality.as_dict(),
        "outputs": [str(DXY_PARQUET), str(DXY_CSV)],
    }
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
