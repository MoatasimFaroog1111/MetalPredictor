from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyarrow

from metal_predictor.local_archives import (
    ContentAddressedArchiveCatalog,
    Sha256FileFingerprinter,
)
from metal_predictor.long_history import LongHistoryH1Builder
from metal_predictor.market_aggregation import ConservativeH1Aggregator
from metal_predictor.market_source import (
    DownloadWindow,
    GenericAsciiM1Parser,
    HistDataArchiveDownloader,
    InstrumentSpec,
)
from metal_predictor.metastock_source import MetaStockM1ArchiveParser


DEFAULT_START = "2009-05-03T22:00:00Z"
DEFAULT_END = "2026-08-07T21:00:00Z"
DEFAULT_OUTPUT_DIR = Path("data/long_history")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative XAGUSD H1 long-history research dataset."
    )
    parser.add_argument(
        "--source",
        choices=("histdata-download", "metastock-local"),
        default="histdata-download",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/local_xagusd"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/histdata/xagusd-long"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-utc", default=DEFAULT_START)
    parser.add_argument("--end-utc", default=DEFAULT_END)
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, object]:
    instrument = InstrumentSpec(asset="XAG", pair="xagusd", source_symbol="XAGUSD")
    window = DownloadWindow(
        start_utc=pd.Timestamp(args.start_utc),
        end_utc=pd.Timestamp(args.end_utc),
    )
    fingerprinter = Sha256FileFingerprinter()
    catalog_report = None
    if args.source == "metastock-local":
        archives, catalog_report = ContentAddressedArchiveCatalog(
            fingerprinter=fingerprinter
        ).discover(args.input_dir)
        parser = MetaStockM1ArchiveParser(expected_symbol="XAGUSD")
    else:
        archives = HistDataArchiveDownloader().download(instrument, window, args.raw_dir)
        parser = GenericAsciiM1Parser()

    source_fingerprints = tuple(
        fingerprinter.fingerprint(path).as_dict() for path in archives
    )
    builder = LongHistoryH1Builder(parser=parser, aggregator=ConservativeH1Aggregator())
    hourly, quality = builder.build(archives, instrument, window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "XAGUSD_H1_2009_2026_USD_PER_KG.parquet"
    report_path = args.output_dir / "long_history_build_report.json"
    hourly.to_parquet(parquet_path, index=False)
    output_fingerprint = fingerprinter.fingerprint(parquet_path).as_dict()

    report: dict[str, object] = {
        "status": "PASS",
        "research_only": True,
        "future_holdout_modified": False,
        "source_mode": args.source,
        "requested_window": {
            "start_utc": pd.Timestamp(window.start_utc).isoformat(),
            "end_utc": pd.Timestamp(window.end_utc).isoformat(),
        },
        "instrument": instrument.__dict__,
        "source_policy": {
            "provider": "HistData",
            "quote_side": "Bid OHLC",
            "source_timezone": "fixed EST UTC-05:00",
            "utc_conversion": "fixed EST -> UTC; no DST guessing",
            "conflicting_duplicate_minutes": "exclude the complete affected H1 hour",
            "source_time_reversal_hours": "exclude both sides of the reversal",
            "over_60_raw_minute_hours": "exclude",
            "partial_source_hours": "retain with explicit PARTIAL_SOURCE_HOUR flag",
            "forward_fill": False,
            "interpolation": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "archive_catalog": catalog_report.as_dict() if catalog_report else None,
        "source_archive_fingerprints": list(source_fingerprints),
        "quality": quality.as_dict(),
        "output": str(parquet_path),
        "output_fingerprint": output_fingerprint,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(_arguments()), indent=2))
