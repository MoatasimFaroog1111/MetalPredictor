from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.market_source import DownloadWindow
from metal_predictor.reference_window import ParquetTimestampWindowProvider
from metal_predictor.vix_source import CboeVixDailyHistoryClient


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
OUTPUT_DIR = Path("data/market")
VIX_PARQUET = OUTPUT_DIR / "VIX_DAILY_PUBLICATION_AWARE.parquet"
VIX_CSV = OUTPUT_DIR / "VIX_DAILY_PUBLICATION_AWARE.csv"
QUALITY_REPORT = OUTPUT_DIR / "VIX_DAILY_quality_report.json"
LOOKBACK_DAYS = 120


def build() -> dict[str, object]:
    silver_window = ParquetTimestampWindowProvider().get(SILVER_PATH)
    source_window = DownloadWindow(
        start_utc=pd.Timestamp(silver_window.start_utc) - pd.Timedelta(days=LOOKBACK_DAYS),
        end_utc=silver_window.end_utc,
    )
    vix, source_report = CboeVixDailyHistoryClient().fetch(source_window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vix.to_parquet(VIX_PARQUET, index=False)
    vix.to_csv(VIX_CSV, index=False)
    report = {
        "status": "PASS",
        "series": "Cboe VIX Index daily OHLC",
        "model_window": {
            "start_utc": str(silver_window.start_utc),
            "end_utc": str(silver_window.end_utc),
        },
        "source_window_lookback_days": LOOKBACK_DAYS,
        "source": source_report.as_dict(),
        "publication_policy": {
            "availability": "same trading date at 16:15 America/New_York",
            "timezone_dst_aware": True,
            "feature_visibility_rule": "available_from_utc <= completed H1 bar decision time",
            "intraday_daily_close_backfill": False,
        },
        "outputs": [str(VIX_PARQUET), str(VIX_CSV)],
    }
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
