from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.market_source import DownloadWindow
from metal_predictor.reference_window import ParquetTimestampWindowProvider
from metal_predictor.treasury_rates_source import TreasuryDailyParYieldCurveClient


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
OUTPUT_DIR = Path("data/market")
RATES_PARQUET = OUTPUT_DIR / "UST_2Y_10Y_H15_PUBLICATION_AWARE.parquet"
RATES_CSV = OUTPUT_DIR / "UST_2Y_10Y_H15_PUBLICATION_AWARE.csv"
QUALITY_REPORT = OUTPUT_DIR / "UST_2Y_10Y_quality_report.json"
LOOKBACK_DAYS = 90


def build() -> dict[str, object]:
    silver_window = ParquetTimestampWindowProvider().get(SILVER_PATH)
    source_window = DownloadWindow(
        start_utc=pd.Timestamp(silver_window.start_utc) - pd.Timedelta(days=LOOKBACK_DAYS),
        end_utc=silver_window.end_utc,
    )
    rates, source_report = TreasuryDailyParYieldCurveClient().fetch(source_window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(RATES_PARQUET, index=False)
    rates.to_csv(RATES_CSV, index=False)
    report = {
        "status": "PASS",
        "series": ["2-Year Treasury CMT", "10-Year Treasury CMT", "10Y-2Y curve"],
        "model_window": {
            "start_utc": str(silver_window.start_utc),
            "end_utc": str(silver_window.end_utc),
        },
        "source_window_lookback_days": LOOKBACK_DAYS,
        "source": source_report.as_dict(),
        "publication_policy": {
            "normal_availability": "next Federal business day at 16:15 America/New_York via H.15",
            "timezone_dst_aware": True,
            "documented_2023_treasury_omissions_delayed": True,
            "board_closure_2025_01_09_delayed": True,
            "feature_visibility_rule": "available_from_utc <= silver feature timestamp",
            "same_observation_day_exposure": False,
        },
        "outputs": [str(RATES_PARQUET), str(RATES_CSV)],
    }
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
