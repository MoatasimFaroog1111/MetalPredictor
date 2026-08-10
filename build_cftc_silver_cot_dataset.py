from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metal_predictor.cftc_cot_source import CftcDisaggregatedFuturesClient
from metal_predictor.market_source import DownloadWindow
from metal_predictor.reference_window import ParquetTimestampWindowProvider


SILVER_PATH = Path("XAGUSD_H1_5Y_USD_PER_KG_CLEAN.parquet")
OUTPUT_DIR = Path("data/market")
COT_PARQUET = OUTPUT_DIR / "CFTC_SILVER_084691_COT_PUBLICATION_AWARE.parquet"
COT_CSV = OUTPUT_DIR / "CFTC_SILVER_084691_COT_PUBLICATION_AWARE.csv"
QUALITY_REPORT = OUTPUT_DIR / "CFTC_SILVER_084691_quality_report.json"
LOOKBACK_DAYS = 450


def build() -> dict[str, object]:
    silver_window = ParquetTimestampWindowProvider().get(SILVER_PATH)
    source_window = DownloadWindow(
        start_utc=pd.Timestamp(silver_window.start_utc) - pd.Timedelta(days=LOOKBACK_DAYS),
        end_utc=silver_window.end_utc,
    )
    cot, source_report = CftcDisaggregatedFuturesClient().fetch(source_window)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cot.to_parquet(COT_PARQUET, index=False)
    cot.to_csv(COT_CSV, index=False)
    report = {
        "status": "PASS",
        "market": "COMEX Silver",
        "cftc_contract_market_code": "084691",
        "model_window": {
            "start_utc": str(silver_window.start_utc),
            "end_utc": str(silver_window.end_utc),
        },
        "source_window_lookback_days": LOOKBACK_DAYS,
        "source": source_report.as_dict(),
        "publication_policy": {
            "normal_release": "Friday 15:30 America/New_York",
            "normal_report_state": "usually preceding Tuesday",
            "timezone_dst_aware": True,
            "documented_disruptions": [
                "2021 Juneteenth delay",
                "2023 ION cyber incident catch-up",
                "2025 National Day of Mourning",
                "2025 appropriations-lapse catch-up",
                "known delayed 2026 schedule dates inside model window",
            ],
            "historical_schedule_limit": (
                "CFTC does not publish a complete historical release-date list. "
                "For otherwise normal holiday weeks, the adapter uses a conservative "
                "post-Friday Federal business-day availability rule to avoid look-ahead."
            ),
            "feature_visibility_rule": (
                "available_from_utc <= completed H1 bar decision time"
            ),
            "backdate_to_report_date": False,
        },
        "outputs": [str(COT_PARQUET), str(COT_CSV)],
    }
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
