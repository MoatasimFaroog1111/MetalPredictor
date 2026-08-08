from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import databento as db

OZ_PER_KG = 32.15074656862798
SYMBOLS = {
    "XAU": "GC.v.0",
    "XAG": "SI.v.0",
    "XPT": "PL.v.0",
    "XPD": "PA.v.0",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-08-08T12:00:00Z")
    p.add_argument("--end", default="2026-08-08T12:00:00Z")
    p.add_argument("--out", default="output_databento")
    args = p.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("Set DATABENTO_API_KEY first")

    client = db.Historical(api_key)
    frames = []
    for asset, symbol in SYMBOLS.items():
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            schema="ohlcv-1h",
            symbols=[symbol],
            stype_in="continuous",
            start=args.start,
            end=args.end,
        )
        df = data.to_df().reset_index()
        if df.empty:
            continue
        ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
        df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["asset"] = asset
        df["source_symbol"] = symbol
        df["market_type"] = "futures_continuous"
        df["source_provider"] = "Databento/CME"
        df["currency"] = "USD"
        df["price_unit"] = "USD/kg"
        for c in ["open", "high", "low", "close"]:
            df[f"{c}_usd_per_oz"] = df[c]
            df[f"{c}_usd_per_kg"] = df[c] * OZ_PER_KG
        df["quality_flags"] = ""
        frames.append(df)

    if not frames:
        raise SystemExit("No data returned")

    out = pd.concat(frames, ignore_index=True)
    cols = [
        "timestamp_utc", "asset", "source_symbol", "market_type", "source_provider",
        "currency", "price_unit",
        "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg",
        "open_usd_per_oz", "high_usd_per_oz", "low_usd_per_oz", "close_usd_per_oz",
        "volume", "quality_flags"
    ]
    cols = [c for c in cols if c in out.columns]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(outdir / "cme_metals_hourly_usd_per_kg.csv", index=False)
    out[cols].to_parquet(outdir / "cme_metals_hourly_usd_per_kg.parquet", index=False)
    print(f"Saved {len(out):,} rows to {outdir}")


if __name__ == "__main__":
    main()
