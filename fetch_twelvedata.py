from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

OZ_PER_KG = 32.15074656862798
BASE_URL = "https://api.twelvedata.com/time_series"
SYMBOLS = {
    "XAU": "XAU/USD",
    "XAG": "XAG/USD",
    "XPT": "XPT/USD",
    "XPD": "XPD/USD",
}


def month_windows(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur < end:
        nxt = min(cur + relativedelta(months=1), end)
        yield cur, nxt
        cur = nxt


def fetch_window(symbol: str, start: pd.Timestamp, end: pd.Timestamp, api_key: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": "1h",
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC",
        "order": "asc",
        "format": "JSON",
        "outputsize": 5000,
        "apikey": api_key,
    }
    for attempt in range(6):
        r = requests.get(BASE_URL, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") == "error":
            raise RuntimeError(json.dumps(payload, ensure_ascii=False))
        return pd.DataFrame(payload.get("values", []))
    raise RuntimeError("Rate limit retries exhausted")


def normalize(df: pd.DataFrame, asset: str, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out.pop("datetime"), utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["asset"] = asset
    out["source_symbol"] = symbol
    out["market_type"] = "spot_aggregate"
    out["source_provider"] = "Twelve Data"
    out["currency"] = "USD"
    out["price_unit"] = "USD/kg"

    for c in ["open", "high", "low", "close"]:
        out[f"{c}_usd_per_oz"] = out[c]
        out[f"{c}_usd_per_kg"] = out[c] * OZ_PER_KG

    keep = [
        "timestamp_utc", "asset", "source_symbol", "market_type", "source_provider",
        "currency", "price_unit",
        "open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg",
        "open_usd_per_oz", "high_usd_per_oz", "low_usd_per_oz", "close_usd_per_oz",
    ]
    if "volume" in out.columns:
        keep.append("volume")
    out["quality_flags"] = ""
    keep.append("quality_flags")
    return out[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-08-08T12:00:00Z")
    p.add_argument("--end", default="2026-08-08T12:00:00Z")
    p.add_argument("--out", default="output_twelvedata")
    args = p.parse_args()

    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise SystemExit("Set TWELVEDATA_API_KEY first")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    outdir = Path(args.out)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for asset, symbol in SYMBOLS.items():
        for s, e in month_windows(start, end):
            raw = fetch_window(symbol, s, e, api_key)
            raw_path = rawdir / f"{asset}_{s:%Y-%m}.json"
            raw_path.write_text(raw.to_json(orient="records", force_ascii=False), encoding="utf-8")
            norm = normalize(raw, asset, symbol)
            if not norm.empty:
                all_parts.append(norm)
            time.sleep(0.25)

    if not all_parts:
        raise SystemExit("No data returned")

    df = pd.concat(all_parts, ignore_index=True)
    df = df.sort_values(["asset", "timestamp_utc"]).drop_duplicates(["asset", "timestamp_utc"], keep="last")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "metals_hourly_usd_per_kg.csv", index=False)
    df.to_json(outdir / "metals_hourly_usd_per_kg.json", orient="records", date_format="iso")
    df.to_parquet(outdir / "metals_hourly_usd_per_kg.parquet", index=False)
    print(f"Saved {len(df):,} rows to {outdir}")


if __name__ == "__main__":
    main()
