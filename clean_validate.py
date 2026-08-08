from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PRICE_COLS = ["open_usd_per_kg", "high_usd_per_kg", "low_usd_per_kg", "close_usd_per_kg"]


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.67448975 * (s - med) / mad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--out", default="cleaned")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["asset", "timestamp_utc"])
    df = df.drop_duplicates(["asset", "timestamp_utc"], keep="last")

    for c in PRICE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    invalid = (
        (df["high_usd_per_kg"] < df["low_usd_per_kg"]) |
        (df["low_usd_per_kg"] > df[["open_usd_per_kg", "close_usd_per_kg"]].min(axis=1)) |
        (df["high_usd_per_kg"] < df[["open_usd_per_kg", "close_usd_per_kg"]].max(axis=1)) |
        (df[PRICE_COLS] <= 0).any(axis=1)
    )

    df["gap_hours"] = df.groupby("asset")["timestamp_utc"].diff().dt.total_seconds().div(3600)
    df["unexpected_gap_before"] = df["gap_hours"] > 1.01
    df["log_close"] = np.log(df["close_usd_per_kg"])
    df["log_return_1h"] = df.groupby("asset")["log_close"].diff()
    df["robust_z_return"] = df.groupby("asset")["log_return_1h"].transform(robust_z)
    df["return_outlier"] = df["robust_z_return"].abs() > 10

    flags = pd.Series("", index=df.index, dtype="object")
    flags.loc[invalid] = "OHLC_INVARIANT_FAIL"
    flags.loc[df["unexpected_gap_before"]] = flags.loc[df["unexpected_gap_before"]].replace("", "UNEXPECTED_GAP").where(
        flags.loc[df["unexpected_gap_before"]].eq(""), flags.loc[df["unexpected_gap_before"]] + "|UNEXPECTED_GAP"
    )
    flags.loc[df["return_outlier"]] = flags.loc[df["return_outlier"]].replace("", "ROBUST_RETURN_OUTLIER").where(
        flags.loc[df["return_outlier"]].eq(""), flags.loc[df["return_outlier"]] + "|ROBUST_RETURN_OUTLIER"
    )
    df["quality_flags"] = flags

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "metals_hourly_clean.csv", index=False)
    df.to_parquet(outdir / "metals_hourly_clean.parquet", index=False)

    summary = df.groupby("asset").agg(
        rows=("timestamp_utc", "size"),
        first_timestamp=("timestamp_utc", "min"),
        last_timestamp=("timestamp_utc", "max"),
        invalid_ohlc=("quality_flags", lambda s: s.str.contains("OHLC_INVARIANT_FAIL").sum()),
        return_outliers=("return_outlier", "sum"),
        gaps=("unexpected_gap_before", "sum"),
    ).reset_index()
    summary.to_csv(outdir / "validation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
