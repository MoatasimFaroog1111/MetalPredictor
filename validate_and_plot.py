from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(series: pd.Series):
    x = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50:
        return None
    adf = adfuller(x, autolag="AIC")
    kp = kpss(x, regression="c", nlags="auto")
    return {
        "adf_stat": adf[0], "adf_p": adf[1],
        "kpss_stat": kp[0], "kpss_p": kp[1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    stat_rows = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("timestamp_utc").copy()
        g["log_close"] = np.log(g["close_usd_per_kg"])
        g["log_return_1h"] = g["log_close"].diff()

        for name, s in [("log_close", g["log_close"]), ("log_return_1h", g["log_return_1h"])]:
            res = stationarity(s)
            if res:
                stat_rows.append({"asset": asset, "series": name, **res})

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(g["timestamp_utc"], g["close_usd_per_kg"], linewidth=0.7)
        ax.set_title(f"{asset} hourly close — USD/kg")
        ax.set_xlabel("UTC")
        ax.set_ylabel("USD/kg")
        fig.tight_layout()
        fig.savefig(outdir / f"{asset}_timeseries.png", dpi=140)
        plt.close(fig)

        r = g["log_return_1h"].dropna()
        if len(r) > 200:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(r, lags=min(168, len(r)//4), zero=False, ax=ax)
            ax.set_title(f"{asset} ACF — hourly log returns")
            fig.tight_layout()
            fig.savefig(outdir / f"{asset}_acf_returns.png", dpi=140)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(r**2, lags=min(168, len(r)//4), zero=False, ax=ax)
            ax.set_title(f"{asset} ACF — squared returns")
            fig.tight_layout()
            fig.savefig(outdir / f"{asset}_acf_squared_returns.png", dpi=140)
            plt.close(fig)

    pd.DataFrame(stat_rows).to_csv(outdir / "stationarity_adf_kpss.csv", index=False)
    print(f"Artifacts saved to {outdir}")


if __name__ == "__main__":
    main()
