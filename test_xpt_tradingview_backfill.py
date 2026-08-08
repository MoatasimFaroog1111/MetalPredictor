#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import websocket

WS_URL = "wss://data.tradingview.com/socket.io/websocket?type=chart"
SYMBOL = "OANDA:XPTUSD"
TIMEFRAME = "60"
BATCH = 5000
TARGET_START = datetime(2021, 8, 8, tzinfo=timezone.utc)
TARGET_END = datetime(2021, 11, 1, tzinfo=timezone.utc)


def frame(method: str, params: list) -> str:
    payload = json.dumps({"m": method, "p": params}, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


def split_frames(raw: str):
    parts = re.split(r"~m~\d+~m~", raw)
    for part in parts:
        if part:
            yield part


def recv_until_completed(ws, rows: dict[int, tuple]) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        raw = ws.recv()
        if not isinstance(raw, str):
            continue
        for part in split_frames(raw):
            if part.startswith("~h~"):
                ws.send(f"~m~{len(part)}~m~{part}")
                continue
            try:
                msg = json.loads(part)
            except json.JSONDecodeError:
                continue

            method = msg.get("m")
            params = msg.get("p", [])
            if method in {"critical_error", "symbol_error", "series_error", "protocol_error"}:
                raise RuntimeError(f"TradingView {method}: {params}")

            if method == "timescale_update" and len(params) >= 2 and isinstance(params[1], dict):
                container = params[1]
                series = container.get("$prices") or container.get("s1") or {}
                candles = series.get("s", []) if isinstance(series, dict) else []
                for candle in candles:
                    v = candle.get("v", []) if isinstance(candle, dict) else []
                    if len(v) >= 5:
                        ts = int(v[0])
                        rows[ts] = (
                            ts,
                            float(v[1]),
                            float(v[2]),
                            float(v[3]),
                            float(v[4]),
                            float(v[5]) if len(v) >= 6 and v[5] is not None else 0.0,
                        )

            if method == "series_completed":
                return

    raise TimeoutError("Timed out waiting for TradingView series_completed")


def fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    rows: dict[int, tuple] = {}
    cs = "cs_xptbackfill01"

    ws = websocket.create_connection(
        WS_URL,
        origin="https://www.tradingview.com",
        timeout=20,
        header=["User-Agent: Mozilla/5.0"],
    )
    try:
        ws.send(frame("set_auth_token", ["unauthorized_user_token"]))
        ws.send(frame("chart_create_session", [cs, ""]))
        symbol_config = {
            "symbol": SYMBOL,
            "adjustment": "splits",
            "session": "regular",
        }
        ws.send(frame("resolve_symbol", [cs, "ser_1", f"={json.dumps(symbol_config, separators=(',', ':'))}"]))
        ws.send(frame("create_series", [cs, "$prices", "s1", "ser_1", TIMEFRAME, BATCH, ""]))

        batch_no = 1
        previous_oldest = None
        while True:
            recv_until_completed(ws, rows)
            if not rows:
                raise RuntimeError("No candle data returned")

            oldest = min(rows)
            newest = max(rows)
            print(f"batch {batch_no:02d} | rows {len(rows):,} | {fmt(oldest)} -> {fmt(newest)}")

            if datetime.fromtimestamp(oldest, tz=timezone.utc) <= TARGET_START:
                break
            if oldest == previous_oldest:
                raise RuntimeError("Backfill stopped making progress before reaching August 2021")

            previous_oldest = oldest
            batch_no += 1
            ws.send(frame("request_more_data", [cs, "$prices", BATCH]))

        start_ts = int(TARGET_START.timestamp())
        end_ts = int(TARGET_END.timestamp())
        gap = [rows[k] for k in sorted(rows) if start_ts <= k < end_ts]

        print(f"\nGap rows: {len(gap):,}")
        if gap:
            print("First gap candle:", fmt(gap[0][0]), gap[0][1:5])
            print("Last gap candle: ", fmt(gap[-1][0]), gap[-1][1:5])

            out = Path("xpt_gap_2021_tradingview.csv")
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Timestamp_UTC", "Open_USD_toz", "High_USD_toz", "Low_USD_toz", "Close_USD_toz", "Volume", "Source"])
                for ts, o, h, l, c, vol in gap:
                    w.writerow([
                        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        o, h, l, c, vol, "TradingView OANDA:XPTUSD",
                    ])
            print("Saved:", out.resolve())
        else:
            print("No candles found inside the target gap.")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
