#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import xlsxwriter

BASE_URL = "https://freeserv.dukascopy.com/2.0/index.php"
SOURCE_URL = "https://freeserv.dukascopy.com/2.0/index.php?path=chart/json3"
TOZ_PER_KG = 32.1507465686
PAGE_LIMIT = 5000
DEFAULT_START = "2021-08-08T00:00:00Z"
DEFAULT_END = "2026-08-08T09:25:00Z"

METALS = {
    "Gold": {"instrument": "XAU/USD", "sheet": "Gold_H1"},
    "Silver": {"instrument": "XAG/USD", "sheet": "Silver_H1"},
    "Platinum": {"instrument": "XPT.CMD/USD", "sheet": "Platinum_H1"},
    "Palladium": {"instrument": "XPD.CMD/USD", "sheet": "Palladium_H1"},
}

Row = Tuple[int, float, float, float, float, float]


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fmt_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def callback_name() -> str:
    chars = string.ascii_letters + string.digits
    return "_callbacks____" + "".join(random.choices(chars, k=9))


def fetch_page(session: requests.Session, instrument: str, cursor_ms: int) -> List[list]:
    callback = callback_name()
    params = {
        "path": "chart/json3",
        "splits": "true",
        "stocks": "true",
        "time_direction": "N",
        "jsonp": callback,
        "last_update": str(cursor_ms),
        "offer_side": "B",
        "instrument": instrument,
        "interval": "1HOUR",
        "limit": str(PAGE_LIMIT),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/151 Safari/537.36",
        "Referer": "https://freeserv.dukascopy.com/2.0/?path=chart/index&presentationType=candle&timezone=0",
        "Accept": "*/*",
    }

    last_error = None
    for attempt in range(6):
        try:
            response = session.get(BASE_URL, params=params, headers=headers, timeout=(15, 90))
            response.raise_for_status()
            text = response.text.strip()
            prefix = callback + "("
            if prefix in text:
                body = text[text.find(prefix) + len(prefix):]
                if body.endswith(");"):
                    body = body[:-2]
                elif body.endswith(")"):
                    body = body[:-1]
                payload = json.loads(body)
            else:
                payload = response.json()

            if isinstance(payload, dict):
                for key in ("data", "candles", "values"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected response type: {type(payload).__name__}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Dukascopy request failed: {last_error}")


def normalize(raw: list) -> Row | None:
    if not isinstance(raw, list) or len(raw) < 5:
        return None
    try:
        return (
            int(raw[0]),
            float(raw[1]),
            float(raw[2]),
            float(raw[3]),
            float(raw[4]),
            float(raw[5]) if len(raw) > 5 and raw[5] is not None else 0.0,
        )
    except (TypeError, ValueError):
        return None


def fetch_history(instrument: str, start_ms: int, end_ms: int) -> Tuple[List[Row], int]:
    rows_by_ts: Dict[int, Row] = {}
    duplicates = 0
    cursor = start_ms
    previous_max = None

    with requests.Session() as session:
        for page_no in range(1, 200):
            raw_page = fetch_page(session, instrument, cursor)
            rows = [normalize(x) for x in raw_page]
            rows = [x for x in rows if x is not None]
            if not rows:
                break
            rows.sort(key=lambda x: x[0])
            max_ts = rows[-1][0]

            for row in rows:
                ts = row[0]
                if ts < start_ms:
                    continue
                if ts > end_ms:
                    return sorted(rows_by_ts.values()), duplicates
                if ts in rows_by_ts:
                    duplicates += 1
                else:
                    rows_by_ts[ts] = row

            print(f"  page {page_no:02d} | rows {len(rows_by_ts):,} | through {fmt_utc(max_ts)} UTC")
            if max_ts >= end_ms:
                break
            if previous_max is not None and max_ts <= previous_max:
                break
            previous_max = max_ts
            cursor = max_ts
            time.sleep(0.15)

    return sorted(rows_by_ts.values()), duplicates


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "DateTime UTC", "Open USD/oz", "High USD/oz", "Low USD/oz", "Close USD/oz", "Volume",
        "Open USD/kg", "High USD/kg", "Low USD/kg", "Close USD/kg", "Source",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ts, op, hi, lo, cl, vol in rows:
            writer.writerow([
                fmt_utc(ts), op, hi, lo, cl, vol,
                op * TOZ_PER_KG, hi * TOZ_PER_KG, lo * TOZ_PER_KG, cl * TOZ_PER_KG,
                "Dukascopy",
            ])


def coverage(rows: List[Row], start_ms: int, end_ms: int):
    if not rows:
        return "", "", "", "", "No data"
    first_ms, last_ms = rows[0][0], rows[-1][0]
    start_gap = max(0.0, (first_ms - start_ms) / 3_600_000)
    end_gap = max(0.0, (end_ms - last_ms) / 3_600_000)
    status = "Full available market coverage" if start_gap <= 72 and end_gap <= 72 else "Partial source coverage"
    return fmt_utc(first_ms), fmt_utc(last_ms), start_gap, end_gap, status


def write_xlsx(path: Path, all_rows: Dict[str, List[Row]], dupes: Dict[str, int], start_ms: int, end_ms: int) -> None:
    wb = xlsxwriter.Workbook(path)
    title = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D", "font_size": 16, "align": "center"})
    header = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#2F75B5", "align": "center", "border": 1})
    cell = wb.add_format({"border": 1})
    num = wb.add_format({"num_format": "#,##0.0000", "border": 1})
    wrap = wb.add_format({"border": 1, "text_wrap": True})
    ok = wb.add_format({"bg_color": "#E2F0D9", "border": 1})
    warn = wb.add_format({"bg_color": "#FFF2CC", "border": 1})

    ws = wb.add_worksheet("README")
    ws.merge_range("A1:H2", "Precious Metals H1 — USD/kg", title)
    info = [
        ("Period", f"{fmt_utc(start_ms)} UTC -> {fmt_utc(end_ms)} UTC"),
        ("Metals", "Gold, Silver, Platinum, Palladium"),
        ("Timeframe", "1HOUR"),
        ("Offer side", "BID"),
        ("Conversion", f"USD/kg = USD/troy oz x {TOZ_PER_KG}"),
        ("Integrity", "No interpolation, forward-fill, or synthetic prices."),
        ("Source", SOURCE_URL),
    ]
    ws.write_row(3, 0, ["Field", "Value"], header)
    for r, row in enumerate(info, 4):
        ws.write_row(r, 0, row, wrap)
    ws.set_column("A:A", 22)
    ws.set_column("B:B", 90)

    headers = [
        "DateTime UTC", "Open USD/oz", "High USD/oz", "Low USD/oz", "Close USD/oz", "Volume",
        "Open USD/kg", "High USD/kg", "Low USD/kg", "Close USD/kg", "Source",
    ]

    for metal, meta in METALS.items():
        rows = all_rows[metal]
        ws = wb.add_worksheet(meta["sheet"])
        ws.write_row(0, 0, headers, header)
        for r, (ts, op, hi, lo, cl, vol) in enumerate(rows, 1):
            values = [fmt_utc(ts), op, hi, lo, cl, vol, op * TOZ_PER_KG, hi * TOZ_PER_KG, lo * TOZ_PER_KG, cl * TOZ_PER_KG, "Dukascopy"]
            ws.write(r, 0, values[0], cell)
            for c in range(1, 10):
                ws.write_number(r, c, float(values[c]), num)
            ws.write(r, 10, values[10], cell)
        ws.set_column("A:A", 21)
        ws.set_column("B:J", 16)
        ws.set_column("K:K", 18)
        ws.freeze_panes(1, 0)
        if rows:
            ws.autofilter(0, 0, len(rows), 10)

    combined: Dict[int, Dict[str, float]] = defaultdict(dict)
    for metal, rows in all_rows.items():
        for ts, _op, _hi, _lo, cl, _vol in rows:
            combined[ts][metal] = cl * TOZ_PER_KG

    ws = wb.add_worksheet("Combined_Close")
    ws.write_row(0, 0, ["DateTime UTC", "Gold Close USD/kg", "Silver Close USD/kg", "Platinum Close USD/kg", "Palladium Close USD/kg"], header)
    for r, ts in enumerate(sorted(combined), 1):
        ws.write(r, 0, fmt_utc(ts), cell)
        for c, metal in enumerate(("Gold", "Silver", "Platinum", "Palladium"), 1):
            value = combined[ts].get(metal)
            if value is None:
                ws.write_blank(r, c, None, num)
            else:
                ws.write_number(r, c, value, num)
    ws.set_column("A:A", 21)
    ws.set_column("B:E", 22)
    ws.freeze_panes(1, 0)

    ws = wb.add_worksheet("Quality_Check")
    qh = ["Metal", "Instrument", "Requested Start UTC", "Requested End UTC", "Rows", "First Timestamp", "Last Timestamp", "Duplicates", "Start Gap Hours", "End Gap Hours", "Status", "Notes"]
    ws.write_row(0, 0, qh, header)
    for r, (metal, meta) in enumerate(METALS.items(), 1):
        rows = all_rows[metal]
        first, last, sg, eg, status = coverage(rows, start_ms, end_ms)
        vals = [metal, meta["instrument"], fmt_utc(start_ms), fmt_utc(end_ms), len(rows), first, last, dupes[metal], sg, eg, status, "Observed source timestamps only; no gap filling applied."]
        for c, value in enumerate(vals):
            fmt = ok if c == 10 and status.startswith("Full") else warn if c == 10 else cell
            if c in (4, 7, 8, 9) and isinstance(value, (int, float)):
                ws.write_number(r, c, value, num)
            else:
                ws.write(r, c, value, fmt)
    ws.set_column("A:B", 20)
    ws.set_column("C:D", 23)
    ws.set_column("E:J", 18)
    ws.set_column("K:K", 30)
    ws.set_column("L:L", 58)
    ws.freeze_panes(1, 0)

    wb.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    start_dt = parse_utc(args.start)
    end_dt = parse_utc(args.end)
    if end_dt <= start_dt:
        raise SystemExit("End must be after start")

    start_ms, end_ms = to_ms(start_dt), to_ms(end_dt)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: Dict[str, List[Row]] = {}
    duplicates: Dict[str, int] = {}

    print(f"Period: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"Conversion: 1 kg = {TOZ_PER_KG} troy oz")

    for metal, meta in METALS.items():
        print(f"\nDownloading {metal} ({meta['instrument']})")
        try:
            rows, dups = fetch_history(meta["instrument"], start_ms, end_ms)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows, dups = [], 0
        all_rows[metal] = rows
        duplicates[metal] = dups
        csv_path = out / "csv" / f"{meta['sheet']}_USDkg.csv"
        write_csv(csv_path, rows)
        print(f"  saved {len(rows):,} rows to {csv_path}")

    xlsx_path = out / f"Precious_Metals_H1_USDkg_{start_dt:%Y-%m-%d}_to_{end_dt:%Y-%m-%d}.xlsx"
    write_xlsx(xlsx_path, all_rows, duplicates, start_ms, end_ms)
    print(f"\nDONE: {xlsx_path.resolve()}")
    print("Review the Quality_Check sheet before using the data.")


if __name__ == "__main__":
    main()
