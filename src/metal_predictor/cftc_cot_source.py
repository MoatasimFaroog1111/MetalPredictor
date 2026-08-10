from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from metal_predictor.cftc_cot_publication import CftcCotPublicationPolicy
from metal_predictor.market_source import DownloadWindow


SILVER_CFTC_CONTRACT_MARKET_CODE = "084691"


@dataclass(frozen=True)
class CftcCotSourceReport:
    annual_archives: int
    source_rows_all_markets: int
    silver_rows: int
    first_report_date: str
    last_report_date: str
    exact_publication_overrides_used: int
    missing_required_values: int
    source: str
    contract_market_code: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CftcDisaggregatedFuturesClient:
    """Official CFTC annual Disaggregated Futures-Only history adapter."""

    _URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    _REQUIRED_SOURCE_COLUMNS = (
        "Market_and_Exchange_Names",
        "As_of_Date_Form_YYYY-MM-DD",
        "CFTC_Contract_Market_Code",
        "Open_Interest_All",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "Swap_Positions_Long_All",
        "Swap__Positions_Short_All",
        "Swap__Positions_Spread_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
        "M_Money_Positions_Spread_All",
        "Other_Rept_Positions_Long_All",
        "Other_Rept_Positions_Short_All",
        "Other_Rept_Positions_Spread_All",
        "NonRept_Positions_Long_All",
        "NonRept_Positions_Short_All",
    )

    def __init__(self, publication_policy: CftcCotPublicationPolicy | None = None) -> None:
        self._publication = publication_policy or CftcCotPublicationPolicy()

    def fetch(self, window: DownloadWindow) -> tuple[pd.DataFrame, CftcCotSourceReport]:
        start = pd.Timestamp(window.start_utc).tz_convert("UTC").date()
        end = pd.Timestamp(window.end_utc).tz_convert("UTC").date()
        frames: list[pd.DataFrame] = []
        source_rows = 0
        for year in range(start.year, end.year + 1):
            yearly = self._fetch_year(year)
            source_rows += len(yearly)
            frames.append(yearly)

        raw = pd.concat(frames, ignore_index=True)
        code = raw["CFTC_Contract_Market_Code"].astype("string").str.strip().str.zfill(6)
        silver = raw.loc[code.eq(SILVER_CFTC_CONTRACT_MARKET_CODE)].copy()
        if silver.empty:
            raise ValueError("CFTC history contains no COMEX Silver rows for code 084691.")

        silver["report_date"] = pd.to_datetime(
            silver["As_of_Date_Form_YYYY-MM-DD"], errors="raise"
        ).dt.normalize()
        silver = silver.loc[
            silver["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        ].copy()
        silver = (
            silver.sort_values("report_date")
            .drop_duplicates("report_date", keep="last")
            .reset_index(drop=True)
        )
        if silver.empty:
            raise ValueError("No CFTC Silver rows remain inside the requested source window.")

        numeric_source = [
            column for column in self._REQUIRED_SOURCE_COLUMNS
            if column not in {
                "Market_and_Exchange_Names",
                "As_of_Date_Form_YYYY-MM-DD",
                "CFTC_Contract_Market_Code",
            }
        ]
        for column in numeric_source:
            silver[column] = pd.to_numeric(silver[column], errors="coerce")
        missing_required_values = int(silver[numeric_source].isna().sum().sum())
        if missing_required_values:
            raise ValueError(
                f"CFTC Silver rows contain {missing_required_values} missing required positioning values."
            )
        if not np.isfinite(silver[numeric_source].to_numpy(float)).all():
            raise ValueError("CFTC Silver positioning data contains infinite values.")
        if (silver["Open_Interest_All"] <= 0).any():
            raise ValueError("CFTC Silver Open Interest must be positive.")

        silver["available_from_utc"] = self._publication.available_from_utc(
            silver["report_date"]
        )
        if silver["available_from_utc"].duplicated().any():
            raise ValueError("CFTC Silver publication timestamps must be unique.")
        if not silver["available_from_utc"].is_monotonic_increasing:
            raise ValueError("CFTC Silver publication timestamps must be chronological.")

        result = pd.DataFrame({
            "report_date": silver["report_date"],
            "available_from_utc": silver["available_from_utc"],
            "market_name": silver["Market_and_Exchange_Names"].astype("string").str.strip(),
            "cftc_contract_market_code": SILVER_CFTC_CONTRACT_MARKET_CODE,
            "open_interest": silver["Open_Interest_All"].astype(float),
            "producer_long": silver["Prod_Merc_Positions_Long_All"].astype(float),
            "producer_short": silver["Prod_Merc_Positions_Short_All"].astype(float),
            "swap_long": silver["Swap_Positions_Long_All"].astype(float),
            "swap_short": silver["Swap__Positions_Short_All"].astype(float),
            "swap_spread": silver["Swap__Positions_Spread_All"].astype(float),
            "managed_long": silver["M_Money_Positions_Long_All"].astype(float),
            "managed_short": silver["M_Money_Positions_Short_All"].astype(float),
            "managed_spread": silver["M_Money_Positions_Spread_All"].astype(float),
            "other_long": silver["Other_Rept_Positions_Long_All"].astype(float),
            "other_short": silver["Other_Rept_Positions_Short_All"].astype(float),
            "other_spread": silver["Other_Rept_Positions_Spread_All"].astype(float),
            "nonreportable_long": silver["NonRept_Positions_Long_All"].astype(float),
            "nonreportable_short": silver["NonRept_Positions_Short_All"].astype(float),
        })
        overrides_used = int(
            result["report_date"].dt.date.isin(
                self._publication.exact_override_report_dates
            ).sum()
        )
        report = CftcCotSourceReport(
            annual_archives=end.year - start.year + 1,
            source_rows_all_markets=int(source_rows),
            silver_rows=int(len(result)),
            first_report_date=str(result["report_date"].iloc[0].date()),
            last_report_date=str(result["report_date"].iloc[-1].date()),
            exact_publication_overrides_used=overrides_used,
            missing_required_values=missing_required_values,
            source="CFTC Historical Compressed Disaggregated Futures Only",
            contract_market_code=SILVER_CFTC_CONTRACT_MARKET_CODE,
        )
        return result, report

    def _fetch_year(self, year: int) -> pd.DataFrame:
        url = self._URL.format(year=year)
        request = Request(url, headers={"User-Agent": "MetalPredictor research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        if payload[:4] != b"PK\x03\x04":
            raise ValueError(f"CFTC {year} response is not a ZIP archive.")
        with ZipFile(BytesIO(payload)) as archive:
            candidates = [
                name for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith((".txt", ".csv"))
            ]
            if not candidates:
                raise ValueError(f"CFTC {year} archive contains no text data file.")
            member = sorted(candidates, key=lambda name: ("disagg" not in name.lower(), len(name)))[0]
            with archive.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    dtype={"CFTC_Contract_Market_Code": "string"},
                    low_memory=False,
                    skipinitialspace=True,
                )
        frame.columns = [str(column).strip() for column in frame.columns]
        missing = set(self._REQUIRED_SOURCE_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(
                f"CFTC {year} schema missing required columns: {sorted(missing)}; "
                f"available={list(frame.columns)}"
            )
        return frame
