from __future__ import annotations

from dataclasses import dataclass
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


@dataclass(frozen=True)
class CmeFilePayload:
    url: str
    content: bytes
    content_type: str | None


class CmePublicFileClient:
    """Reads public CME archive files through official HTTP archive paths.

    CME currently exposes equivalent daily-volume archive listings through both a
    nested FTP-style web path and a shorter alias. The client treats them as
    ordered official mirrors, maintains one cookie session, uses browser-compatible
    request headers, and accepts a workbook only after validating the OOXML ZIP
    signature. It does not bypass authentication or protected DataMine endpoints.
    """

    _BASE_URLS = (
        "https://www.cmegroup.com/ftp/pub/pub/pub/daily_volume/",
        "https://www.cmegroup.com/ftp/daily_volume/",
    )
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/151.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "application/octet-stream;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    def __init__(self) -> None:
        self._opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def fetch_daily_volume_workbook(self, trading_date_yyyymmdd: str) -> CmeFilePayload:
        if len(trading_date_yyyymmdd) != 8 or not trading_date_yyyymmdd.isdigit():
            raise ValueError("trading_date_yyyymmdd must be YYYYMMDD digits.")
        filename = f"daily_volume_{trading_date_yyyymmdd}.xlsx"
        errors: list[str] = []
        for base_url in self._BASE_URLS:
            try:
                self._warm_archive(base_url)
                payload = self._get(base_url + filename, referer=base_url)
                self._validate_xlsx(payload.content, payload.content_type, payload.url)
                return payload
            except (HTTPError, URLError, ValueError) as exc:
                errors.append(f"{base_url}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Unable to read CME public daily-volume workbook from official archive mirrors: "
            + " | ".join(errors)
        )

    def _warm_archive(self, base_url: str) -> None:
        # A warm-up request establishes the same public archive session a browser uses.
        request = Request(
            base_url,
            headers={**self._HEADERS, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
        )
        with self._opener.open(request, timeout=30) as response:
            response.read(1024)

    def _get(self, url: str, referer: str) -> CmeFilePayload:
        request = Request(url, headers={**self._HEADERS, "Referer": referer})
        with self._opener.open(request, timeout=60) as response:
            return CmeFilePayload(
                url=response.geturl(),
                content=response.read(),
                content_type=response.headers.get("Content-Type"),
            )

    @staticmethod
    def _validate_xlsx(content: bytes, content_type: str | None, url: str) -> None:
        if len(content) < 4 or content[:4] != b"PK\x03\x04":
            preview = content[:120].decode("utf-8", errors="replace").replace("\n", " ")
            raise ValueError(
                f"CME response is not an OOXML ZIP workbook: url={url}, "
                f"content_type={content_type}, bytes={len(content)}, preview={preview!r}"
            )
