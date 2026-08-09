from __future__ import annotations

import json

from metal_predictor.cme_public_files import CmePublicFileClient
from metal_predictor.ooxml_reader import XlsxWorkbookReader


SAMPLE_DATE = "20250807"
TOKENS = ("SILVER", "COMEX", "METALS", "OPEN INTEREST", "VOLUME")


def nonempty(row):
    return any(value not in (None, "") for value in row)


def normalize(value) -> str:
    return "" if value is None else str(value).strip().upper()


def run() -> dict[str, object]:
    fetched = CmePublicFileClient().fetch_daily_volume_workbook(SAMPLE_DATE)
    workbook = XlsxWorkbookReader().read_bytes(fetched.content)
    report: dict[str, object] = {
        "source_url": fetched.url,
        "content_type": fetched.content_type,
        "bytes": len(fetched.content),
        "sheet_count": len(workbook),
        "sheets": [],
    }
    for sheet in workbook:
        rows = list(sheet.rows)
        first_nonempty = [list(row) for row in rows if nonempty(row)][:20]
        matches = []
        for index, row in enumerate(rows):
            text = " | ".join(normalize(value) for value in row)
            if any(token in text for token in TOKENS):
                start = max(0, index - 3)
                end = min(len(rows), index + 4)
                matches.append({
                    "row_index_1based": index + 1,
                    "row": list(row),
                    "context": [list(candidate) for candidate in rows[start:end]],
                })
        report["sheets"].append({
            "name": sheet.name,
            "row_count": sheet.row_count,
            "max_columns": sheet.max_columns,
            "first_nonempty_rows": first_nonempty,
            "keyword_matches": matches[:80],
        })
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
