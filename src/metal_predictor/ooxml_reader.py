from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
import re
from typing import BinaryIO, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")


@dataclass(frozen=True)
class WorksheetSnapshot:
    name: str
    rows: tuple[tuple[object | None, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def max_columns(self) -> int:
        return max((len(row) for row in self.rows), default=0)


class XlsxWorkbookReader:
    """Minimal read-only OOXML reader for stable machine-generated CME workbooks.

    It deliberately supports values only (shared strings, inline strings, booleans,
    strings and numeric cached values). It never executes formulas or macros.
    This keeps the market-data adapter independent from a heavyweight spreadsheet
    runtime while retaining deterministic parsing of the official XLSX snapshots.
    """

    def read_bytes(self, payload: bytes) -> tuple[WorksheetSnapshot, ...]:
        return self._read(BytesIO(payload))

    def _read(self, source: BinaryIO) -> tuple[WorksheetSnapshot, ...]:
        with ZipFile(source) as archive:
            shared_strings = self._shared_strings(archive)
            sheets = self._sheet_targets(archive)
            return tuple(
                WorksheetSnapshot(name=name, rows=self._worksheet_rows(archive, target, shared_strings))
                for name, target in sheets
            )

    def _shared_strings(self, archive: ZipFile) -> tuple[str, ...]:
        path = "xl/sharedStrings.xml"
        if path not in archive.namelist():
            return ()
        root = ET.fromstring(archive.read(path))
        values: list[str] = []
        for item in root.findall(f"{{{_MAIN_NS}}}si"):
            text = "".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t"))
            values.append(text)
        return tuple(values)

    def _sheet_targets(self, archive: ZipFile) -> tuple[tuple[str, str], ...]:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target_by_id = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }
        result: list[tuple[str, str]] = []
        sheets = workbook.find(f"{{{_MAIN_NS}}}sheets")
        if sheets is None:
            return ()
        for sheet in sheets.findall(f"{{{_MAIN_NS}}}sheet"):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{_REL_NS}}}id"]
            target = target_by_id[rel_id]
            if target.startswith("/"):
                normalized = target.lstrip("/")
            else:
                normalized = str(PurePosixPath("xl") / target)
            result.append((name, normalized))
        return tuple(result)

    def _worksheet_rows(
        self,
        archive: ZipFile,
        target: str,
        shared_strings: tuple[str, ...],
    ) -> tuple[tuple[object | None, ...], ...]:
        root = ET.fromstring(archive.read(target))
        sheet_data = root.find(f"{{{_MAIN_NS}}}sheetData")
        if sheet_data is None:
            return ()
        rows: list[tuple[object | None, ...]] = []
        for row_node in sheet_data.findall(f"{{{_MAIN_NS}}}row"):
            cells: dict[int, object | None] = {}
            max_index = -1
            for cell in row_node.findall(f"{{{_MAIN_NS}}}c"):
                ref = cell.attrib.get("r")
                if not ref:
                    continue
                match = _CELL_REF_RE.fullmatch(ref)
                if match is None:
                    continue
                index = self._column_index(match.group(1))
                cells[index] = self._cell_value(cell, shared_strings)
                max_index = max(max_index, index)
            if max_index < 0:
                rows.append(())
                continue
            rows.append(tuple(cells.get(index) for index in range(max_index + 1)))
        return tuple(rows)

    def _cell_value(self, cell: ET.Element, shared_strings: tuple[str, ...]) -> object | None:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            inline = cell.find(f"{{{_MAIN_NS}}}is")
            if inline is None:
                return ""
            return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))
        value_node = cell.find(f"{{{_MAIN_NS}}}v")
        if value_node is None or value_node.text is None:
            return None
        raw = value_node.text
        if cell_type == "s":
            index = int(raw)
            if index < 0 or index >= len(shared_strings):
                raise ValueError(f"Shared-string index out of range: {index}")
            return shared_strings[index]
        if cell_type in {"str", "e"}:
            return raw
        if cell_type == "b":
            return raw == "1"
        try:
            number = float(raw)
        except ValueError:
            return raw
        if number.is_integer():
            return int(number)
        return number

    @staticmethod
    def _column_index(letters: str) -> int:
        value = 0
        for character in letters:
            value = value * 26 + (ord(character) - ord("A") + 1)
        return value - 1
