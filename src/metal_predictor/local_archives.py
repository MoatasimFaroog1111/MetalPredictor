from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ArchiveFingerprint:
    path: str
    name: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FileFingerprinter(Protocol):
    def fingerprint(self, path: Path) -> ArchiveFingerprint: ...


class Sha256FileFingerprinter:
    """Single-purpose streaming SHA-256 file fingerprint service."""

    def __init__(self, chunk_size_bytes: int = 1024 * 1024) -> None:
        if chunk_size_bytes < 4096:
            raise ValueError("chunk_size_bytes must be >= 4096.")
        self._chunk_size = chunk_size_bytes

    def fingerprint(self, path: Path) -> ArchiveFingerprint:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(self._chunk_size), b""):
                digest.update(chunk)
        return ArchiveFingerprint(
            path=str(path),
            name=path.name,
            size_bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
        )


@dataclass(frozen=True)
class ArchiveCatalogReport:
    discovered_files: int
    unique_content_files: int
    duplicate_content_files: int
    selected: tuple[ArchiveFingerprint, ...]
    duplicates_by_sha256: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected"] = [item.as_dict() for item in self.selected]
        payload["duplicates_by_sha256"] = {
            key: list(value) for key, value in self.duplicates_by_sha256.items()
        }
        return payload


class ContentAddressedArchiveCatalog:
    """Discovers local ZIP inputs and removes only byte-identical duplicate archives.

    Content deduplication is deliberately separate from market-row deduplication. A ZIP
    whose bytes differ is never discarded just because its filename resembles another
    file; downstream timestamp-quality logic must decide whether its observations can
    coexist safely.
    """

    def __init__(
        self,
        pattern: str = "HISTDATA_COM_MS_XAGUSD_M1*.zip",
        fingerprinter: FileFingerprinter | None = None,
    ) -> None:
        self._pattern = pattern
        self._fingerprinter = fingerprinter or Sha256FileFingerprinter()

    def discover(self, directory: Path) -> tuple[tuple[Path, ...], ArchiveCatalogReport]:
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(directory)
        paths = tuple(sorted(path for path in directory.glob(self._pattern) if path.is_file()))
        if not paths:
            raise FileNotFoundError(
                f"No local HistData archives matching {self._pattern!r} in {directory}."
            )

        grouped: dict[str, list[Path]] = {}
        fingerprints: dict[Path, ArchiveFingerprint] = {}
        for path in paths:
            fingerprint = self._fingerprinter.fingerprint(path)
            fingerprints[path] = fingerprint
            grouped.setdefault(fingerprint.sha256, []).append(path)

        selected: list[Path] = []
        duplicates: dict[str, tuple[str, ...]] = {}
        for digest, candidates in sorted(grouped.items()):
            canonical = min(
                candidates,
                key=lambda path: (
                    "(" in path.name,
                    len(path.name),
                    path.name,
                ),
            )
            selected.append(canonical)
            if len(candidates) > 1:
                duplicates[digest] = tuple(sorted(path.name for path in candidates))

        selected = sorted(selected, key=lambda path: path.name)
        report = ArchiveCatalogReport(
            discovered_files=len(paths),
            unique_content_files=len(selected),
            duplicate_content_files=len(paths) - len(selected),
            selected=tuple(fingerprints[path] for path in selected),
            duplicates_by_sha256=duplicates,
        )
        return tuple(selected), report
