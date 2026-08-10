from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class ArchiveFingerprint:
    path: str
    name: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


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

    def __init__(self, pattern: str = "HISTDATA_COM_MS_XAGUSD_M1*.zip") -> None:
        self._pattern = pattern

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
            digest = self._sha256(path)
            fingerprint = ArchiveFingerprint(
                path=str(path),
                name=path.name,
                size_bytes=path.stat().st_size,
                sha256=digest,
            )
            fingerprints[path] = fingerprint
            grouped.setdefault(digest, []).append(path)

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

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
