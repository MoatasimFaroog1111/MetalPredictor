from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class SourceIntegrityError(RuntimeError):
    """Raised when immutable research input bytes differ from preregistration."""


def git_blob_sha1(path: Path) -> str:
    """Return the SHA-1 Git assigns to the exact file bytes as a blob object."""

    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


@dataclass(frozen=True)
class GitBlobIntegrityVerifier:
    """Fail closed when an immutable preregistered source blob has changed."""

    expected_sha1: str

    def verify(self, path: Path) -> str:
        actual = git_blob_sha1(Path(path))
        if actual != self.expected_sha1:
            raise SourceIntegrityError(
                f"Stage-7 source integrity failure for {path}: expected Git blob "
                f"{self.expected_sha1}, got {actual}."
            )
        return actual
