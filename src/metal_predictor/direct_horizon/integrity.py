from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class SourceIntegrityError(RuntimeError):
    """Raised when immutable research input bytes differ from preregistration."""


def git_blob_sha1_bytes(data: bytes) -> str:
    """Return the SHA-1 Git assigns to the exact bytes as a blob object."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return the SHA-1 Git assigns to the exact file bytes as a blob object."""

    return git_blob_sha1_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class GitBlobIntegrityVerifier:
    """Fail closed when an immutable preregistered source blob has changed."""

    expected_sha1: str

    def verify_bytes(self, data: bytes, *, source_label: str) -> str:
        actual = git_blob_sha1_bytes(data)
        if actual != self.expected_sha1:
            raise SourceIntegrityError(
                f"Stage-7 source integrity failure for {source_label}: expected Git blob "
                f"{self.expected_sha1}, got {actual}."
            )
        return actual

    def verify(self, path: Path) -> str:
        path = Path(path)
        return self.verify_bytes(path.read_bytes(), source_label=str(path))


@dataclass(frozen=True)
class VerifiedSourceSnapshot:
    """One verified immutable byte snapshot consumed by every Stage-7 horizon."""

    data: bytes
    git_blob_sha1: str
    source_label: str

    @classmethod
    def capture(cls, path: Path, *, expected_sha1: str) -> "VerifiedSourceSnapshot":
        path = Path(path)
        data = path.read_bytes()
        actual = GitBlobIntegrityVerifier(expected_sha1).verify_bytes(
            data,
            source_label=str(path),
        )
        return cls(data=data, git_blob_sha1=actual, source_label=str(path))
