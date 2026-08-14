from __future__ import annotations

import os
from pathlib import Path
import pwd
import sys


APP_USER = "appuser"
DEFAULT_DB_PATH = Path("/data/live_predictions.sqlite3")
DEFAULT_MICROSTRUCTURE_DB_PATH = Path("/data/bullionvault_microstructure.sqlite3")
DEFAULT_FORWARD_BARS_DB_PATH = Path("/data/bullionvault_forward_bars.sqlite3")
DEFAULT_SHADOW62_DB_PATH = Path("/data/xpt_xpd_shadow62.sqlite3")
DEFAULT_VOLUME_PATH = Path("/data")


def _chown_if_exists(path: Path, uid: int, gid: int) -> None:
    if path.exists():
        os.chown(path, uid, gid)


def _resolve_runtime_path(environment_name: str, default: Path) -> Path:
    path = Path(os.getenv(environment_name, str(default))).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _prepare_sqlite_path(path: Path, uid: int, gid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chown(path.parent, uid, gid)
    _chown_if_exists(path, uid, gid)
    for suffix in ("-wal", "-shm", "-journal"):
        _chown_if_exists(Path(f"{path}{suffix}"), uid, gid)


def _prepare_runtime_storage(uid: int, gid: int) -> None:
    db_paths = {
        _resolve_runtime_path("LIVE_DB_PATH", DEFAULT_DB_PATH),
        _resolve_runtime_path(
            "BULLIONVAULT_MICROSTRUCTURE_DB_PATH",
            DEFAULT_MICROSTRUCTURE_DB_PATH,
        ),
        _resolve_runtime_path(
            "BULLIONVAULT_FORWARD_BARS_DB_PATH",
            DEFAULT_FORWARD_BARS_DB_PATH,
        ),
        _resolve_runtime_path("SHADOW62_DB_PATH", DEFAULT_SHADOW62_DB_PATH),
    }
    volume_path = _resolve_runtime_path("RAILWAY_VOLUME_MOUNT_PATH", DEFAULT_VOLUME_PATH)
    volume_path.mkdir(parents=True, exist_ok=True)
    os.chown(volume_path, uid, gid)

    for db_path in db_paths:
        _prepare_sqlite_path(db_path, uid, gid)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("docker_entrypoint_live.py requires a command to execute")

    account = pwd.getpwnam(APP_USER)
    if os.geteuid() == 0:
        _prepare_runtime_storage(account.pw_uid, account.pw_gid)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
