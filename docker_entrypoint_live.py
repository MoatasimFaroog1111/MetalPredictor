from __future__ import annotations

import os
from pathlib import Path
import pwd
import sys


APP_USER = "appuser"
DEFAULT_DB_PATH = Path("/data/live_predictions.sqlite3")
DEFAULT_VOLUME_PATH = Path("/data")


def _chown_if_exists(path: Path, uid: int, gid: int) -> None:
    if path.exists():
        os.chown(path, uid, gid)


def _prepare_runtime_storage(uid: int, gid: int) -> None:
    db_path = Path(os.getenv("LIVE_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_dir = db_path.parent

    volume_path = Path(
        os.getenv("RAILWAY_VOLUME_MOUNT_PATH", str(DEFAULT_VOLUME_PATH))
    ).expanduser()
    if not volume_path.is_absolute():
        volume_path = Path.cwd() / volume_path

    for directory in {db_dir, volume_path}:
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, uid, gid)

    _chown_if_exists(db_path, uid, gid)
    for suffix in ("-wal", "-shm", "-journal"):
        _chown_if_exists(Path(f"{db_path}{suffix}"), uid, gid)


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
