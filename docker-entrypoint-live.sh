#!/bin/sh
set -eu

DB_PATH="${LIVE_DB_PATH:-/data/live_predictions.sqlite3}"
DB_DIR="$(dirname "$DB_PATH")"
VOLUME_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"

mkdir -p "$DB_DIR" "$VOLUME_DIR"
chown -R appuser:appuser "$DB_DIR" "$VOLUME_DIR"

exec gosu appuser "$@"
