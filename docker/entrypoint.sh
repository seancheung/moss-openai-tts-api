#!/usr/bin/env bash
set -euo pipefail

: "${MOSS_VARIANT:=auto}"
: "${MOSS_VOICES_DIR:=/voices}"
: "${MOSS_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export MOSS_VARIANT MOSS_VOICES_DIR MOSS_DEVICE HOST PORT LOG_LEVEL
if [ -n "${MOSS_MODEL:-}" ]; then
  export MOSS_MODEL
fi

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
