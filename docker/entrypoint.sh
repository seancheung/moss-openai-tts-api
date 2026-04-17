#!/usr/bin/env bash
set -euo pipefail

: "${MOSS_VARIANT:=auto}"
: "${MOSS_VOICES_DIR:=/voices}"
: "${MOSS_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

# MOSS-TTS' remote-code processors reject cache_dir kwargs, so we drive the
# cache location through HF_HOME. When the user sets MOSS_CACHE_DIR, honour it
# here — before python/transformers gets a chance to cache the default path.
if [ -n "${MOSS_CACHE_DIR:-}" ]; then
  export HF_HOME="$MOSS_CACHE_DIR"
fi

export MOSS_VARIANT MOSS_VOICES_DIR MOSS_DEVICE HOST PORT LOG_LEVEL
if [ -n "${MOSS_MODEL:-}" ]; then
  export MOSS_MODEL
fi

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
