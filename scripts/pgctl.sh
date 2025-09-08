#!/usr/bin/env bash
set -euo pipefail

PG_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/install/bin"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Prefer ./pgdata if present, else fallback to ./data/postgres
if [[ -d "${ROOT_DIR}/pgdata" ]]; then
  PGDATA="${ROOT_DIR}/pgdata"
else
  PGDATA="${ROOT_DIR}/data/postgres"
fi

PG_CTL="${PG_BIN}/pg_ctl"

usage() {
  echo "Usage: $0 {start|stop|restart|status|tail-log}"
  echo "  PGDATA: ${PGDATA}"
}

ensure_dirs() {
  mkdir -p "${PGDATA}"
}

start() {
  ensure_dirs
  "${PG_CTL}" -w -t 30 -D "${PGDATA}" -l "${PGDATA}/server.log" start
}

stop() {
  "${PG_CTL}" -m fast -D "${PGDATA}" stop || true
}

restart() {
  ensure_dirs
  # Fast restart first with timeout; fallback to immediate if it times out
  if ! "${PG_CTL}" -w -t 30 -m fast -D "${PGDATA}" restart; then
    echo "Fast restart timed out; forcing immediate restart..." >&2
    "${PG_CTL}" -m immediate -D "${PGDATA}" restart
  fi
}

status() {
  "${PG_CTL}" -D "${PGDATA}" status || true
}

tail_log() {
  tail -n 200 -f "${PGDATA}/server.log"
}

cmd="${1:-}" || true
case "${cmd}" in
  start) start;;
  stop) stop;;
  restart) restart;;
  status) status;;
  tail-log) tail_log;;
  *) usage; exit 1;;
esac

