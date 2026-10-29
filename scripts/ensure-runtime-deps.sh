#!/usr/bin/env bash
set -euo pipefail

# Guard a Hermes runtime venv against environment sync tools pruning the
# MemPalace provider's runtime dependencies. Intended for systemd ExecStartPre,
# post-update hooks, or quiet watchdog jobs. --check never changes a runtime.

HERMES_SRC=${HERMES_SRC:-$(pwd)}
PY=${PY:-$HERMES_SRC/.venv/bin/python}
UV=${UV:-uv}
PALACE=${MEMPALACE_PALACE_PATH:-${PALACE:-$HOME/.mempalace/palace}}
LOG_FILE=${LOG_FILE:-}
REQ_MEMPALACE="${REQ_MEMPALACE:-3.11.0}"
REQ_CHROMADB=${REQ_CHROMADB:-1.5.9}
REQ_AIOHTTP_SPEC=${REQ_AIOHTTP_SPEC:-">=3.14,<3.15"}
CHECK_ONLY=0
if [[ ${1:-} == "--check" ]]; then CHECK_ONLY=1; shift; fi
if (($#)); then echo "usage: $0 [--check]" >&2; exit 2; fi

log() {
  local line
  line="$(date --iso-8601=seconds) $*"
  if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    printf '%s\n' "$line" | tee -a "$LOG_FILE" >&2
  else
    printf '%s\n' "$line" >&2
  fi
}

if [[ ! -x "$PY" ]]; then log "ERROR Hermes runtime python not found: $PY"; exit 2; fi

check_runtime() {
  "$PY" - "$PALACE" "$REQ_MEMPALACE" "$REQ_CHROMADB" "$REQ_AIOHTTP_SPEC" <<'PY'
import importlib.metadata as md
import importlib.util
import sqlite3
import sys
from pathlib import Path

palace = Path(sys.argv[1]).expanduser()
required_versions = {"mempalace": sys.argv[2], "chromadb": sys.argv[3]}
aiohttp_spec = sys.argv[4]
required_modules = ["mempalace", "mempalace.searcher", "mempalace.query_sanitizer", "mempalace.knowledge_graph", "mempalace.palace_graph", "mempalace.config", "mempalace.palace", "chromadb"]
problems = []
for pkg, want in required_versions.items():
    try: got = md.version(pkg)
    except md.PackageNotFoundError:
        problems.append(f"missing package: {pkg}")
        continue
    if got != want: problems.append(f"wrong package version: {pkg}={got}, want {want}")
try:
    aiohttp_version = tuple(int(part) for part in md.version("aiohttp").split(".")[:3])
except (md.PackageNotFoundError, ValueError):
    problems.append("missing or invalid package version: aiohttp")
else:
    if not ((3, 14, 0) <= aiohttp_version < (3, 15, 0)):
        problems.append(f"wrong package version: aiohttp={md.version('aiohttp')}, want {aiohttp_spec}")
for mod in required_modules:
    try: spec = importlib.util.find_spec(mod)
    except ModuleNotFoundError as exc: problems.append(f"missing module: {mod} ({exc})"); continue
    if spec is None: problems.append(f"missing module: {mod}")
db = palace / "chroma.sqlite3"
if not db.exists(): problems.append(f"missing palace DB: {db}")
else:
    try:
        with sqlite3.connect(db) as con: row = con.execute("PRAGMA quick_check;").fetchone()
        if not row or row[0] != "ok": problems.append(f"chroma.sqlite3 quick_check={row!r}")
    except Exception as exc: problems.append(f"chroma.sqlite3 quick_check_error={exc!r}")
if problems:
    print("\n".join(problems)); raise SystemExit(1)
print("OK")
PY
}

check_log=$(mktemp -t hermes-plugin-mempalace-runtime-check.XXXXXX)
trap 'rm -f "$check_log"' EXIT
if check_runtime >"$check_log" 2>&1; then
  [[ -n "$LOG_FILE" ]] && cat "$check_log" >>"$LOG_FILE"
  log "OK MemPalace runtime dependencies and palace DB are healthy"
  exit 0
fi
if ((CHECK_ONLY)); then
  cat "$check_log" >&2 || true
  log "ERROR MemPalace runtime dependency check failed"
  exit 1
fi
log "Repairing MemPalace runtime dependencies for $PY"
[[ -n "$LOG_FILE" ]] && cat "$check_log" >>"$LOG_FILE" || true
"$UV" pip install --no-config --python "$PY" "mempalace==$REQ_MEMPALACE" "chromadb==$REQ_CHROMADB" "aiohttp$REQ_AIOHTTP_SPEC" ${UV_PIP_EXTRA_ARGS:-}
if check_runtime; then log "OK repaired MemPalace runtime dependencies"; exit 0; fi
log "ERROR MemPalace runtime dependency repair failed"
exit 1
