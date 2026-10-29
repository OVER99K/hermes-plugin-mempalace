# Surviving Hermes updates

Hermes source checkouts are commonly maintained with `uv sync`. A plain sync reconciles the virtualenv to Hermes's declared project dependencies and may remove packages installed only for profile-local plugins. If this plugin is installed as a profile plugin, that can leave Hermes able to import the plugin file but unable to execute the richer MemPalace tool surface.

Typical partial failure after a sync/prune:

```text
mempalace_status works
mempalace_search      -> No module named 'mempalace.query_sanitizer'
mempalace_kg_stats    -> No module named 'mempalace.knowledge_graph'
mempalace_graph_stats -> No module named 'mempalace.palace_graph'
```

That is not palace database corruption. It means the active Hermes runtime Python is missing the `mempalace` package modules.

## Prefer inexact sync when keeping plugin dependencies

If your operational Hermes environment has profile-local plugins, prefer:

```bash
uv sync --active --all-extras --no-dev --inexact
```

`--inexact` keeps packages that are not declared by Hermes itself, including plugin dependencies installed into the same runtime venv.

If you need an exact sync, run the guard below immediately afterwards.

## Runtime dependency guard

This repo includes:

```text
scripts/ensure-runtime-deps.sh
```

It verifies the active Hermes runtime venv has:

- `mempalace`
- `chromadb`
- `aiohttp`
- rich MemPalace modules used by the tool surface:
  - `mempalace.searcher`
  - `mempalace.query_sanitizer`
  - `mempalace.knowledge_graph`
  - `mempalace.palace_graph`
  - `mempalace.config`
  - `mempalace.palace`
- a readable palace DB with `PRAGMA quick_check == ok`

If a check fails, it repairs the runtime venv with `uv pip install --python ...` and rechecks.

Run it from a Hermes source checkout:

```bash
cd /path/to/hermes-agent
/path/to/hermes-plugin-mempalace/scripts/ensure-runtime-deps.sh
```

Or pass explicit paths:

```bash
HERMES_SRC=/path/to/hermes-agent \
PY=/path/to/hermes-agent/.venv/bin/python \
PALACE=$HOME/.mempalace/palace \
/path/to/hermes-plugin-mempalace/scripts/ensure-runtime-deps.sh
```

Useful overrides:

| Env var | Default | Purpose |
|---|---|---|
| `HERMES_SRC` | current directory | Hermes source checkout containing `.venv` |
| `PY` | `$HERMES_SRC/.venv/bin/python` | Runtime Python to check/repair |
| `UV` | `uv` | `uv` executable |
| `PALACE` / `MEMPALACE_PALACE_PATH` | `$HOME/.mempalace/palace` | Palace data directory |
| `REQ_MEMPALACE` | `3.11.0` | MemPalace package version to enforce |
| `REQ_CHROMADB` | `1.5.9` | ChromaDB package version to enforce |
| `REQ_AIOHTTP` | `3.14.1` | aiohttp package version to enforce |
| `LOG_FILE` | unset | Optional append-only log path |
| `UV_PIP_EXTRA_ARGS` | unset | Extra args passed to `uv pip install` |

## systemd ExecStartPre

For long-running gateway/dashboard services, wire the guard into startup so a post-update restart self-repairs before Hermes starts:

```ini
[Service]
ExecStartPre=/path/to/hermes-plugin-mempalace/scripts/ensure-runtime-deps.sh
```

If the service's `WorkingDirectory` is not the Hermes source checkout, set `HERMES_SRC` and `PY` explicitly:

```ini
[Service]
Environment=HERMES_SRC=/path/to/hermes-agent
Environment=PY=/path/to/hermes-agent/.venv/bin/python
Environment=PALACE=/home/agent/.mempalace/palace
ExecStartPre=/path/to/hermes-plugin-mempalace/scripts/ensure-runtime-deps.sh
```

## Quiet watchdog

For scheduled drift detection, run the guard from a wrapper that only prints when repair or failure occurs:

```bash
#!/usr/bin/env bash
set -euo pipefail
if /path/to/hermes-plugin-mempalace/scripts/ensure-runtime-deps.sh >/tmp/mempalace-guard.log 2>&1; then
  if grep -q 'OK repaired' /tmp/mempalace-guard.log; then
    cat /tmp/mempalace-guard.log
  fi
else
  cat /tmp/mempalace-guard.log
  exit 1
fi
```

In Hermes cron, make it a `no_agent` job so empty stdout is silent.

## Verification after repair

Do not stop at imports. Verify the actual Hermes tool surface from a fresh session/context:

- `mempalace_search`
- `mempalace_kg_stats`
- `mempalace_graph_stats`

Those are the tools most likely to expose partial package pruning.
