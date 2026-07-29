# hermes-plugin-mempalace

MemPalace memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Bridges Hermes's `MemoryProvider` interface to [MemPalace](https://github.com/MemPalace/mempalace) — a local-first semantic memory system with ChromaDB vector search, a SQLite knowledge graph, agent diaries, and wing/room/drawer organisation. Once installed, the agent gains the full MemPalace tool surface, automatic session-end diary writes, and pre-compression distillation that preserves key turns before Hermes discards them.

## Install

```
hermes plugins install OVER99K/hermes-plugin-mempalace
```

Then enable it in your Hermes config:

```yaml
# ~/.hermes/config.yaml  (or profile config)
memory:
  provider: mempalace
```

You also need MemPalace itself installed:

```
pip install 'mempalace>=3.6.0' 'chromadb==1.5.9' 'aiohttp>=3.14,<3.15'
```

If Hermes is maintained from a source checkout with `uv sync`, read [Surviving Hermes updates](docs/hermes-update-survival.md). Exact syncs can prune plugin-only dependencies from the active Hermes venv; the included guard script can verify and repair the runtime environment.

## Configuration

All settings are optional — sensible defaults are used.

| Env var | Default | Purpose |
|---|---|---|
| `MEMPALACE_PALACE_PATH` | `~/.mempalace/palace` | Location of the palace data directory |
| `MEMPALACE_WING` | `agent` | Default wing used when auto-storing conversation turns |
| `MEMPALACE_N_RECALL` | `5` | Results fetched per prefetch lookup |

If `~/.mempalace/config.json` exists and contains a `palace_path` field, that takes precedence over the default (but `MEMPALACE_PALACE_PATH` still wins overall).

## Native tools exposed to the agent

The plugin calls the MemPalace Python library directly in the Hermes process. It does not start or connect to an MCP server.

| Tool | What it does |
|---|---|
| `mempalace_search` | Semantically search stored memories; optionally scope by wing, room, or exact source |
| `mempalace_add_drawer` | File a memory into a `wing/room` drawer |
| `mempalace_checkpoint` | Deduplicate and save several memories plus an optional diary entry in one call |
| `mempalace_list_drawers` | Browse drawers with `authored_at`/`filed_at` chronology and ISO time windows |
| `mempalace_delete_by_source` | Dry-run-first cleanup of drawers and closets from one exact source |
| `mempalace_diary_write` | Write a dated diary entry for an agent |
| `mempalace_status` | Palace stats — drawer count and wing/room breakdown |
| `mempalace_kg_query` | Query the knowledge graph for relationships about an entity |
| `mempalace_kg_supersede` | Atomically replace a temporal fact at one shared boundary |

The full native surface also includes drawer CRUD, duplicate checks, taxonomy, KG add/invalidate/timeline/stats, graph traversal/stats, diary reads, hook settings, and source-scoped search. MemPalace 3.6 provenance fields (`valid_to`, source drawer/file, adapter, confidence) are accepted by `mempalace_kg_add` and `mempalace_kg_supersede`.

## Tests

The focused provider tests are self-contained and do not require a Hermes Agent checkout. They install MemPalace and its runtime dependencies from `requirements-test.txt`; `tests/conftest.py` supplies only the minimal `MemoryProvider` interface needed to import the plugin.

```bash
python -m venv .test-venv
.test-venv/bin/pip install -r requirements-test.txt
.test-venv/bin/pytest -q
```

## Automatic behaviour

- **Prefetch**: on every user turn, kicks off a background semantic search. The result is injected into context on the *next* turn if it's ready, or fetched synchronously as a fallback.
- **Turn sync**: user↔assistant turns are stored as drawers in the `conversations` room (primary agent context only — cron/subagent turns are skipped to avoid noise).
- **Session end**: the last ten salient turns of the session are written to the agent's diary, tagged by `agent_identity`.
- **Pre-compression hook**: when Hermes is about to compress the message log, the last five assistant messages are distilled into a `compressed-context` drawer so the information survives compression.
- **Memory mirroring**: when Hermes's built-in memory writes an entry (`add`/`replace`), it's mirrored into the palace under the `memory` or `user-profile` room.

## How it differs from the other Hermes memory plugins

Hermes ships with several memory providers (`mem0`, `honcho`, `byterover`, `hindsight`, `holographic`, `openviking`, `retaindb`, `supermemory`). This one is distinct in that:

- It runs **entirely local** — no API calls, no hosted service, no account needed.
- It organises memories **structurally** (`wing → room → drawer`) in addition to semantic search, which fits a palace-of-the-mind mental model for long-term context.
- It maintains a **knowledge graph** alongside vector search, so relationship queries (`who is X`, `what does X use`) don't rely on embedding similarity alone.
- It has a per-agent **diary**, making multi-agent setups (e.g. Hermes profiles sharing a palace) legible.

## Compatibility

Built against Hermes Agent's `agent.memory_provider.MemoryProvider` interface. The 1.2 tool surface requires MemPalace 3.6+.

## Credits

- MemPalace authored by Milla Jovovich and Ben Sigman — see [MemPalace/mempalace](https://github.com/MemPalace/mempalace).
- Hermes Agent by [Nous Research](https://github.com/NousResearch/hermes-agent).
- Plugin authored by [@OVER99K](https://github.com/OVER99K).

## License

MIT — see [LICENSE](LICENSE).
