# hermes-plugin-mempalace

MemPalace memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Bridges Hermes's `MemoryProvider` interface to [MemPalace](https://github.com/MemPalace/mempalace) — a local-first semantic memory system with ChromaDB vector search, a SQLite knowledge graph, agent diaries, and wing/room/drawer organisation. Once installed, the agent gains five memory tools, automatic session-end diary writes, and pre-compression distillation that preserves key turns before Hermes discards them.

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
pip install 'mempalace>=3.0'
```

## Configuration

All settings are optional — sensible defaults are used.

| Env var | Default | Purpose |
|---|---|---|
| `MEMPALACE_PALACE_PATH` | `~/.mempalace/palace` | Location of the palace data directory |
| `MEMPALACE_WING` | `agent` | Default wing used when auto-storing conversation turns |
| `MEMPALACE_N_RECALL` | `5` | Results fetched per prefetch lookup |

If `~/.mempalace/config.json` exists and contains a `palace_path` field, that takes precedence over the default (but `MEMPALACE_PALACE_PATH` still wins overall).

## Tools exposed to the agent

| Tool | What it does |
|---|---|
| `mempalace_search` | Semantically search stored memories; optionally scope to a wing |
| `mempalace_add` | File a memory into a `wing/room` drawer |
| `mempalace_diary` | Write a dated diary entry for the current agent identity |
| `mempalace_status` | Palace stats — drawer count and wing/room breakdown |
| `mempalace_kg_query` | Query the knowledge graph for relationships about an entity |

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

Built against Hermes Agent's `agent.memory_provider.MemoryProvider` interface. Tested with MemPalace 3.x.

## Credits

- MemPalace authored by Milla Jovovich and Ben Sigman — see [MemPalace/mempalace](https://github.com/MemPalace/mempalace).
- Hermes Agent by [Nous Research](https://github.com/NousResearch/hermes-agent).
- Plugin authored by [@OVER99K](https://github.com/OVER99K).

## License

MIT — see [LICENSE](LICENSE).
