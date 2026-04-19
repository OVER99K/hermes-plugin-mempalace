"""MemPalace memory provider for Hermes.

Local-first semantic memory with ChromaDB vector search, knowledge graph,
agent diary, and structured wing/room/drawer retrieval.

All operations run in-process — no MCP subprocess required. The 23 tools
exposed here cover drawer CRUD, navigation, knowledge-graph reads/writes,
diary, and palace-level status. Handlers delegate to the mempalace library
(`searcher`, `backends.chroma`, `knowledge_graph`, `palace_graph`).

Config (config.yaml):
  memory:
    provider: mempalace

Optional env vars:
  MEMPALACE_PALACE_PATH  — path to the palace data dir (default: ~/.mempalace/palace)
  MEMPALACE_WING         — default wing for auto-stored turns (default: agent)
  MEMPALACE_N_RECALL     — results to fetch per prefetch (default: 5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_PALACE_PATH_ENV = "MEMPALACE_PALACE_PATH"
_DEFAULT_HERMES_HOME_PALACE = os.path.expanduser("~/.mempalace/palace")
_METADATA_CACHE_TTL = 5.0
_MAX_RESULTS = 100


PALACE_PROTOCOL = """MemPalace Memory Protocol — how to actually remember:
1. ON WAKE-UP: call mempalace_status to load the palace overview.
2. BEFORE RESPONDING about any person, project, or past event: call mempalace_kg_query or mempalace_search first. Never guess — verify.
3. IF UNSURE about a fact (name, date, relationship, preference): say "let me check" and query the palace. Wrong is worse than slow.
4. WHEN YOU LEARN SOMETHING WORTH KEEPING: call mempalace_add_drawer (content) or mempalace_kg_add (structured fact).
5. WHEN FACTS CHANGE: call mempalace_kg_invalidate on the old fact, mempalace_kg_add for the new one.
6. AT SESSION END: call mempalace_diary_write with what happened, what was learned, what matters.

Storage is not memory — but storage + this protocol = memory."""


AAAK_SPEC = """AAAK is a compressed memory dialect MemPalace uses for efficient storage.
It is readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).
  IMPORTANCE: ★ to ★★★★★ (1-5 scale).
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_palace_path(hermes_home: str = "") -> str:
    env = os.environ.get(_PALACE_PATH_ENV, "").strip()
    if env:
        return env
    cfg_file = os.path.expanduser("~/.mempalace/config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file) as f:
                d = json.load(f)
            if d.get("palace_path"):
                return d["palace_path"]
        except Exception:
            pass
    return _DEFAULT_HERMES_HOME_PALACE


def _fmt_results(results: dict, max_chars: int = 2000) -> str:
    """Format search results into a compact context block."""
    items = results.get("results", [])
    if not items:
        return ""
    parts = []
    total = 0
    for r in items:
        text = r.get("text", "").strip()
        wing = r.get("wing", "")
        room = r.get("room", "")
        label = f"[{wing}/{room}] " if wing else ""
        entry = f"{label}{text}"
        if total + len(entry) > max_chars:
            entry = entry[: max_chars - total - 3] + "..."
            parts.append(entry)
            break
        parts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MemPalaceProvider(MemoryProvider):
    """Hermes MemoryProvider backed by MemPalace."""

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        try:
            import mempalace  # noqa: F401
            palace_path = _resolve_palace_path()
            chroma_db = os.path.join(palace_path, "chroma.sqlite3")
            return os.path.exists(chroma_db)
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._palace_path = _resolve_palace_path(kwargs.get("hermes_home", ""))
        self._wing = os.environ.get("MEMPALACE_WING", "agent")
        self._n_recall = int(os.environ.get("MEMPALACE_N_RECALL", "5"))
        self._agent_context = kwargs.get("agent_context", "primary")
        self._agent_identity = kwargs.get("agent_identity", "nicolette")
        self._prefetch_cache: dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        # In-process chroma + KG caches (avoid re-opening per call)
        self._col = None
        self._col_lock = threading.Lock()
        self._kg = None
        self._kg_lock = threading.Lock()
        self._meta_cache: Optional[list] = None
        self._meta_cache_time: float = 0.0
        self._meta_lock = threading.Lock()
        logger.info("MemPalace provider initialized (palace=%s)", self._palace_path)

    def system_prompt_block(self) -> str:
        try:
            from mempalace.palace import get_collection
            col = get_collection(self._palace_path, create=False)
            count = col.count() if col else 0
        except Exception:
            count = "?"
        return (
            f"You have access to MemPalace — a persistent local memory system with "
            f"{count} stored memories organised into wings and rooms.\n\n"
            f"{PALACE_PROTOCOL}\n\n"
            f"Hot path: mempalace_search (recall), mempalace_add_drawer (store), "
            f"mempalace_diary_write (session end). "
            f"For structured facts about people/projects: mempalace_kg_query / mempalace_kg_add."
        )

    # -- Prefetch / write hooks --------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = session_id or self._session_id
        with self._prefetch_lock:
            cached = self._prefetch_cache.get(key, "")
        if cached:
            with self._prefetch_lock:
                self._prefetch_cache.pop(key, None)
            return cached
        return self._do_search(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        key = session_id or self._session_id

        def _bg():
            result = self._do_search(query)
            with self._prefetch_lock:
                self._prefetch_cache[key] = result

        threading.Thread(target=_bg, daemon=True).start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        entry = f"User: {user_content[:400]}\nAssistant: {assistant_content[:800]}"
        threading.Thread(
            target=self._store_drawer_bg,
            args=(self._wing, "conversations", entry),
            daemon=True,
        ).start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._agent_context != "primary" or not messages:
            return
        summary_parts = []
        for m in messages[-20:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if role in ("user", "assistant") and content:
                summary_parts.append(f"{role.title()}: {content[:300]}")
        if summary_parts:
            threading.Thread(
                target=self._write_diary_bg,
                args=(self._agent_identity, "\n".join(summary_parts[-10:])),
                daemon=True,
            ).start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action in ("add", "replace") and content:
            room = "memory" if target == "memory" else "user-profile"
            threading.Thread(
                target=self._store_drawer_bg,
                args=(self._wing, room, content),
                daemon=True,
            ).start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        texts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if m.get("role") == "assistant" and content:
                texts.append(content[:500])
        if texts:
            combined = "\n".join(texts[-5:])
            threading.Thread(
                target=self._store_drawer_bg,
                args=(self._wing, "compressed-context", combined),
                daemon=True,
            ).start()
        return ""

    def shutdown(self) -> None:
        logger.info("MemPalace provider shutting down")
        with self._kg_lock:
            if self._kg is not None:
                try:
                    self._kg.close()
                except Exception:
                    pass
                self._kg = None

    # -- Tool surface ------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mempalace_status",
                "description": "Palace overview — total drawers, wing and room counts, palace path.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_list_wings",
                "description": "List all wings with their drawer counts.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_list_rooms",
                "description": "List rooms within a wing (or all rooms if no wing given).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wing": {"type": "string", "description": "Wing name (optional — omit for all rooms)"},
                    },
                    "required": [],
                },
            },
            {
                "name": "mempalace_get_taxonomy",
                "description": "Full taxonomy tree: wing → room → drawer count.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_get_aaak_spec",
                "description": "Return the AAAK dialect specification — the compressed memory format MemPalace uses. Call before reading or writing AAAK-compressed memories.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_search",
                "description": "Semantic search over drawers. Returns the most relevant stored memories matching the query. Use to recall facts, past decisions, research, or context before answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "limit": {"type": "integer", "description": "Max results (default 5, max 100)"},
                        "wing": {"type": "string", "description": "Limit to a wing (optional)"},
                        "room": {"type": "string", "description": "Limit to a room (optional)"},
                        "max_distance": {"type": "number", "description": "Max cosine distance; lower=stricter (default 1.5)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mempalace_check_duplicate",
                "description": "Before filing, check whether content (or something very similar) already exists. Returns matches above a similarity threshold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to check for duplicates"},
                        "threshold": {"type": "number", "description": "Similarity threshold (default 0.9)"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mempalace_add_drawer",
                "description": "File content into a wing/room. Idempotent — the same content filed twice is a no-op. Use for facts, decisions, research findings, anything worth long-term recall. 'wing' groups by entity/project, 'room' by topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wing": {"type": "string", "description": "Entity or project (e.g. 'nicolette', 'homelab')"},
                        "room": {"type": "string", "description": "Topic (e.g. 'facts', 'decisions', 'research')"},
                        "content": {"type": "string", "description": "The memory content"},
                        "source_file": {"type": "string", "description": "Origin file (optional)"},
                    },
                    "required": ["wing", "room", "content"],
                },
            },
            {
                "name": "mempalace_delete_drawer",
                "description": "Delete a drawer by its ID. Use carefully — prefer update_drawer unless you truly want it gone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drawer_id": {"type": "string", "description": "Drawer ID to delete"},
                    },
                    "required": ["drawer_id"],
                },
            },
            {
                "name": "mempalace_get_drawer",
                "description": "Fetch a single drawer by ID. Returns full content and metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drawer_id": {"type": "string", "description": "Drawer ID"},
                    },
                    "required": ["drawer_id"],
                },
            },
            {
                "name": "mempalace_list_drawers",
                "description": "List drawers with pagination, optionally filtered by wing and/or room. Returns drawer IDs, location, and content previews.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wing": {"type": "string", "description": "Filter by wing (optional)"},
                        "room": {"type": "string", "description": "Filter by room (optional)"},
                        "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
                        "offset": {"type": "integer", "description": "Pagination offset (default 0)"},
                    },
                    "required": [],
                },
            },
            {
                "name": "mempalace_update_drawer",
                "description": "Update an existing drawer's content and/or metadata. Any field omitted is left unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drawer_id": {"type": "string", "description": "Drawer ID to update"},
                        "content": {"type": "string", "description": "New content (optional)"},
                        "wing": {"type": "string", "description": "New wing (optional)"},
                        "room": {"type": "string", "description": "New room (optional)"},
                    },
                    "required": ["drawer_id"],
                },
            },
            {
                "name": "mempalace_kg_query",
                "description": "Query the knowledge graph for an entity's relationships. Returns typed facts with temporal validity. E.g. 'Max' → child_of Alice, loves chess. Filter by date with as_of to see what was true at a point in time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity to query (e.g. 'Max', 'hummingbird-tet')"},
                        "as_of": {"type": "string", "description": "Only facts valid at this date (YYYY-MM-DD, optional)"},
                        "direction": {"type": "string", "description": "outgoing / incoming / both (default both)"},
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "mempalace_kg_add",
                "description": "Add a structured fact to the knowledge graph. Subject → predicate → object with optional time window. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Subject entity"},
                        "predicate": {"type": "string", "description": "Relationship (snake_case)"},
                        "object": {"type": "string", "description": "Object entity or value"},
                        "valid_from": {"type": "string", "description": "Start date (YYYY-MM-DD, optional)"},
                        "source_closet": {"type": "string", "description": "Closet this fact came from (optional)"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
            {
                "name": "mempalace_kg_invalidate",
                "description": "Mark a fact as no longer true. Sets its end date instead of deleting, so history is preserved.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Subject"},
                        "predicate": {"type": "string", "description": "Predicate"},
                        "object": {"type": "string", "description": "Object"},
                        "ended": {"type": "string", "description": "End date (YYYY-MM-DD, default today)"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
            {
                "name": "mempalace_kg_timeline",
                "description": "Chronological timeline of facts, optionally scoped to a single entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity to scope timeline to (optional)"},
                    },
                    "required": [],
                },
            },
            {
                "name": "mempalace_kg_stats",
                "description": "Knowledge-graph overview: entity count, triple count, relationship types, current vs expired.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_traverse",
                "description": "Walk the palace graph from a starting room, following implicit links between drawers. Use to discover connected ideas across wings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_room": {"type": "string", "description": "Room to start from"},
                        "max_hops": {"type": "integer", "description": "Max graph hops (default 2, max 10)"},
                    },
                    "required": ["start_room"],
                },
            },
            {
                "name": "mempalace_graph_stats",
                "description": "Palace graph overview: nodes, edges, connectivity stats.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_diary_write",
                "description": "Write a diary entry for an agent. Entries are timestamped and accumulate — this is the agent's personal journal: observations, decisions, what mattered. Call at session end or after anything noteworthy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Agent whose diary to write (e.g. 'nicolette')"},
                        "entry": {"type": "string", "description": "The diary entry"},
                        "topic": {"type": "string", "description": "Topic tag (default 'general')"},
                    },
                    "required": ["agent_name", "entry"],
                },
            },
            {
                "name": "mempalace_diary_read",
                "description": "Read an agent's most recent diary entries in chronological (newest-first) order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Agent whose diary to read"},
                        "last_n": {"type": "integer", "description": "How many recent entries (default 10, max 100)"},
                    },
                    "required": ["agent_name"],
                },
            },
            {
                "name": "mempalace_hook_settings",
                "description": "Get or set MemPalace hook behaviour (silent_save, desktop_toast). Call with no args to see current settings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "silent_save": {"type": "boolean", "description": "If True, hook saves silently with no MCP clutter"},
                        "desktop_toast": {"type": "boolean", "description": "If True, desktop notify-send toast on save"},
                    },
                    "required": [],
                },
            },
            {
                "name": "mempalace_memories_filed_away",
                "description": "Acknowledge and consume the latest silent-checkpoint marker from the hook state file. Returns a short summary.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**(args or {}))
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except TypeError as e:
            logger.warning("Tool '%s' arg mismatch: %s", tool_name, e)
            return json.dumps({"error": f"Invalid arguments: {e}"})
        except Exception as e:
            logger.exception("MemPalace tool '%s' failed", tool_name)
            return json.dumps({"error": str(e)})

    # -- Internal: collection / KG / metadata caches -----------------------

    def _get_col(self, create: bool = False):
        with self._col_lock:
            if self._col is not None:
                return self._col
        from mempalace.backends.chroma import ChromaBackend
        backend = ChromaBackend()
        try:
            col = backend.get_collection(self._palace_path, "mempalace_drawers", create=create)
        except Exception as e:
            logger.warning("get_collection failed: %s", e)
            return None
        with self._col_lock:
            self._col = col
        return col

    def _get_kg(self):
        with self._kg_lock:
            if self._kg is not None:
                return self._kg
        from mempalace.knowledge_graph import KnowledgeGraph
        # KG sqlite lives in the palace's parent directory
        # (e.g. /home/over/.mempalace/knowledge_graph.sqlite3).
        kg_db = os.path.join(os.path.dirname(self._palace_path), "knowledge_graph.sqlite3")
        kg = KnowledgeGraph(db_path=kg_db)
        with self._kg_lock:
            self._kg = kg
        return kg

    def _invalidate_meta_cache(self) -> None:
        with self._meta_lock:
            self._meta_cache = None
            self._meta_cache_time = 0.0

    def _fetch_all_metadata(self, col, where=None) -> list:
        total = col.count()
        out: list = []
        offset = 0
        while offset < total:
            kwargs = {"include": ["metadatas"], "limit": 1000, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
            if not batch["metadatas"]:
                break
            out.extend(batch["metadatas"])
            offset += len(batch["metadatas"])
        return out

    def _get_cached_metadata(self, col, where=None) -> list:
        now = time.time()
        with self._meta_lock:
            cached = self._meta_cache
            cached_time = self._meta_cache_time
        if where is None and cached is not None and (now - cached_time) < _METADATA_CACHE_TTL:
            return cached
        result = self._fetch_all_metadata(col, where=where)
        if where is None:
            with self._meta_lock:
                self._meta_cache = result
                self._meta_cache_time = now
        return result

    def _no_palace(self) -> dict:
        return {
            "error": "No palace found",
            "hint": f"Palace not initialized at {self._palace_path}",
        }

    def _sanitize_optional(self, value, field_name="name"):
        if value is None:
            return None
        from mempalace.config import sanitize_name
        return sanitize_name(value, field_name)

    # -- Background write helpers (used by sync_turn, on_session_end, etc) -

    def _store_drawer_bg(self, wing: str, room: str, content: str) -> None:
        try:
            col = self._get_col(create=True)
            if col is None:
                return
            drawer_id = (
                f"drawer_{wing}_{room}_"
                + hashlib.sha256((wing + room + content).encode()).hexdigest()[:24]
            )
            existing = col.get(ids=[drawer_id])
            if existing and existing["ids"]:
                return  # idempotent
            col.upsert(
                ids=[drawer_id],
                documents=[content],
                metadatas=[{
                    "wing": wing,
                    "room": room,
                    "source_file": "",
                    "chunk_index": 0,
                    "added_by": "hermes",
                    "filed_at": datetime.now().isoformat(),
                }],
            )
            self._invalidate_meta_cache()
        except Exception as e:
            logger.warning("MemPalace _store_drawer_bg failed: %s", e)

    def _write_diary_bg(self, agent_name: str, entry: str, topic: str = "general") -> None:
        try:
            self._tool_mempalace_diary_write(agent_name=agent_name, entry=entry, topic=topic)
        except Exception as e:
            logger.debug("Diary write (bg) failed: %s", e)

    def _do_search(self, query: str, wing: str = None, n: int = None) -> str:
        """Lifecycle path (prefetch) — returns a preformatted context block."""
        try:
            from mempalace.searcher import search_memories
            results = search_memories(
                query,
                palace_path=self._palace_path,
                wing=wing,
                n_results=n or self._n_recall,
            )
            if not isinstance(results, dict) or results.get("error"):
                return ""
            return _fmt_results(results)
        except Exception as e:
            logger.warning("MemPalace prefetch search failed: %s", e)
            return ""

    # -- Tool handlers (23) ------------------------------------------------
    # Naming: _tool_<tool_name>. Each returns a dict; the dispatcher wraps
    # it in json.dumps.

    def _tool_mempalace_status(self) -> dict:
        col = self._get_col(create=os.path.isfile(os.path.join(self._palace_path, "chroma.sqlite3")))
        if not col:
            return self._no_palace()
        count = col.count()
        wings: dict = {}
        rooms: dict = {}
        result = {
            "total_drawers": count,
            "wings": wings,
            "rooms": rooms,
            "palace_path": self._palace_path,
        }
        try:
            for m in self._get_cached_metadata(col):
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                wings[w] = wings.get(w, 0) + 1
                rooms[r] = rooms.get(r, 0) + 1
        except Exception as e:
            logger.exception("status metadata fetch failed")
            result["error"] = str(e)
            result["partial"] = True
        return result

    def _tool_mempalace_list_wings(self) -> dict:
        col = self._get_col()
        if not col:
            return self._no_palace()
        wings: dict = {}
        result = {"wings": wings}
        try:
            for m in self._get_cached_metadata(col):
                w = m.get("wing", "unknown")
                wings[w] = wings.get(w, 0) + 1
        except Exception as e:
            result["error"] = str(e)
            result["partial"] = True
        return result

    def _tool_mempalace_list_rooms(self, wing: str = None) -> dict:
        try:
            wing = self._sanitize_optional(wing, "wing")
        except ValueError as e:
            return {"error": str(e)}
        col = self._get_col()
        if not col:
            return self._no_palace()
        rooms: dict = {}
        result = {"wing": wing or "all", "rooms": rooms}
        try:
            where = {"wing": wing} if wing else None
            for m in self._fetch_all_metadata(col, where=where):
                r = m.get("room", "unknown")
                rooms[r] = rooms.get(r, 0) + 1
        except Exception as e:
            result["error"] = str(e)
            result["partial"] = True
        return result

    def _tool_mempalace_get_taxonomy(self) -> dict:
        col = self._get_col()
        if not col:
            return self._no_palace()
        taxonomy: dict = {}
        result = {"taxonomy": taxonomy}
        try:
            for m in self._get_cached_metadata(col):
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                taxonomy.setdefault(w, {}).setdefault(r, 0)
                taxonomy[w][r] += 1
        except Exception as e:
            result["error"] = str(e)
            result["partial"] = True
        return result

    def _tool_mempalace_get_aaak_spec(self) -> dict:
        return {"aaak_spec": AAAK_SPEC}

    def _tool_mempalace_search(
        self,
        query: str,
        limit: int = 5,
        wing: str = None,
        room: str = None,
        max_distance: float = 1.5,
    ) -> dict:
        from mempalace.searcher import search_memories
        from mempalace.query_sanitizer import sanitize_query
        limit = max(1, min(int(limit), _MAX_RESULTS))
        try:
            wing = self._sanitize_optional(wing, "wing")
            room = self._sanitize_optional(room, "room")
        except ValueError as e:
            return {"error": str(e)}
        sanitized = sanitize_query(query)
        result = search_memories(
            sanitized["clean_query"],
            palace_path=self._palace_path,
            wing=wing,
            room=room,
            n_results=limit,
            max_distance=max_distance,
        )
        if not isinstance(result, dict):
            return {"error": "search returned unexpected result"}
        if sanitized.get("was_sanitized"):
            result["query_sanitized"] = True
            result["sanitizer"] = {
                "method": sanitized["method"],
                "original_length": sanitized["original_length"],
                "clean_length": sanitized["clean_length"],
                "clean_query": sanitized["clean_query"],
            }
        return result

    def _tool_mempalace_check_duplicate(self, content: str, threshold: float = 0.9) -> dict:
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            res = col.query(
                query_texts=[content],
                n_results=5,
                include=["metadatas", "documents", "distances"],
            )
            dupes = []
            if res["ids"] and res["ids"][0]:
                for i, did in enumerate(res["ids"][0]):
                    dist = res["distances"][0][i]
                    sim = round(1 - dist, 3)
                    if sim >= threshold:
                        meta = res["metadatas"][0][i]
                        doc = res["documents"][0][i]
                        dupes.append({
                            "id": did,
                            "wing": meta.get("wing", "?"),
                            "room": meta.get("room", "?"),
                            "similarity": sim,
                            "content": (doc[:200] + "...") if len(doc) > 200 else doc,
                        })
            return {"is_duplicate": bool(dupes), "matches": dupes}
        except Exception as e:
            return {"error": str(e)}

    def _tool_mempalace_add_drawer(
        self, wing: str, room: str, content: str, source_file: str = None,
    ) -> dict:
        from mempalace.config import sanitize_name, sanitize_content
        try:
            wing = sanitize_name(wing, "wing")
            room = sanitize_name(room, "room")
            content = sanitize_content(content)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        col = self._get_col(create=True)
        if not col:
            return self._no_palace()
        drawer_id = (
            f"drawer_{wing}_{room}_"
            + hashlib.sha256((wing + room + content).encode()).hexdigest()[:24]
        )
        try:
            existing = col.get(ids=[drawer_id])
            if existing and existing["ids"]:
                return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
        except Exception:
            pass
        try:
            col.upsert(
                ids=[drawer_id],
                documents=[content],
                metadatas=[{
                    "wing": wing,
                    "room": room,
                    "source_file": source_file or "",
                    "chunk_index": 0,
                    "added_by": "hermes",
                    "filed_at": datetime.now().isoformat(),
                }],
            )
            self._invalidate_meta_cache()
            logger.info("Filed drawer: %s → %s/%s", drawer_id, wing, room)
            return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_mempalace_delete_drawer(self, drawer_id: str) -> dict:
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            existing = col.get(ids=[drawer_id])
            if not existing["ids"]:
                return {"success": False, "error": f"Drawer not found: {drawer_id}"}
            col.delete(ids=[drawer_id])
            self._invalidate_meta_cache()
            return {"success": True, "drawer_id": drawer_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_mempalace_get_drawer(self, drawer_id: str) -> dict:
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            res = col.get(ids=[drawer_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return {"error": f"Drawer not found: {drawer_id}"}
            meta = res["metadatas"][0]
            doc = res["documents"][0]
            return {
                "drawer_id": drawer_id,
                "content": doc,
                "wing": meta.get("wing", ""),
                "room": meta.get("room", ""),
                "metadata": meta,
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_mempalace_list_drawers(
        self, wing: str = None, room: str = None, limit: int = 20, offset: int = 0,
    ) -> dict:
        limit = max(1, min(int(limit), _MAX_RESULTS))
        offset = max(0, int(offset))
        try:
            wing = self._sanitize_optional(wing, "wing")
            room = self._sanitize_optional(room, "room")
        except ValueError as e:
            return {"error": str(e)}
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            conditions = []
            if wing:
                conditions.append({"wing": wing})
            if room:
                conditions.append({"room": room})
            where = None
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}
            kwargs = {"include": ["documents", "metadatas"], "limit": limit, "offset": offset}
            if where:
                kwargs["where"] = where
            res = col.get(**kwargs)
            drawers = []
            for i, did in enumerate(res["ids"]):
                meta = res["metadatas"][i]
                doc = res["documents"][i]
                drawers.append({
                    "drawer_id": did,
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                    "content_preview": (doc[:200] + "...") if len(doc) > 200 else doc,
                })
            return {"drawers": drawers, "count": len(drawers), "offset": offset, "limit": limit}
        except Exception as e:
            return {"error": str(e)}

    def _tool_mempalace_update_drawer(
        self, drawer_id: str, content: str = None, wing: str = None, room: str = None,
    ) -> dict:
        from mempalace.config import sanitize_name, sanitize_content
        if content is None and wing is None and room is None:
            return {"success": True, "drawer_id": drawer_id, "noop": True}
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
            if not existing["ids"]:
                return {"success": False, "error": f"Drawer not found: {drawer_id}"}
            old_meta = existing["metadatas"][0]
            old_doc = existing["documents"][0]
            new_doc = old_doc
            if content is not None:
                try:
                    new_doc = sanitize_content(content)
                except ValueError as e:
                    return {"success": False, "error": str(e)}
            new_meta = dict(old_meta)
            if wing is not None:
                try:
                    new_meta["wing"] = sanitize_name(wing, "wing")
                except ValueError as e:
                    return {"success": False, "error": str(e)}
            if room is not None:
                try:
                    new_meta["room"] = sanitize_name(room, "room")
                except ValueError as e:
                    return {"success": False, "error": str(e)}
            update_kwargs = {"ids": [drawer_id]}
            if content is not None:
                update_kwargs["documents"] = [new_doc]
            update_kwargs["metadatas"] = [new_meta]
            col.update(**update_kwargs)
            self._invalidate_meta_cache()
            return {
                "success": True, "drawer_id": drawer_id,
                "wing": new_meta.get("wing", ""), "room": new_meta.get("room", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -- Knowledge Graph ---------------------------------------------------

    def _tool_mempalace_kg_query(self, entity: str, as_of: str = None, direction: str = "both") -> dict:
        from mempalace.config import sanitize_kg_value
        try:
            entity = sanitize_kg_value(entity, "entity")
        except ValueError as e:
            return {"error": str(e)}
        if direction not in ("outgoing", "incoming", "both"):
            return {"error": "direction must be 'outgoing', 'incoming', or 'both'"}
        kg = self._get_kg()
        results = kg.query_entity(entity, as_of=as_of, direction=direction)
        return {"entity": entity, "as_of": as_of, "facts": results, "count": len(results)}

    def _tool_mempalace_kg_add(
        self, subject: str, predicate: str, object: str,
        valid_from: str = None, source_closet: str = None,
    ) -> dict:
        from mempalace.config import sanitize_name, sanitize_kg_value
        try:
            subject = sanitize_kg_value(subject, "subject")
            predicate = sanitize_name(predicate, "predicate")
            object = sanitize_kg_value(object, "object")
        except ValueError as e:
            return {"success": False, "error": str(e)}
        kg = self._get_kg()
        triple_id = kg.add_triple(
            subject, predicate, object,
            valid_from=valid_from, source_closet=source_closet,
        )
        return {
            "success": True, "triple_id": triple_id,
            "fact": f"{subject} → {predicate} → {object}",
        }

    def _tool_mempalace_kg_invalidate(
        self, subject: str, predicate: str, object: str, ended: str = None,
    ) -> dict:
        from mempalace.config import sanitize_name, sanitize_kg_value
        try:
            subject = sanitize_kg_value(subject, "subject")
            predicate = sanitize_name(predicate, "predicate")
            object = sanitize_kg_value(object, "object")
        except ValueError as e:
            return {"success": False, "error": str(e)}
        kg = self._get_kg()
        kg.invalidate(subject, predicate, object, ended=ended)
        return {
            "success": True,
            "fact": f"{subject} → {predicate} → {object}",
            "ended": ended or "today",
        }

    def _tool_mempalace_kg_timeline(self, entity: str = None) -> dict:
        from mempalace.config import sanitize_kg_value
        if entity is not None:
            try:
                entity = sanitize_kg_value(entity, "entity")
            except ValueError as e:
                return {"error": str(e)}
        kg = self._get_kg()
        results = kg.timeline(entity)
        return {"entity": entity or "all", "timeline": results, "count": len(results)}

    def _tool_mempalace_kg_stats(self) -> dict:
        kg = self._get_kg()
        return kg.stats()

    # -- Palace graph (implicit room links) -------------------------------

    def _tool_mempalace_traverse(self, start_room: str, max_hops: int = 2) -> dict:
        from mempalace.palace_graph import traverse
        max_hops = max(1, min(int(max_hops), 10))
        col = self._get_col()
        if not col:
            return self._no_palace()
        return traverse(start_room, col=col, max_hops=max_hops)

    def _tool_mempalace_graph_stats(self) -> dict:
        from mempalace.palace_graph import graph_stats
        col = self._get_col()
        if not col:
            return self._no_palace()
        return graph_stats(col=col)

    # -- Diary --------------------------------------------------------------

    def _tool_mempalace_diary_write(self, agent_name: str, entry: str, topic: str = "general") -> dict:
        from mempalace.config import sanitize_name, sanitize_content
        try:
            agent_name = sanitize_name(agent_name, "agent_name")
            entry = sanitize_content(entry)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        wing = f"wing_{agent_name.lower().replace(' ', '_')}"
        room = "diary"
        col = self._get_col(create=True)
        if not col:
            return self._no_palace()
        now = datetime.now()
        entry_id = (
            f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S%f')}_"
            + hashlib.sha256(entry.encode()).hexdigest()[:12]
        )
        try:
            col.add(
                ids=[entry_id],
                documents=[entry],
                metadatas=[{
                    "wing": wing,
                    "room": room,
                    "hall": "hall_diary",
                    "topic": topic,
                    "type": "diary_entry",
                    "agent": agent_name,
                    "filed_at": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                }],
            )
            self._invalidate_meta_cache()
            logger.info("Diary entry: %s → %s/diary/%s", entry_id, wing, topic)
            return {
                "success": True, "entry_id": entry_id,
                "agent": agent_name, "topic": topic, "timestamp": now.isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_mempalace_diary_read(self, agent_name: str, last_n: int = 10) -> dict:
        from mempalace.config import sanitize_name
        try:
            agent_name = sanitize_name(agent_name, "agent_name")
        except ValueError as e:
            return {"error": str(e)}
        last_n = max(1, min(int(last_n), 100))
        wing = f"wing_{agent_name.lower().replace(' ', '_')}"
        col = self._get_col()
        if not col:
            return self._no_palace()
        try:
            results = col.get(
                where={"$and": [{"wing": wing}, {"room": "diary"}]},
                include=["documents", "metadatas"],
                limit=10000,
            )
            if not results["ids"]:
                return {"agent": agent_name, "entries": [], "message": "No diary entries yet."}
            entries = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                entries.append({
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("filed_at", ""),
                    "topic": meta.get("topic", ""),
                    "content": doc,
                })
            entries.sort(key=lambda x: x["timestamp"], reverse=True)
            entries = entries[:last_n]
            return {
                "agent": agent_name, "entries": entries,
                "total": len(results["ids"]), "showing": len(entries),
            }
        except Exception as e:
            return {"error": str(e)}

    # -- Hook settings / filed-away ----------------------------------------

    def _tool_mempalace_hook_settings(
        self, silent_save: bool = None, desktop_toast: bool = None,
    ) -> dict:
        from mempalace.config import MempalaceConfig
        try:
            config = MempalaceConfig()
        except Exception as e:
            return {"success": False, "error": str(e)}
        changed = []
        if silent_save is not None:
            config.set_hook_setting("silent_save", bool(silent_save))
            changed.append(f"silent_save → {silent_save}")
        if desktop_toast is not None:
            config.set_hook_setting("desktop_toast", bool(desktop_toast))
            changed.append(f"desktop_toast → {desktop_toast}")
        try:
            config = MempalaceConfig()
        except Exception:
            pass
        out = {
            "success": True,
            "settings": {
                "silent_save": config.hook_silent_save,
                "desktop_toast": config.hook_desktop_toast,
            },
        }
        if changed:
            out["updated"] = changed
        return out

    def _tool_mempalace_memories_filed_away(self) -> dict:
        state_dir = Path.home() / ".mempalace" / "hook_state"
        ack_file = state_dir / "last_checkpoint"
        if not ack_file.is_file():
            return {"status": "quiet", "message": "No recent journal entry", "count": 0, "timestamp": None}
        try:
            data = json.loads(ack_file.read_text(encoding="utf-8"))
            ack_file.unlink(missing_ok=True)
            msgs = data.get("msgs", 0)
            return {
                "status": "ok",
                "message": f"\u2726 {msgs} messages tucked into drawers",
                "count": msgs,
                "timestamp": data.get("ts"),
            }
        except (json.JSONDecodeError, OSError):
            ack_file.unlink(missing_ok=True)
            return {
                "status": "error",
                "message": "\u2726 Journal entry filed in the palace",
                "count": 0,
                "timestamp": None,
            }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_memory_provider(MemPalaceProvider())
