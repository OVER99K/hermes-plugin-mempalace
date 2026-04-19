"""MemPalace memory provider for Hermes.

Local-first semantic memory with ChromaDB vector search, knowledge graph,
agent diary, and structured wing/room/drawer retrieval.

Config (config.yaml):
  memory:
    provider: mempalace

Optional env vars:
  MEMPALACE_PALACE_PATH  — path to the palace data dir (default: ~/.mempalace/palace)
  MEMPALACE_WING         — default wing for storing turns (default: agent)
  MEMPALACE_N_RECALL     — results to fetch per prefetch (default: 5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_PALACE_PATH_ENV = "MEMPALACE_PALACE_PATH"
_DEFAULT_PALACE_SUBDIR = "palace"
_DEFAULT_HERMES_HOME_PALACE = os.path.expanduser("~/.mempalace/palace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_palace_path(hermes_home: str = "") -> str:
    env = os.environ.get(_PALACE_PATH_ENV, "").strip()
    if env:
        return env
    # Try config.json in ~/.mempalace
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
        self._write_queue: list[tuple[str, str]] = []
        self._write_lock = threading.Lock()
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
            f"{count} stored memories organised into wings and rooms.\n"
            f"Use mempalace_search to recall relevant context. "
            f"Use mempalace_add to store important facts, decisions, or observations. "
            f"Use mempalace_diary to write session notes."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = session_id or self._session_id
        with self._prefetch_lock:
            cached = self._prefetch_cache.get(key, "")
        if cached:
            with self._prefetch_lock:
                self._prefetch_cache.pop(key, None)
            return cached
        # Synchronous fallback if background prefetch hasn't completed
        return self._do_search(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        key = session_id or self._session_id
        def _bg():
            result = self._do_search(query)
            with self._prefetch_lock:
                self._prefetch_cache[key] = result
        threading.Thread(target=_bg, daemon=True).start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Only store for primary agent context to avoid cron/subagent noise
        if self._agent_context != "primary":
            return
        entry = f"User: {user_content[:400]}\nAssistant: {assistant_content[:800]}"
        threading.Thread(
            target=self._store_drawer,
            args=(self._wing, "conversations", entry),
            daemon=True,
        ).start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._agent_context != "primary" or not messages:
            return
        # Write session summary to diary
        summary_parts = []
        for m in messages[-20:]:  # last 20 messages
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
                target=self._write_diary,
                args=(self._agent_identity, "\n".join(summary_parts[-10:])),
                daemon=True,
            ).start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        # Mirror built-in memory writes into MemPalace
        if action in ("add", "replace") and content:
            room = "memory" if target == "memory" else "user-profile"
            threading.Thread(
                target=self._store_drawer,
                args=(self._wing, room, content),
                daemon=True,
            ).start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # Distill key facts before compression discards messages
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
                target=self._store_drawer,
                args=(self._wing, "compressed-context", combined),
                daemon=True,
            ).start()
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mempalace_search",
                "description": (
                    "Semantically search your MemPalace memory store. "
                    "Returns the most relevant stored memories matching the query. "
                    "Use for recalling facts, past decisions, research, or context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "wing": {"type": "string", "description": "Limit to a specific wing (optional)"},
                        "n": {"type": "integer", "description": "Number of results (default 5, max 20)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mempalace_add",
                "description": (
                    "Store a memory in MemPalace. "
                    "Use for facts, decisions, research findings, or anything worth remembering long-term. "
                    "wing organises by entity/project, room by topic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The memory content to store"},
                        "wing": {"type": "string", "description": "Entity or project name (e.g. 'nicolette', 'homelab')"},
                        "room": {"type": "string", "description": "Topic category (e.g. 'facts', 'decisions', 'research')"},
                    },
                    "required": ["content", "wing", "room"],
                },
            },
            {
                "name": "mempalace_diary",
                "description": "Write a diary entry for this agent session. Useful for logging what happened, decisions made, or lessons learned.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry": {"type": "string", "description": "Diary entry text"},
                        "topic": {"type": "string", "description": "Topic tag (default: general)"},
                    },
                    "required": ["entry"],
                },
            },
            {
                "name": "mempalace_status",
                "description": "Show MemPalace statistics: total drawers, wings, rooms.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "mempalace_kg_query",
                "description": "Query the MemPalace knowledge graph for relationships about an entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity to query (e.g. 'Commander', 'homelab')"},
                        "direction": {"type": "string", "description": "Relationship direction: both/out/in (default: both)"},
                    },
                    "required": ["entity"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "mempalace_search":
                return self._handle_search(args)
            elif tool_name == "mempalace_add":
                return self._handle_add(args)
            elif tool_name == "mempalace_diary":
                return self._handle_diary(args)
            elif tool_name == "mempalace_status":
                return self._handle_status()
            elif tool_name == "mempalace_kg_query":
                return self._handle_kg_query(args)
        except Exception as e:
            logger.exception("MemPalace tool error: %s", e)
            return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        logger.info("MemPalace provider shutting down")

    # -- Internal helpers --------------------------------------------------

    def _do_search(self, query: str, wing: str = None, n: int = None) -> str:
        try:
            from mempalace.searcher import search_memories
            results = search_memories(
                query,
                palace_path=self._palace_path,
                wing=wing,
                n_results=n or self._n_recall,
            )
            if results.get("error"):
                return ""
            return _fmt_results(results)
        except Exception as e:
            logger.warning("MemPalace search failed: %s", e)
            return ""

    def _store_drawer(self, wing: str, room: str, content: str) -> None:
        try:
            from mempalace.palace import get_collection
            import chromadb
            col = get_collection(self._palace_path, create=True)
            if not col:
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
        except Exception as e:
            logger.warning("MemPalace store_drawer failed: %s", e)

    def _write_diary(self, agent_name: str, entry: str, topic: str = "general") -> None:
        try:
            from mempalace.knowledge_graph import KnowledgeGraph
            import sqlite3
            # Diary entries stored as KG triples: agent -[diary_entry]-> content
            kg_db = os.path.join(os.path.dirname(self._palace_path), "knowledge_graph.sqlite3")
            kg = KnowledgeGraph(db_path=kg_db)
            kg.add(
                subject=agent_name,
                predicate="diary_entry",
                object_=f"[{topic}] {datetime.now().strftime('%Y-%m-%d %H:%M')} {entry[:500]}",
            )
        except Exception as e:
            # Fallback: store as a drawer
            logger.debug("KG diary write failed (%s), falling back to drawer", e)
            self._store_drawer(self._wing, "diary", f"[{topic}] {entry[:800]}")

    def _handle_search(self, args: Dict) -> str:
        query = args.get("query", "")
        wing = args.get("wing") or None
        n = min(int(args.get("n", self._n_recall)), 20)
        if not query:
            return json.dumps({"error": "query is required"})
        from mempalace.searcher import search_memories
        results = search_memories(
            query, palace_path=self._palace_path, wing=wing, n_results=n
        )
        if results.get("error"):
            return json.dumps(results)
        items = results.get("results", [])
        return json.dumps({
            "count": len(items),
            "results": [
                {
                    "text": r.get("text", "")[:600],
                    "wing": r.get("wing", ""),
                    "room": r.get("room", ""),
                    "score": round(1.0 - r.get("distance", 0), 3),
                }
                for r in items
            ],
        })

    def _handle_add(self, args: Dict) -> str:
        content = args.get("content", "").strip()
        wing = args.get("wing", self._wing).strip()
        room = args.get("room", "general").strip()
        if not content:
            return json.dumps({"error": "content is required"})
        self._store_drawer(wing, room, content)
        return json.dumps({"success": True, "wing": wing, "room": room})

    def _handle_diary(self, args: Dict) -> str:
        entry = args.get("entry", "").strip()
        topic = args.get("topic", "general")
        if not entry:
            return json.dumps({"error": "entry is required"})
        self._write_diary(self._agent_identity, entry, topic)
        return json.dumps({"success": True, "topic": topic})

    def _handle_status(self) -> str:
        try:
            from mempalace.palace import get_collection
            col = get_collection(self._palace_path, create=False)
            if not col:
                return json.dumps({"error": "palace not found"})
            count = col.count()
            # Get wing/room breakdown
            meta = col.get(include=["metadatas"])
            wings: dict = {}
            for m in (meta.get("metadatas") or []):
                w = m.get("wing", "?")
                r = m.get("room", "?")
                wings.setdefault(w, {}).setdefault(r, 0)
                wings[w][r] += 1
            return json.dumps({"total_drawers": count, "wings": wings})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_kg_query(self, args: Dict) -> str:
        entity = args.get("entity", "").strip()
        direction = args.get("direction", "both")
        if not entity:
            return json.dumps({"error": "entity is required"})
        try:
            from mempalace.knowledge_graph import KnowledgeGraph
            kg_db = os.path.join(os.path.dirname(self._palace_path), "knowledge_graph.sqlite3")
            kg = KnowledgeGraph(db_path=kg_db)
            triples = kg.query(entity, direction=direction)
            return json.dumps({"entity": entity, "triples": triples})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_memory_provider(MemPalaceProvider())
