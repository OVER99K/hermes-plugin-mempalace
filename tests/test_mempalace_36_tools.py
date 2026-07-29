import importlib.util
from pathlib import Path

import pytest
from mempalace.knowledge_graph import KnowledgeGraph


PLUGIN = Path(__file__).resolve().parents[1] / "__init__.py"
spec = importlib.util.spec_from_file_location("hermes_plugin_mempalace_test", PLUGIN)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
MemPalaceProvider = module.MemPalaceProvider


class FakeCollection:
    def __init__(self, rows):
        self.rows = list(rows)
        self.deleted_where = None
        self.calls = []

    def count(self):
        return len(self.rows)

    def get(self, ids=None, where=None, include=None, limit=None, offset=0):
        self.calls.append({"ids": ids, "where": where, "include": include, "limit": limit, "offset": offset})
        rows = self.rows
        if ids is not None:
            wanted = set(ids)
            rows = [row for row in rows if row[0] in wanted]
        if where:
            conditions = where.get("$and", [where])
            rows = [
                row for row in rows
                if all(row[2].get(next(iter(condition))) == next(iter(condition.values())) for condition in conditions)
            ]
        rows = rows[offset : offset + limit if limit is not None else None]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }

    def delete(self, where=None, ids=None):
        self.deleted_where = where
        if where:
            self.rows = [row for row in self.rows if any(row[2].get(k) != v for k, v in where.items())]


@pytest.fixture
def provider(tmp_path):
    p = MemPalaceProvider()
    p.initialize("test", agent_identity="tester")
    p._palace_path = str(tmp_path / "palace")
    return p


def test_schema_exposes_36_features_without_mcp(provider):
    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}
    assert len(schemas) == 26
    assert "mempalace_kg_supersede" in schemas
    assert "mempalace_checkpoint" in schemas
    assert "mempalace_delete_by_source" in schemas
    assert {"since", "before"} <= set(schemas["mempalace_list_drawers"]["parameters"]["properties"])
    assert {"source_file", "min_similarity"} <= set(schemas["mempalace_search"]["parameters"]["properties"])
    assert {"valid_to", "confidence", "source_file", "source_drawer_id", "adapter_name"} <= set(
        schemas["mempalace_kg_add"]["parameters"]["properties"]
    )


def test_list_drawers_prefers_authored_at_and_applies_half_open_window(provider):
    provider._col = FakeCollection([
        ("old", "old content", {"wing": "agent", "room": "conversations", "authored_at": "2026-01-10T09:00:00Z", "filed_at": "2026-07-01T00:00:00Z"}),
        ("middle", "middle content", {"wing": "agent", "room": "conversations", "authored_at": "2026-02-10T09:00:00Z", "filed_at": "2026-07-01T00:00:00Z"}),
        ("boundary", "boundary content", {"wing": "agent", "room": "conversations", "authored_at": "2026-03-01T00:00:00Z", "filed_at": "2026-07-01T00:00:00Z"}),
    ])
    result = provider._tool_mempalace_list_drawers(
        wing="agent", room="conversations",
        since="2026-02-01", before="2026-03-01",
    )
    assert [item["drawer_id"] for item in result["drawers"]] == ["middle"]
    assert result["total"] == 1
    assert provider._col.calls[0]["include"] == ["metadatas"]
    assert provider._col.calls[-1]["ids"] == ["middle"]
    assert provider._col.calls[-1]["include"] == ["documents"]


def test_delete_by_source_is_dry_run_first(provider, monkeypatch):
    provider._col = FakeCollection([
        ("one", "a", {"wing": "agent", "room": "research", "source_file": "bad.jsonl"}),
        ("two", "b", {"wing": "agent", "room": "research", "source_file": "keep.md"}),
    ])
    monkeypatch.setattr("mempalace.palace.get_closets_collection", lambda *a, **k: None)
    preview = provider._tool_mempalace_delete_by_source("bad.jsonl")
    assert preview["dry_run"] is True
    assert preview["match_count"] == 1
    assert provider._col.count() == 2
    applied = provider._tool_mempalace_delete_by_source("bad.jsonl", dry_run=False)
    assert applied["deleted"] == 1
    assert provider._col.count() == 1


def test_delete_by_source_aborts_before_drawers_when_closet_cleanup_fails(provider, monkeypatch):
    class FailingClosets:
        def get(self, **kwargs):
            return {"ids": ["closet-one"]}

        def delete(self, **kwargs):
            raise RuntimeError("closet delete failed")

    provider._col = FakeCollection([
        ("one", "a", {"wing": "agent", "room": "research", "source_file": "bad.jsonl"}),
    ])
    monkeypatch.setattr("mempalace.palace.get_closets_collection", lambda *a, **k: FailingClosets())
    result = provider._tool_mempalace_delete_by_source("bad.jsonl", dry_run=False)
    assert result["success"] is False
    assert result["deleted"] == 0
    assert provider._col.deleted_where is None
    assert provider._col.count() == 1


def test_checkpoint_batches_added_duplicates_errors_and_diary(provider, monkeypatch):
    monkeypatch.setattr(provider, "_tool_mempalace_check_duplicate", lambda content, threshold=0.9: {"is_duplicate": content == "duplicate", "matches": [{"id": "existing"}]})
    monkeypatch.setattr(provider, "_tool_mempalace_add_drawer", lambda wing, room, content, source_file=None: {"success": True, "drawer_id": content, "source_file": source_file})
    monkeypatch.setattr(provider, "_tool_mempalace_diary_write", lambda agent_name, entry, topic="general": {"success": True, "agent": agent_name, "topic": topic})
    result = provider._tool_mempalace_checkpoint(
        [
            {"wing": "agent", "room": "facts", "content": "new", "source_file": "source.md"},
            {"wing": "agent", "room": "facts", "content": "duplicate"},
            {"wing": "agent", "room": "facts", "content": ""},
        ],
        diary={"agent_name": "tester", "entry": "checkpointed", "topic": "tests"},
    )
    assert [item["drawer_id"] for item in result["added"]] == ["new"]
    assert len(result["duplicates"]) == 1
    assert len(result["errors"]) == 1
    assert result["diary"]["topic"] == "tests"


def test_duplicate_thresholds_are_bounded(provider):
    assert "between 0.0 and 1.0" in provider._tool_mempalace_check_duplicate("x", threshold=1.1)["error"]
    assert "between 0.0 and 1.0" in provider._tool_mempalace_checkpoint([], dedup_threshold=-0.1)["error"]


def test_kg_supersede_is_atomic_and_preserves_provenance(provider, tmp_path):
    kg = KnowledgeGraph(db_path=str(tmp_path / "knowledge_graph.sqlite3"))
    provider._kg = kg
    kg.add_triple("runtime", "uses_model", "old", valid_from="2026-01-01T00:00:00Z")
    result = provider._tool_mempalace_kg_supersede(
        "runtime", "uses_model", "old", "new",
        at="2026-02-01T00:00:00Z", confidence=0.95,
        source_file="decision.md", source_drawer_id="drawer-1", adapter_name="hermes",
    )
    assert result["success"] is True
    at_boundary = kg.query_entity("runtime", as_of="2026-02-01T00:00:00Z", direction="outgoing")
    assert [fact["object"] for fact in at_boundary if fact["predicate"] == "uses_model"] == ["new"]
    assert at_boundary[0]["confidence"] == pytest.approx(0.95)
    kg.close()


def test_kg_supersede_forwards_provenance(provider):
    class CapturingKG:
        def __init__(self):
            self.kwargs = None

        def supersede(self, *args, **kwargs):
            self.kwargs = kwargs
            return "triple-1"

    kg = CapturingKG()
    provider._kg = kg
    result = provider._tool_mempalace_kg_supersede(
        "runtime", "uses_model", "old", "new",
        source_file="decision.md", source_drawer_id="drawer-1", adapter_name="hermes",
    )
    assert result["success"] is True
    assert kg.kwargs["source_file"] == "decision.md"
    assert kg.kwargs["source_drawer_id"] == "drawer-1"
    assert kg.kwargs["adapter_name"] == "hermes"
