#!/usr/bin/env python3
"""MemPalace → Obsidian Vault sync bridge.

Layer 3 sync script: reads from MemPalace's knowledge graph, diary, and
drawers, then writes mirrored markdown notes into the Obsidian vault.

Architecture:
  MemPalace KG  →  Gazetteer/<Entity>.md      (one note per entity)
  MemPalace diary  →  Sessions/<date>.md       (one note per diary date)
  MemPalace drawers  →  Projects/<room>.md     (one note per room)

Human-edited vault notes are preserved — only the `synced_from` tag and
structured fact blocks are overwritten on each run.

Usage:
  python3 sync_vault.py [--dry-run] [--verbose]

Requires:
  mempalace >= 3.6.0  (installed by hermes-plugin-mempalace)
  ChromaDB            (embedding model downloaded on first use)

Environment variables:
  MEMPALACE_PALACE_PATH  — MemPalace data directory (default: ~/.mempalace/palace)
  OBSIDIAN_VAULT_PATH    — Vault root (default: ~/Documents/Obsidian Vault)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PALACE_ENV = "MEMPALACE_PALACE_PATH"
DEFAULT_PALACE = os.path.expanduser("~/.mempalace/palace")
DEFAULT_VAULT = os.path.expanduser("~/Documents/Obsidian Vault")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_palace_path() -> str:
    env = os.environ.get(PALACE_ENV, "").strip()
    if env:
        return env
    cfg = os.path.expanduser("~/.mempalace/config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as f:
                d = json.load(f)
            if d.get("palace_path"):
                return d["palace_path"]
        except Exception:
            pass
    return DEFAULT_PALACE


def resolve_vault_path() -> str:
    env = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    return env or DEFAULT_VAULT


def ensure_dirs(vault: str) -> Dict[str, Path]:
    """Ensure vault subdirectories exist. Returns dict of path -> Path."""
    subdirs = {
        "gazetteer": "Gazetteer",
        "sessions": "Sessions",
        "projects": "Projects",
    }
    result = {}
    for key, name in subdirs.items():
        p = Path(vault) / name
        p.mkdir(parents=True, exist_ok=True)
        result[key] = p
    return result


def slugify(text: str) -> str:
    """Create a file-safe slug from text."""
    return text.strip().replace(" ", "-").replace("_", "-").lower()


def wikiname(text: str) -> str:
    """Create a wikilink name from text."""
    return text.strip()


def safe_path(text: str) -> str:
    """Create a safe filename from entity text."""
    return text.strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def fmt_date(ts: str) -> str:
    """Format an ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


def markdown_escape(text: str) -> str:
    """Escape pipe characters for markdown tables."""
    return text.replace("|", "\\|")



def sync_gazetteer(palace_path: str, vault_dir: Path, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Sync KG entities and triples to individual notes in the Gazetteer folder."""
    from mempalace import knowledge_graph

    stats = {"entities_synced": 0, "notes_written": 0, "notes_updated": 0, "errors": []}
    kg_path = os.path.join(palace_path, "kg.db")

    if not os.path.exists(kg_path):
        print(f"[SYNC] KG database not found at {kg_path} — no Gazetteer to sync")
        return stats

    kg = knowledge_graph.KnowledgeGraph(kg_path)

    try:
        # Get all entities and build an id->display_name map
        entity_rows = kg.dump_rows("entities", limit=10000)
        triples = kg.dump_rows("triples", limit=10000)

        if not entity_rows:
            print("[SYNC] No entities found in KG")
            return stats

        # Build id->display_name mapping (KG stores lowercase ids internally)
        id_to_name: Dict[str, str] = {}
        entity_names: set = set()
        for r in entity_rows:
            eid = r.get("id", "")
            name = r.get("name", "")
            if eid and name:
                id_to_name[eid] = name
                entity_names.add(name)

        # Build a lookup of display-name -> triples
        entity_triples: Dict[str, List[Dict]] = {}
        for triple in triples:
            subj_id = triple.get("subject", "")
            obj_id = triple.get("object", "")
            subj_name = id_to_name.get(subj_id, subj_id)
            obj_name = id_to_name.get(obj_id, obj_id)

            entity_triples.setdefault(subj_name, []).append({
                "predicate": triple.get("predicate", ""),
                "object": obj_name,
                "valid_from": triple.get("valid_from", ""),
                "valid_to": triple.get("valid_to", ""),
                "confidence": triple.get("confidence", ""),
                "source_closet": triple.get("source_closet", ""),
                "current": triple.get("valid_to") is None,
            })

        for entity_row in entity_rows:
            name = entity_row.get("name", "").strip()
            if not name:
                continue
            stats["entities_synced"] += 1

            facts = entity_triples.get(name, [])

            # Also gather triples where this entity is the object (incoming)
            # and resolve the subject display name from id_to_name
            incoming = []
            for triple in triples:
                obj_id = triple.get("object", "")
                obj_display = id_to_name.get(obj_id, obj_id)
                if obj_display == name:
                    subj_id = triple.get("subject", "")
                    subj_display = id_to_name.get(subj_id, subj_id)
                    incoming.append({
                        "subject": subj_display,
                        "predicate": triple.get("predicate", ""),
                        "valid_from": triple.get("valid_from", ""),
                        "valid_to": triple.get("valid_to", ""),
                        "confidence": triple.get("confidence", ""),
                        "current": triple.get("valid_to") is None,
                    })

            # If no outgoing facts and no incoming, this might be a value entity
            # — skip creating a note for it
            if not facts and not incoming:
                continue

            # Build the note content
            note = build_entity_note(name, facts, incoming, entity_names, id_to_name)

            # Write the file
            filename = f"{safe_path(name)}.md"
            filepath = vault_dir / filename

            if dry_run:
                if filepath.exists():
                    print(f"[DRY-RUN] Would update: {filename}")
                else:
                    print(f"[DRY-RUN] Would create: {filename}")
                stats["notes_written"] += 1
                continue

            existing = ""
            needs_write = False
            if not filepath.exists():
                needs_write = True
            else:
                # Check if content changed
                existing = filepath.read_text(encoding="utf-8")
                if existing != note:
                    needs_write = True

            if needs_write:
                existed = filepath.exists()
                filepath.write_text(note, encoding="utf-8")
                stats["notes_written"] += 1
                stats["notes_updated"] += 1
                if verbose:
                    print(f"[SYNC] {'Updated' if existed else 'Created'}: {filename}")
            else:
                if verbose:
                    print(f"[SYNC] Unchanged: {filename}")

        # Remove notes for entities that no longer exist in KG
        known_filenames = {safe_path(name) for name in id_to_name.values()}
        for existing_file in vault_dir.glob("*.md"):
            fname = existing_file.stem
            if fname not in known_filenames:
                if dry_run:
                    print(f"[DRY-RUN] Would remove: {fname}")
                else:
                    existing_file.unlink()
                    if verbose:
                        print(f"[SYNC] Removed: {fname}")

    except Exception as e:
        stats["errors"].append(f"Gazetteer sync error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    finally:
        kg.close()

    return stats


def build_entity_note(
    name: str,
    outgoing: List[Dict],
    incoming: List[Dict],
    related_entities: set,
    id_to_name: Dict[str, str],
) -> str:
    """Build a markdown note for a KG entity."""

    lines = [
        f"---",
        f"tags:",
        f"  - entity",
        f"synced_from: mempalace_kg",
        f"---",
        f"",
        f"# {name}",
        f"",
        f"> **Source:** [[MemPalace]] knowledge graph — Layer 3 sync bridge",
        f"",
    ]

    # Relationships section
    if outgoing:
        lines.append(f"## Relationships")
        lines.append(f"")
        for t in outgoing:
            obj = t.get("object", "?")
            pred = t.get("predicate", "?")
            vf = t.get("valid_from", "")
            vt = t.get("valid_to", "")
            conf = t.get("confidence", "")

            # Build wikilink for object if it's a known entity
            obj_link = f"[[{wikiname(obj)}]]" if obj in related_entities else obj

            validity = ""
            if vf:
                validity = f" (from {vf})"
                if vt:
                    validity += f" → {vt}"

            lines.append(f"- **{pred}**: {obj_link}{validity}")
        lines.append(f"")

    if incoming:
        lines.append(f"## Referenced By")
        lines.append(f"")
        seen = set()
        for t in incoming:
            subj = t.get("subject", "?")
            pred = t.get("predicate", "?")
            subj_link = f"[[{wikiname(subj)}]]" if subj in related_entities else subj
            ref_str = f"{subj} → {pred}"
            if ref_str not in seen:
                seen.add(ref_str)
                lines.append(f"- **{subj_link}** → {pred} → [[{wikiname(name)}]]")
        lines.append(f"")

    # Temporal section (if any facts have valid_from/to)
    temporal = [t for t in outgoing if t.get("valid_from")]
    if temporal:
        lines.append(f"## Timeline")
        lines.append(f"")
        for t in temporal:
            vf = t.get("valid_from", "")
            vt = t.get("valid_to", "")
            pred = t.get("predicate", "")
            obj = t.get("object", "")
            status = "current" if not vt else "expired"
            lines.append(f"- [{vf}{' → ' + vt if vt else ' → now'}] {pred}: {obj} ({status})")
        lines.append(f"")

    # Properties table (for well-known predicates)
    known_preds = {"type", "full_name", "author", "stores_as", "lossless", "path",
                   "format", "uses", "demoted_reason", "compression_ratio", "model",
                   "provider", "purpose", "parent"}
    props = [t for t in outgoing if t.get("predicate") in known_preds]
    if props:
        lines.append(f"## Properties")
        lines.append(f"")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        for t in props:
            pred = markdown_escape(t.get("predicate", ""))
            obj = markdown_escape(str(t.get("object", "")))
            lines.append(f"| {pred} | {obj} |")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Synced from MemPalace KG* ")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sync: Diary → Sessions
# ---------------------------------------------------------------------------

def sync_sessions(palace_path: str, vault_dir: Path, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Sync MemPalace diary entries to session summary notes."""
    from mempalace.palace import get_collection

    stats = {"entries_synced": 0, "notes_written": 0, "notes_updated": 0, "errors": []}
    col = get_collection(palace_path, create=False)
    if not col:
        print("[SYNC] Palace collection not available — no diary to sync")
        return stats

    try:
        # Get all diary entries
        results = col.get(
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            print("[SYNC] No diary entries found")
            return stats

        # Filter to diary room only
        entries_by_date: Dict[str, List[Dict]] = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("room") != "diary":
                continue
            date = meta.get("date", "unknown")
            entries_by_date.setdefault(date, []).append({
                "content": doc,
                "agent": meta.get("agent", ""),
                "topic": meta.get("topic", ""),
                "timestamp": meta.get("filed_at", ""),
                "hall": meta.get("hall", ""),
            })

        for date, entries in sorted(entries_by_date.items(), reverse=True):
            filename = f"{date}.md"
            filepath = vault_dir / filename

            note = build_session_note(date, entries)

            if dry_run:
                if filepath.exists():
                    print(f"[DRY-RUN] Would update session: {filename}")
                else:
                    print(f"[DRY-RUN] Would create session: {filename}")
                stats["notes_written"] += 1
                stats["entries_synced"] += len(entries)
                continue

            existed = filepath.exists()
            existing = filepath.read_text(encoding="utf-8") if existed else ""
            needs_write = not existed or existing != note

            if needs_write:
                filepath.write_text(note, encoding="utf-8")
                stats["notes_updated"] += 1
                if verbose:
                    print(f"[SYNC] {'Updated' if existed else 'Created'} session: {filename}")

            stats["notes_written"] += 1 if needs_write else 0
            stats["entries_synced"] += len(entries)

    except Exception as e:
        stats["errors"].append(f"Session sync error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    return stats


def build_session_note(date: str, entries: List[Dict]) -> str:
    """Build a markdown session summary note."""

    lines = [
        f"---",
        f"tags:",
        f"  - session-summary",
        f"  - mempalace-diary",
        f"synced_from: mempalace_diary",
        f"---",
        f"",
        f"# Session Summary — {date}",
        f"",
        f"> **Source:** [[MemPalace]] agent diary — Layer 3 sync bridge",
        f"",
    ]

    for entry in entries:
        agent = entry.get("agent", "?")
        topic = entry.get("topic", "general")
        ts = fmt_date(entry.get("timestamp", ""))
        content = entry.get("content", "")

        lines.append(f"## {agent} — {topic}")
        lines.append(f"*{ts}*")
        lines.append(f"")
        lines.append(f"{content}")
        lines.append(f"")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sync: Drawers → Projects
# ---------------------------------------------------------------------------

def sync_projects(palace_path: str, vault_dir: Path, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Sync MemPalace drawers by room into project notes."""
    from mempalace.palace import get_collection

    stats = {"drawers_synced": 0, "notes_written": 0, "notes_updated": 0, "errors": []}
    col = get_collection(palace_path, create=False)
    if not col:
        print("[SYNC] Palace collection not available — no drawers to sync")
        return stats

    try:
        # Get all drawers (excluding diary room which is handled separately)
        results = col.get(
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            print("[SYNC] No drawer content found (excluding diary)")
            return stats

        # Group by room (filter out diary entries which are handled in sync_sessions)
        by_room: Dict[str, List[Dict]] = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            room = meta.get("room", "uncategorized")
            if room == "diary":
                continue
            by_room.setdefault(room, []).append({
                "content": doc,
                "wing": meta.get("wing", ""),
                "type": meta.get("type", ""),
                "timestamp": meta.get("filed_at", ""),
                "hall": meta.get("hall", ""),
            })

        for room, drawers in sorted(by_room.items()):
            filename = f"{slugify(room)}.md"
            filepath = vault_dir / filename

            note = build_project_note(room, drawers)

            if dry_run:
                if filepath.exists():
                    print(f"[DRY-RUN] Would update project: {filename}")
                else:
                    print(f"[DRY-RUN] Would create project: {filename}")
                stats["notes_written"] += 1
                stats["drawers_synced"] += len(drawers)
                continue

            existed = filepath.exists()
            existing = filepath.read_text(encoding="utf-8") if existed else ""
            needs_write = not existed or existing != note

            if needs_write:
                filepath.write_text(note, encoding="utf-8")
                stats["notes_updated"] += 1
                if verbose:
                    print(f"[SYNC] {'Updated' if existed else 'Created'} project: {filename}")

            stats["notes_written"] += 1 if needs_write else 0
            stats["drawers_synced"] += len(drawers)

    except Exception as e:
        stats["errors"].append(f"Project sync error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    return stats


def build_project_note(room: str, drawers: List[Dict]) -> str:
    """Build a markdown note from drawers in a room."""

    lines = [
        f"---",
        f"tags:",
        f"  - project",
        f"  - mempalace-drawers",
        f"synced_from: mempalace_drawers",
        f"---",
        f"",
        f"# {room}",
        f"",
        f"> **Source:** [[MemPalace]] drawers — Layer 3 sync bridge",
        f"> **Room:** {room}",
        f"> **Drawers:** {len(drawers)}",
        f"",
    ]

    for drawer in drawers:
        ts = fmt_date(drawer.get("timestamp", ""))
        wing = drawer.get("wing", "")
        dtype = drawer.get("type", "")
        content = drawer.get("content", "")

        meta_parts = []
        if wing:
            meta_parts.append(f"wing: {wing}")
        if dtype:
            meta_parts.append(f"type: {dtype}")
        meta_str = ", ".join(meta_parts) if meta_parts else ""

        lines.append(f"## {ts} {f'[{meta_str}]' if meta_str else ''}")
        lines.append(f"")
        lines.append(f"{content}")
        lines.append(f"")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="MemPalace → Obsidian Vault sync bridge")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    palace_path = resolve_palace_path()
    vault_path = resolve_vault_path()

    if not os.path.exists(palace_path):
        print(f"[SYNC] Palace not found at {palace_path}")
        return 1

    vault = Path(vault_path)
    if not vault.exists():
        print(f"[SYNC] Vault not found at {vault_path}, creating...")
        vault.mkdir(parents=True, exist_ok=True)

    subdirs = ensure_dirs(str(vault))
    prefix = "[DRY-RUN] " if args.dry_run else "[SYNC] "

    print(f"{prefix}MemPalace palace: {palace_path}")
    print(f"{prefix}Obsidian vault:   {vault_path}")
    print()

    # Phase 1: Gazetteer
    print("=== Phase 1: Knowledge Graph → Gazetteer ===")
    gaz_stats = sync_gazetteer(palace_path, subdirs["gazetteer"], args.dry_run, args.verbose)
    print(f"  Entities synced: {gaz_stats.get('entities_synced', 0)}")
    print(f"  Notes updated:   {gaz_stats.get('notes_updated', 0)}")
    if gaz_stats.get("errors"):
        for e in gaz_stats["errors"]:
            print(f"  ERROR: {e}")

    # Phase 2: Diary → Sessions
    print()
    print("=== Phase 2: Diary → Sessions ===")
    sess_stats = sync_sessions(palace_path, subdirs["sessions"], args.dry_run, args.verbose)
    print(f"  Entries synced:  {sess_stats.get('entries_synced', 0)}")
    print(f"  Notes updated:   {sess_stats.get('notes_written', 0)}")

    # Phase 3: Drawers → Projects
    print()
    print("=== Phase 3: Drawers → Projects ===")
    proj_stats = sync_projects(palace_path, subdirs["projects"], args.dry_run, args.verbose)
    print(f"  Drawers synced:  {proj_stats.get('drawers_synced', 0)}")
    print(f"  Notes updated:   {proj_stats.get('notes_written', 0)}")

    # Summary
    print()
    total_updates = (gaz_stats.get("notes_updated", 0) + sess_stats.get("notes_updated", 0)
                     + proj_stats.get("notes_updated", 0))
    print(f"=== Sync Complete ===")
    print(f"  Total files touched: {total_updates}")
    if args.dry_run:
        print(f"  (dry-run — no files were written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
