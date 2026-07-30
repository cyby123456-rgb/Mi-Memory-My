from __future__ import annotations

import json
import os
import re
import math
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Protocol

from .models import DiagnosticTrace, MemoryLayer, MemoryRecord, MemoryStatus, utc_now


class MemoryStore(Protocol):
    def put(self, record: MemoryRecord) -> MemoryRecord: ...

    def get(self, record_id: str) -> MemoryRecord | None: ...

    def list(self, *, include_inactive: bool = False) -> list[MemoryRecord]: ...

    def delete(self, record_id: str) -> bool: ...

    def append_raw(self, content: str, *, session_id: str | None = None) -> Path: ...

    def write_trace(self, trace: DiagnosticTrace) -> Path: ...


class LiteMemStore:
    """Markdown/JSON-frontmatter memory store inspired by LiteMem.

    The frontmatter is JSON, which is valid YAML, so files stay dependency-free
    while remaining readable by standard YAML tooling.
    """

    _LAYER_DIR = {
        MemoryLayer.L0: "entity",
        MemoryLayer.L1: "sessions",
        MemoryLayer.L2: "user/profile",
        MemoryLayer.SM: "sessions/current",
        MemoryLayer.PROCEDURE: "knowledge/skill",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in set(self._LAYER_DIR.values()) | {"daily", "knowledge/learning", "traces"}:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        corrections = self.root / "knowledge/learning/corrections.md"
        if not corrections.exists():
            self._atomic_write(corrections, "# Corrections\n\n<!-- SUMMARY_END -->\n")
        self._index_path = self.root / "_index.json"
        if not self._index_path.exists():
            self._write_index({})
            self._write_human_index({})

    def _read_index(self) -> dict[str, str]:
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return self.rebuild_index()

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            stream.write(content)
            temp_name = stream.name
        os.replace(temp_name, path)

    def _write_index(self, index: dict[str, str]) -> None:
        self._atomic_write(self._index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")

    def _write_human_index(self, index: dict[str, str]) -> None:
        lines = ["# LiteMem index", "", "Active memory summary:", ""]
        for record_id, path in sorted(index.items(), key=lambda item: item[1]):
            record = self.get(record_id) if hasattr(self, "_index_path") else None
            suffix = f" -- {record.summary}" if record and record.status is MemoryStatus.ACTIVE else ""
            lines.append(f"- `{record_id}`: [{path}]({path}){suffix}")
        self._atomic_write(self.root / "_index.md", "\n".join(lines) + "\n")

    def _relative_path(self, record: MemoryRecord) -> Path:
        # Table 31 uses singleton user files for high-priority persistent context.
        if record.layer is MemoryLayer.L2 and record.metadata.get("kind") == "style":
            return Path("user/style.md")
        if record.layer is MemoryLayer.L2 and record.metadata.get("kind") == "profile":
            return Path("user/profile.md")
        folder = self._LAYER_DIR[record.layer]
        return Path(folder) / f"{record.id}.md"

    def _serialize(self, record: MemoryRecord) -> str:
        metadata = record.to_dict()
        content = metadata.pop("content")
        frontmatter = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        return f"---\n{frontmatter}\n---\n\n{content.rstrip()}\n\n<!-- SUMMARY_END -->\n"

    def _deserialize(self, path: Path) -> MemoryRecord:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n(?:\n)?(.*)\Z", text, re.DOTALL)
        if not match:
            raise ValueError(f"invalid LiteMem record: {path}")
        metadata = json.loads(match.group(1))
        metadata["content"] = match.group(2).replace("<!-- SUMMARY_END -->", "").rstrip()
        return MemoryRecord.from_dict(metadata)

    def put(self, record: MemoryRecord) -> MemoryRecord:
        index = self._read_index()
        old_relative = Path(index[record.id]) if record.id in index else None
        relative = self._relative_path(record)
        # A singleton path must have exactly one index owner after replacement.
        if relative in {Path("user/profile.md"), Path("user/style.md")}:
            for existing_id, existing_path in list(index.items()):
                if Path(existing_path) == relative and existing_id != record.id:
                    index.pop(existing_id)
        self._atomic_write(self.root / relative, self._serialize(record))
        if old_relative and old_relative != relative:
            old_path = self.root / old_relative
            if old_path.exists():
                old_path.unlink()
        index[record.id] = relative.as_posix()
        self._write_index(index)
        self._write_human_index(index)
        self._append_audit("put", record.id, {"path": relative.as_posix(), "status": record.status.value})
        if record.metadata.get("kind") == "correction":
            self._append_correction(record)
        return record

    def _append_correction(self, record: MemoryRecord) -> None:
        path = self.root / "knowledge/learning/corrections.md"
        existing = path.read_text(encoding="utf-8").replace("<!-- SUMMARY_END -->", "").rstrip()
        marker = f"- `{record.id}`: {record.summary}"
        if marker not in existing:
            self._atomic_write(path, f"{existing}\n{marker}\n\n<!-- SUMMARY_END -->\n")

    def get(self, record_id: str) -> MemoryRecord | None:
        relative = self._read_index().get(record_id)
        if not relative:
            return None
        path = self.root / relative
        if not path.exists():
            self.rebuild_index()
            return None
        return self._deserialize(path)

    def list(self, *, include_inactive: bool = False) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for record_id in self._read_index():
            record = self.get(record_id)
            if record is None:
                continue
            if include_inactive or record.status is MemoryStatus.ACTIVE:
                records.append(record)
        return sorted(records, key=lambda item: (item.created_at, item.id))

    def delete(self, record_id: str) -> bool:
        index = self._read_index()
        relative = index.pop(record_id, None)
        if relative is None:
            return False
        path = self.root / relative
        if path.exists():
            path.unlink()
        self._write_index(index)
        self._write_human_index(index)
        self._append_audit("delete", record_id, {})
        return True

    def rebuild_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for path in self.root.rglob("*.md"):
            if path.name == "_index.md" or "daily" in path.parts or "traces" in path.parts:
                continue
            try:
                record = self._deserialize(path)
            except (ValueError, json.JSONDecodeError):
                continue
            index[record.id] = path.relative_to(self.root).as_posix()
        self._write_index(index)
        self._write_human_index(index)
        return index

    def append_raw(self, content: str, *, session_id: str | None = None) -> Path:
        date = datetime.now(UTC).date().isoformat()
        path = self.root / "daily" / f"{date}.md"
        if not path.exists():
            path.write_text(f"# Daily log: {date}\n\n", encoding="utf-8")
        timestamp = utc_now()
        session = f" session={session_id}" if session_id else ""
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"- [{timestamp}]{session} {content.strip()}\n")
        return path

    def write_profile(self, content: str, *, style: bool = False) -> MemoryRecord:
        """Replace, rather than append, the public singleton profile/style files."""
        kind = "style" if style else "profile"
        target = self.root / ("user/style.md" if style else "user/profile.md")
        old = next((record for record in self.list(include_inactive=True) if self._relative_path(record) == target.relative_to(self.root)), None)
        record = MemoryRecord(content=content, layer=MemoryLayer.L2, id=old.id if old else f"user-{kind}",
            importance=1.0, metadata={"kind": kind})
        return self.put(record)

    def active_context(self) -> list[MemoryRecord]:
        """Always prepend style then profile before any lazily routed evidence."""
        order = {"style": 0, "profile": 1}
        return sorted((record for record in self.list() if record.metadata.get("kind") in order),
                      key=lambda record: order[record.metadata["kind"]])

    @staticmethod
    def importance_score(record: MemoryRecord, *, now: datetime | None = None) -> float:
        """Eq. 21's observable ingredients: importance, recency and access, penalised by skips."""
        now = now or datetime.now(UTC)
        try:
            age_hours = max(0.0, (now - datetime.fromisoformat(record.last_accessed_at)).total_seconds() / 3600)
        except ValueError:
            age_hours = 0.0
        return record.importance * math.exp(-age_hours / 720.0) + 0.02 * record.access_count - 0.01 * record.skip_count

    def route_file_native(self, query: str, *, daily_line_window: int = 40, limit: int = 12) -> list[MemoryRecord]:
        """File-native Eq. 20 router: grep-like matching, lazy daily windows, then priority context."""
        terms = {term for term in re.findall(r"\w+", query.casefold()) if term}
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.list():
            haystack = f"{record.title} {record.summary} {record.content}".casefold()
            lexical = sum(term in haystack for term in terms)
            if lexical:
                scored.append((lexical + self.importance_score(record), record))
        # Daily files are read only around matching lines, never as whole logs.
        for daily in (self.root / "daily").glob("*.md"):
            lines = daily.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if terms and any(term in line.casefold() for term in terms):
                    start, end = max(0, index - daily_line_window // 2), min(len(lines), index + daily_line_window // 2)
                    window = "\n".join(lines[start:end])
                    scored.append((1.0, MemoryRecord(content=window, layer=MemoryLayer.SM, title=f"daily:{daily.stem}", metadata={"daily_path": str(daily), "line_window": [start + 1, end]})))
                    break
        prefixes = self.active_context()
        prefix_ids = {record.id for record in prefixes}
        ordered = [record for _, record in sorted(scored, key=lambda item: (-item[0], item[1].id)) if record.id not in prefix_ids]
        return [*prefixes, *ordered[:limit]]

    def write_trace(self, trace: DiagnosticTrace) -> Path:
        path = self.root / "traces" / f"{trace.query_id}.json"
        self._atomic_write(path, json.dumps(trace.to_dict(), ensure_ascii=False, indent=2) + "\n")
        return path

    def _append_audit(self, action: str, record_id: str, details: dict[str, object]) -> None:
        event = {"timestamp": utc_now(), "action": action, "record_id": record_id, "details": details}
        with (self.root / "audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def touch_access(self, records: Iterable[MemoryRecord]) -> None:
        now = utc_now()
        for record in records:
            record.access_count += 1
            record.last_accessed_at = now
            self.put(record)
