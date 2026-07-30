"""LiteMem capture, consolidation, atomic indexing, and optional Git audit."""

from __future__ import annotations

from typing import Any

from .git_provenance import GitProvenance
from .models import MemoryLayer, MemoryRecord, MemoryStatus
from .providers import ProviderError, json_from_completion
from .storage import LiteMemStore


CONSOLIDATION_PROMPT = """You are LiteMem idle consolidation. Given active memory summaries and raw logs,
return JSON only: {"actions": [{"action": "archive|deprecate|profile|promote|correction", "id": str, "content": str|null}]}. Every action must cite an input id. Do not remove correction or forget constraints."""


class LiteMemLifecycle:
    def __init__(self, store: LiteMemStore, consolidator: Any, *, git_enabled: bool = False) -> None:
        self.store, self.consolidator, self.git = store, consolidator, GitProvenance(store.root) if git_enabled else None

    def consolidate(self) -> dict[str, list[str]]:
        records = self.store.list(include_inactive=True)
        payload = [{"id": item.id, "layer": item.layer.value, "status": item.status.value, "summary": item.summary} for item in records]
        response = json_from_completion(self.consolidator.complete([{"role": "system", "content": CONSOLIDATION_PROMPT}, {"role": "user", "content": str(payload)}]))
        actions = response.get("actions")
        if not isinstance(actions, list): raise ProviderError("consolidator must return actions")
        changed: list[str] = []
        by_id = {item.id: item for item in records}
        for action in actions:
            if not isinstance(action, dict) or action.get("id") not in by_id: raise ProviderError("consolidator cited an unknown id")
            record = by_id[action["id"]]
            if record.metadata.get("kind") in {"correction", "constraint"}: continue
            if action.get("action") == "archive": record.status = MemoryStatus.ARCHIVED
            elif action.get("action") == "deprecate": record.status = MemoryStatus.DEPRECATED
            elif action.get("action") == "profile" and isinstance(action.get("content"), str):
                self.store.write_profile(action["content"])
            elif action.get("action") == "promote" and isinstance(action.get("content"), str):
                self.store.put(MemoryRecord(content=action["content"], layer=MemoryLayer.L1, importance=max(record.importance, 0.75), metadata={"consolidated_from": record.id, "action": "promote"}))
            elif action.get("action") == "correction" and isinstance(action.get("content"), str):
                self.store.put(MemoryRecord(content=action["content"], layer=MemoryLayer.PROCEDURE, importance=1.0, metadata={"kind": "correction", "consolidated_from": record.id}))
            else: continue
            self.store.put(record); changed.append(record.id)
        self.store.rebuild_index()
        commit = self.git.commit("LiteMem idle consolidation") if self.git and changed else None
        return {"changed": changed, "commit": [commit] if commit else []}
