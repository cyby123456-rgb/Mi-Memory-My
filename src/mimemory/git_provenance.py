"""Explicit Git provenance for LiteMem repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitProvenance:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _run(self, *args: str) -> str:
        completed = subprocess.run(["git", *args], cwd=self.root, check=True, text=True, capture_output=True)
        return completed.stdout.strip()

    def initialize(self) -> None:
        if not (self.root / ".git").exists():
            self._run("init")
        self._run("config", "user.name", "Mi-Memory")
        self._run("config", "user.email", "mimemory@local")

    def commit(self, message: str) -> str | None:
        self.initialize()
        self._run("add", "-A")
        if not self._run("status", "--porcelain"):
            return None
        self._run("commit", "-m", message)
        return self._run("rev-parse", "HEAD")

    def diff(self, revision: str = "HEAD") -> str:
        return self._run("diff", f"{revision}~1", revision)

