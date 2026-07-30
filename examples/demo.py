from __future__ import annotations

import json
from pathlib import Path

from mimemory import MemoryService


def main() -> None:
    service = MemoryService.local(Path(".demo-memory"))
    service.remember_profile("The user trains every Tuesday and Thursday.", source_id="profile-1")
    service.ingest_text(
        "I put my blue training bag in the car. It contains a jersey and spare shoes.",
        source_id="dialogue-1",
        session_id="training-handoff",
    )
    service.remember_procedure(
        "the user asks about training preparation",
        ["list the required equipment", "state where the bag is", "cite the supporting memory"],
        constraints=["do not invent missing equipment"],
        validation=["every item must have provenance"],
    )
    context = service.recall("Where is my training bag and what is in it?")
    print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

