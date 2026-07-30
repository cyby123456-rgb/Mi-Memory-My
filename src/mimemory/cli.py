from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .api import create_server
from .models import MemoryLayer
from .service import MemoryService


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimemory", description="Mi-Memory clean-room runtime")
    parser.add_argument("--root", default=".mimemory", help="runtime data directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest a dialogue turn or session")
    ingest.add_argument("text")
    ingest.add_argument("--source-id")
    ingest.add_argument("--session-id")

    add = subparsers.add_parser("add", help="add a typed memory directly")
    add.add_argument("content")
    add.add_argument("--layer", choices=[item.value for item in MemoryLayer], default="L0")
    add.add_argument("--importance", type=float, default=0.5)
    add.add_argument("--confidence", type=float, default=1.0)

    recall = subparsers.add_parser("recall", help="retrieve and assemble evidence")
    recall.add_argument("query")

    list_parser = subparsers.add_parser("list", help="list memory records")
    list_parser.add_argument("--all", action="store_true", dest="include_inactive")

    correct = subparsers.add_parser("correct", help="supersede a record with a correction")
    correct.add_argument("record_id")
    correct.add_argument("replacement")

    forget = subparsers.add_parser("forget", help="mark a record forgotten")
    forget.add_argument("record_id")
    forget.add_argument("--reason", default="user_request")

    subparsers.add_parser("organize", help="apply decay and archival policy")

    serve = subparsers.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    service = MemoryService.local(Path(args.root))
    if args.command == "ingest":
        _print([item.to_dict() for item in service.ingest_text(args.text, source_id=args.source_id, session_id=args.session_id)])
    elif args.command == "add":
        _print(service.add_memory(args.content, layer=MemoryLayer(args.layer), importance=args.importance, confidence=args.confidence).to_dict())
    elif args.command == "recall":
        _print(service.recall(args.query).to_dict())
    elif args.command == "list":
        _print([item.to_dict() for item in service.store.list(include_inactive=args.include_inactive)])
    elif args.command == "correct":
        _print(service.correct(args.record_id, args.replacement).to_dict())
    elif args.command == "forget":
        _print(service.forget(args.record_id, reason=args.reason).to_dict())
    elif args.command == "organize":
        _print(service.organize())
    elif args.command == "serve":
        server = create_server(service, args.host, args.port)
        print(f"Mi-Memory API listening on http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()

