from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .models import MemoryLayer
from .service import MemoryService
from .strategy import EvaluationReport


class MemoryAPIHandler(BaseHTTPRequestHandler):
    service: MemoryService
    server_version = "MiMemory/0.1"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write_json(self, status: int, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok", "version": "0.1.0"})
        elif path == "/memories":
            self._write_json(HTTPStatus.OK, [item.to_dict() for item in self.service.store.list(include_inactive=True)])
        elif path == "/strategy":
            self._write_json(HTTPStatus.OK, self.service.strategy)
        else:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            body = self._read_json()
            if path == "/ingest":
                records = self.service.ingest_text(
                    body["text"], source_id=body.get("source_id"), session_id=body.get("session_id")
                )
                self._write_json(HTTPStatus.CREATED, [item.to_dict() for item in records])
            elif path == "/memories":
                record = self.service.add_memory(
                    body["content"],
                    layer=MemoryLayer(body.get("layer", "L0")),
                    importance=float(body.get("importance", 0.5)),
                    confidence=float(body.get("confidence", 1.0)),
                    metadata=body.get("metadata", {}),
                )
                self._write_json(HTTPStatus.CREATED, record.to_dict())
            elif path == "/recall":
                self._write_json(HTTPStatus.OK, self.service.recall(body["query"]).to_dict())
            elif path == "/correct":
                self._write_json(HTTPStatus.OK, self.service.correct(body["record_id"], body["replacement"]).to_dict())
            elif path == "/forget":
                self._write_json(HTTPStatus.OK, self.service.forget(body["record_id"], reason=body.get("reason", "user_request")).to_dict())
            elif path == "/organize":
                self._write_json(HTTPStatus.OK, self.service.organize())
            elif path == "/strategy/evaluate":
                if self.service.strategy_manager is None:
                    raise RuntimeError("strategy manager is unavailable")
                candidate = self.service.strategy_manager.propose(body["changes"])
                report = EvaluationReport(**body["report"])
                decision = self.service.strategy_manager.evaluate(candidate, report)
                self.service.refresh_strategy()
                self._write_json(HTTPStatus.OK, decision.to_dict())
            elif path == "/strategy/rollback":
                if self.service.strategy_manager is None:
                    raise RuntimeError("strategy manager is unavailable")
                value = self.service.strategy_manager.rollback()
                self.service.refresh_strategy()
                self._write_json(HTTPStatus.OK, value)
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # Keep the zero-dependency server observable.
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(service: MemoryService, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("ConfiguredMemoryAPIHandler", (MemoryAPIHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mi-Memory HTTP API")
    parser.add_argument("--root", default=".mimemory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(MemoryService.local(args.root), args.host, args.port)
    print(f"Mi-Memory API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()

