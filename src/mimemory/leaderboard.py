from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import threading
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .models import MemoryLayer, MemoryRecord, SourceRef, utc_now
from .memstack import MemStackRuntime
from .memstack import MemStackModels
from .config import default_strategy
from .providers import OpenAICompatibleClient, PaperModelRoles
from .storage import LiteMemStore
from .retrieval import HybridRetriever
from .service import MemoryService


class ContractError(ValueError):
    pass


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _timestamp_from_milliseconds(value: Any) -> str:
    if value is None:
        return utc_now()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("message timestamp must be Unix milliseconds")
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise ContractError("message timestamp is out of range") from exc


def _scope_digest(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


class ScopedServiceRegistry:
    """Creates one physical memory repository per benchmark user_id."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.scope_root = self.root / "scopes"
        self.scope_root.mkdir(parents=True, exist_ok=True)
        self._services: dict[str, MemoryService] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._registry_lock = threading.RLock()

    def scope_path(self, user_id: str) -> Path:
        return self.scope_root / _scope_digest(user_id)

    def service_for(self, user_id: str) -> MemoryService:
        digest = _scope_digest(user_id)
        with self._registry_lock:
            if digest not in self._services:
                self._services[digest] = MemoryService.local(self.scope_root / digest)
            return self._services[digest]

    def lock_for(self, user_id: str) -> threading.RLock:
        digest = _scope_digest(user_id)
        with self._registry_lock:
            return self._locks.setdefault(digest, threading.RLock())

    def touch_scope(self, user_id: str) -> None:
        path = self.scope_path(user_id) / ".scope.json"
        _atomic_json(path, {"user_id": user_id, "last_seen_at": utc_now()})

    def purge_expired(self, retention_days: int = 30, *, now: datetime | None = None) -> list[str]:
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
        removed: list[str] = []
        for scope in self.scope_root.iterdir():
            if not scope.is_dir():
                continue
            metadata_path = scope / ".scope.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                last_seen = datetime.fromisoformat(metadata["last_seen_at"])
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=UTC)
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                continue
            if last_seen < cutoff:
                shutil.rmtree(scope)
                removed.append(scope.name)
                with self._registry_lock:
                    self._services.pop(scope.name, None)
                    self._locks.pop(scope.name, None)
        return removed


class LeaderboardAdapter:
    def __init__(self, root: str | Path) -> None:
        self.registry = ScopedServiceRegistry(root)

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = _required_string(payload, "request_id")
        user_id = _required_string(payload, "user_id")
        session_id = _required_string(payload, "session_id")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ContractError("messages must be a non-empty array")

        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ContractError(f"messages[{index}] must be an object")
            role = _required_string(message, "role")
            if role not in {"user", "assistant"}:
                raise ContractError(f"messages[{index}].role must be user or assistant")
            content = _required_string(message, "content")
            timestamp_ms = message.get("timestamp")
            timestamp = _timestamp_from_milliseconds(timestamp_ms)
            normalized.append(
                {"role": role, "content": content, "timestamp": timestamp, "timestamp_ms": timestamp_ms}
            )

        canonical = json.dumps(
            {
                "session_id": session_id,
                "messages": [
                    {"role": item["role"], "content": item["content"], "timestamp": item["timestamp_ms"]}
                    for item in normalized
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        payload_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        scope = self.registry.scope_path(user_id)
        receipt = scope / "requests" / f"{hashlib.sha256(request_id.encode()).hexdigest()}.json"

        with self.registry.lock_for(user_id):
            if receipt.exists():
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                if previous.get("payload_digest") != payload_digest:
                    raise ContractError("request_id was already used with different messages")
            else:
                service = self.registry.service_for(user_id)
                for index, message in enumerate(normalized):
                    stable_id = hashlib.sha256(
                        f"{user_id}\0{request_id}\0{index}".encode("utf-8")
                    ).hexdigest()[:32]
                    record = MemoryRecord(
                        id=stable_id,
                        content=message["content"],
                        layer=MemoryLayer.L0,
                        created_at=message["timestamp"],
                        updated_at=message["timestamp"],
                        last_accessed_at=message["timestamp"],
                        sources=[
                            SourceRef(
                                source_id=f"{request_id}:{index}",
                                source_type="benchmark_message",
                                timestamp=message["timestamp"],
                            )
                        ],
                        metadata={
                            "request_id": request_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "message_index": index,
                            "role": message["role"],
                        },
                    )
                    service.store.put(record)
                _atomic_json(
                    receipt,
                    {
                        "request_id": request_id,
                        "payload_digest": payload_digest,
                        "stored_count": len(normalized),
                        "completed_at": utc_now(),
                    },
                )
            self.registry.touch_scope(user_id)
        return {
            "success": True,
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
        }


    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(payload, "query")
        user_id = _required_string(payload, "user_id")
        top_k = payload.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
            raise ContractError("top_k must be an integer between 1 and 1000")
        options = payload.get("options", [])
        if options is None:
            options = []
        if not isinstance(options, list) or any(not isinstance(item, str) for item in options):
            raise ContractError("options must be an array of strings")

        with self.registry.lock_for(user_id):
            service = self.registry.service_for(user_id)
            strategy = deepcopy(service.strategy)
            strategy["retrieval"]["top_k"] = top_k
            retrieval_query = query
            if options:
                retrieval_query += "\n" + "\n".join(options)
            result = HybridRetriever(strategy).retrieve(retrieval_query, service.store.list())
            hits = result.hits[:top_k]
            if hasattr(service.store, "touch_access"):
                service.store.touch_access(hit.record for hit in hits)
            self.registry.touch_scope(user_id)
        return {
            "data": [
                {
                    "id": hit.record.id,
                    "content": hit.record.content,
                    "score": hit.score,
                    "created_at": hit.record.created_at,
                }
                for hit in hits
            ]
        }


class PaperLeaderboardAdapter:
    """Leaderboard contract backed by the strict, model-driven MemStack runtime."""

    def __init__(self, runtime_factory: Any) -> None:
        self.runtime_factory = runtime_factory
        self._runtimes: dict[str, MemStackRuntime] = {}
        self._lock = threading.RLock()

    def _runtime(self, user_id: str) -> MemStackRuntime:
        if user_id not in self._runtimes:
            self._runtimes[user_id] = self.runtime_factory(user_id)
        return self._runtimes[user_id]

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = _required_string(payload, "request_id")
        user_id = _required_string(payload, "user_id")
        session_id = _required_string(payload, "session_id")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ContractError("messages must be a non-empty array")
        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ContractError(f"messages[{index}] must be an object")
            role = _required_string(message, "role")
            content = _required_string(message, "content")
            if role not in {"user", "assistant"}:
                raise ContractError(f"messages[{index}].role must be user or assistant")
            normalized.append({"source_id": f"{request_id}:{index}", "role": role, "content": content, "timestamp": _timestamp_from_milliseconds(message.get("timestamp"))})
        canonical = json.dumps({"session_id": session_id, "messages": normalized}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            runtime = self._runtime(user_id)
            receipt = runtime.store.root / "requests" / f"{hashlib.sha256(request_id.encode()).hexdigest()}.json"
            if receipt.exists():
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                if previous.get("payload_digest") != payload_digest:
                    raise ContractError("request_id was already used with different messages")
            else:
                runtime.ingest(normalized, user_id=user_id, session_id=session_id)
                _atomic_json(receipt, {"request_id": request_id, "payload_digest": payload_digest, "completed_at": utc_now()})
        return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(payload, "query")
        user_id = _required_string(payload, "user_id")
        top_k = payload.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ContractError("top_k must be a positive integer")
        options = payload.get("options", [])
        if options is None:
            options = []
        if not isinstance(options, list) or any(not isinstance(item, str) for item in options):
            raise ContractError("options must be an array of strings")
        with self._lock:
            bundle = self._runtime(user_id).retrieve(query + ("\n" + "\n".join(options) if options else ""), user_id=user_id)
        return {"data": [{"id": hit.record.id, "content": hit.record.content, "score": hit.score, "created_at": hit.record.created_at} for hit in bundle.evidence[:top_k]]}


class LeaderboardAPIHandler(BaseHTTPRequestHandler):
    adapter: LeaderboardAdapter
    api_token: str | None = None
    max_body_bytes = 10 * 1024 * 1024
    server_version = "MiMemoryLeaderboard/0.2"

    def _authorized(self) -> bool:
        if not self.api_token:
            return True
        authorization = self.headers.get("Authorization", "")
        x_api_key = self.headers.get("X-API-Key", "")
        return hmac.compare_digest(authorization, f"Bearer {self.api_token}") or hmac.compare_digest(
            x_api_key, self.api_token
        )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ContractError("invalid Content-Length") from exc
        if length <= 0:
            raise ContractError("request body is required")
        if length > self.max_body_bytes:
            raise ContractError("request body is too large")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ContractError("request body must be a JSON object")
        return value

    def _write_json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            self._write_json(HTTPStatus.OK, {"status": "ok", "service": "mi-memory-add-search"})
        else:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            payload = self._read_json()
            path = self.path.rstrip("/")
            if path in {"/add", "/v1/add"}:
                self._write_json(HTTPStatus.OK, self.adapter.add(payload))
            elif path in {"/search", "/v1/search"}:
                self._write_json(HTTPStatus.OK, self.adapter.search(payload))
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ContractError, json.JSONDecodeError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_leaderboard_server(
    adapter: LeaderboardAdapter,
    host: str = "0.0.0.0",
    port: int = 8765,
    *,
    api_token: str | None = None,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredLeaderboardAPIHandler",
        (LeaderboardAPIHandler,),
        {"adapter": adapter, "api_token": api_token},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent Memory Leaderboard Add/Search API")
    parser.add_argument("--root", default=os.getenv("MIMEMORY_ROOT", ".mimemory-eval"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    parser.add_argument("--retention-days", type=int, default=int(os.getenv("RETENTION_DAYS", "30")))
    parser.add_argument("--runtime", choices=("paper", "baseline"), default=os.getenv("MIMEMORY_RUNTIME", "paper"))
    parser.add_argument("--purge-only", action="store_true")
    args = parser.parse_args()
    if args.runtime == "paper":
        roles = PaperModelRoles.from_environment()
        def runtime_factory(user_id: str) -> MemStackRuntime:
            scope = Path(args.root) / "paper_scopes" / _scope_digest(user_id)
            models = MemStackModels(
                extractor=OpenAICompatibleClient(roles.extraction),
                planner=OpenAICompatibleClient(roles.extraction),
                reranker=OpenAICompatibleClient(roles.evaluator),
                embeddings=OpenAICompatibleClient(roles.embedding),
            )
            return MemStackRuntime(LiteMemStore(scope), default_strategy(), models)
        adapter: Any = PaperLeaderboardAdapter(runtime_factory)
        removed: list[str] = []
    else:
        adapter = LeaderboardAdapter(args.root)
        removed = adapter.registry.purge_expired(args.retention_days)
    if args.purge_only:
        print(json.dumps({"removed_scopes": removed}))
        return
    server = create_leaderboard_server(
        adapter,
        args.host,
        args.port,
        api_token=os.getenv("MIMEMORY_API_TOKEN"),
    )
    print(f"Mi-Memory Add/Search API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
