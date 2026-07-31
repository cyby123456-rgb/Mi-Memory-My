"""OpenAI-compatible providers for the paper-faithful runtime.

Credentials are intentionally read only from the process environment. This
module never writes them to disk or includes them in traces.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


class ProviderConfigurationError(ProviderError):
    pass


class RemoteCallNotApproved(ProviderError):
    pass


@dataclass(frozen=True, slots=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str
    json_mode: bool = False

    @classmethod
    def from_environment(cls, prefix: str) -> "Endpoint":
        key = os.getenv(f"{prefix}_API_KEY", "")
        base = os.getenv(f"{prefix}_API_BASE", "")
        model = os.getenv(f"{prefix}_MODEL", "")
        missing = [name for name, value in (("API_KEY", key), ("API_BASE", base), ("MODEL", model)) if not value]
        if missing:
            raise ProviderConfigurationError(f"{prefix} is missing {', '.join(missing)}")
        return cls(base.rstrip("/"), key, model, os.getenv(f"{prefix}_JSON_MODE") == "1")


@dataclass(frozen=True, slots=True)
class PaperModelRoles:
    """Separate model authorities, matching the role separation in Table 27."""

    extraction: Endpoint
    embedding: Endpoint
    evaluator: Endpoint
    embedding_dimensions: int

    @classmethod
    def from_environment(cls) -> "PaperModelRoles":
        if os.getenv("MIMEMORY_REMOTE_MODELS_ENABLED") != "1":
            raise ProviderConfigurationError("set MIMEMORY_REMOTE_MODELS_ENABLED=1 to enable remote model calls")
        dimensions = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024"))
        if dimensions < 1:
            raise ProviderConfigurationError("OPENAI_EMBEDDING_DIMENSIONS must be positive")
        return cls(
            extraction=Endpoint.from_environment("OPENAI"),
            embedding=Endpoint.from_environment("OPENAI_EMBEDDING"),
            evaluator=Endpoint.from_environment("EVALUATOR"),
            embedding_dimensions=dimensions,
        )


class ChatProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = 0.0) -> str: ...


class EmbeddingProvider(Protocol):
    def embed(self, inputs: list[str], *, model: str | None = None) -> list[list[float]]: ...


class OpenAICompatibleClient:
    def __init__(self, endpoint: Endpoint, *, timeout_seconds: float = 90.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if os.getenv("MIMEMORY_LIVE_PROVIDER_APPROVED") != "1":
            raise RemoteCallNotApproved(
                "remote provider calls require explicit approval; set MIMEMORY_LIVE_PROVIDER_APPROVED=1 for one approved run"
            )
        request = Request(
            f"{self.endpoint.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.endpoint.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"provider returned HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderError(f"provider request failed: {exc.reason if isinstance(exc, URLError) else exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("provider returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderError("provider response must be a JSON object")
        return value

    def complete(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = 0.0) -> str:
        payload: dict[str, Any] = {"model": model or self.endpoint.model, "messages": messages, "temperature": temperature}
        if self.endpoint.json_mode:
            # OpenAI-compatible JSON mode constrains transport syntax; semantic
            # contracts are still validated by the consuming runtime.
            payload["response_format"] = {"type": "json_object"}
        value = self._post(
            "/chat/completions",
            payload,
        )
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("provider response has no chat completion content") from exc
        if not isinstance(content, str):
            raise ProviderError("chat completion content must be text")
        return content

    def embed(self, inputs: list[str], *, model: str | None = None) -> list[list[float]]:
        if not inputs:
            return []
        vectors: list[list[float]] = []
        # The configured compatible endpoint accepts at most ten inputs per
        # request. Keep the original sequence when joining the response batches.
        for start in range(0, len(inputs), 10):
            batch = inputs[start:start + 10]
            value = self._post("/embeddings", {"model": model or self.endpoint.model, "input": batch})
            try:
                rows = sorted(value["data"], key=lambda item: int(item["index"]))
                batch_vectors = [item["embedding"] for item in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("provider response has no usable embeddings") from exc
            if len(batch_vectors) != len(batch) or any(not isinstance(vector, list) for vector in batch_vectors):
                raise ProviderError("embedding response count does not match input")
            vectors.extend([[float(item) for item in vector] for vector in batch_vectors])
        return vectors


def json_from_completion(content: str) -> dict[str, Any]:
    """Parse a structured model response, accepting an optional fenced block."""

    stripped = content.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            stripped = match.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError("model did not return the required JSON object") from exc
    if not isinstance(value, dict):
        raise ProviderError("model JSON response must be an object")
    return value
