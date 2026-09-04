from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class NormalizedToolResult:
    """Provider-neutral view of an MCP result envelope."""

    body: Any
    ok: bool = True
    status: str = "success"
    error: str = ""
    artifacts: tuple[Any, ...] = ()
    provenance: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


ResultAdapter = Callable[[Any], NormalizedToolResult]

_TOOL_RESULT_ENVELOPE_KEYS = {
    "result",
    "success",
    "ok",
    "status",
    "error",
    "message",
    "query_info",
    "metrics",
    "artifacts",
    "provenance",
}

_EMPTY_COLLECTION_KEYS = {
    "data",
    "items",
    "matches",
    "records",
    "result_set",
    "results",
}
_EMPTY_COUNT_KEYS = {"count", "result_count", "total_count"}


def _is_explicit_empty_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "raw_text" in value and value["raw_text"] == "":
        return True
    if any(key in value and value[key] == 0 for key in _EMPTY_COUNT_KEYS):
        return True
    return any(
        key in value and isinstance(value[key], list) and not value[key]
        for key in _EMPTY_COLLECTION_KEYS
    )


def _status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "success": "succeeded",
        "completed": "succeeded",
        "complete": "succeeded",
        "done": "succeeded",
        "timeout": "timed_out",
        "timedout": "timed_out",
        "canceled": "cancelled",
        "retrying": "retry_wait",
        "retrying_wait": "retry_wait",
        "manual": "manual_review",
        "review": "manual_review",
        "dead-letter": "dead_letter",
        "deadletter": "dead_letter",
    }.get(normalized, normalized)


def normalize_generic_result(payload: Any) -> NormalizedToolResult:
    """Unwrap common MCP/Biomni envelopes without inventing provider fields."""
    if not isinstance(payload, dict):
        return NormalizedToolResult(body=payload, ok=not isinstance(payload, str))

    envelope = dict(payload)
    status = _status(envelope.get("status")) or "success"
    ok = envelope.get("ok", envelope.get("success", True)) is not False
    error_value = envelope.get("error") or envelope.get("message") or ""
    error = (
        str(error_value.get("message") or error_value.get("code") or error_value)
        if isinstance(error_value, dict)
        else str(error_value)
    )
    body: Any = envelope.get("result", envelope)
    result_container = body if isinstance(body, dict) else {}
    data_container: dict[str, Any] | None = None
    tool_result_containers: list[dict[str, Any]] = []
    empty_result = False

    # biomni.task.v1 puts the result envelope in result.data and the actual
    # internal-tool payload in data.output. Verifiers must see that payload,
    # while task metadata remains available on the normalized result.
    if isinstance(result_container, dict) and "data" in result_container:
        candidate = result_container.get("data")
        data_container = candidate if isinstance(candidate, dict) else None
        body = candidate
    elif isinstance(body, dict) and set(body).issubset(
        {"ok", "success", "status", "error", "message"}
    ):
        body = {}

    if isinstance(body, dict) and "output" in body:
        output = body.get("output")
        if output is not None:
            body = output

    # Biomni internal tools commonly wrap their value with execution and query
    # metadata. Unknown sibling fields make the object business data instead.
    while (
        isinstance(body, dict)
        and "result" in body
        and set(body).issubset(_TOOL_RESULT_ENVELOPE_KEYS)
    ):
        tool_result_containers.append(body)
        nested_result = body.get("result")
        empty_result = empty_result or _is_explicit_empty_result(nested_result)
        if body.get("success") is False or body.get("ok") is False:
            ok = False
        if not error:
            wrapper_error = body.get("error") or body.get("message")
            if wrapper_error:
                error = str(wrapper_error)
        if status == "success" and body.get("status"):
            status = _status(body.get("status")) or status
        body = body.get("result")

    if isinstance(body, dict):
        empty_result = empty_result or _is_explicit_empty_result(body)
        if body.get("success") is False or body.get("ok") is False:
            ok = False
        if not error:
            body_error = body.get("error") or body.get("message")
            if body_error:
                error = str(body_error)
        if status == "success" and body.get("status"):
            status = _status(body.get("status")) or status

    artifacts = envelope.get("artifacts")
    provenance = envelope.get("provenance")
    for container in (
        result_container,
        data_container,
        *tool_result_containers,
        body if isinstance(body, dict) else None,
    ):
        if not isinstance(container, dict):
            continue
        if artifacts is None:
            artifacts = container.get("artifacts")
        if provenance is None:
            provenance = container.get("provenance")
    if not isinstance(artifacts, (list, tuple)):
        artifacts = () if artifacts in (None, "") else (artifacts,)
    if not isinstance(provenance, (list, tuple)):
        provenance = () if provenance in (None, "") else (provenance,)
    if not ok and status == "success":
        status = "error"
    if status in {
        "failed",
        "cancelled",
        "timed_out",
        "manual_review",
        "dead_letter",
        "error",
    }:
        ok = False

    metadata = {
        key: envelope[key]
        for key in (
            "protocol",
            "task_id",
            "request_id",
            "task_type",
            "tool_name",
            "status",
            "worker_id",
            "attempt_count",
            "updated_at",
        )
        if key in envelope
    }
    for container in tool_result_containers:
        if "query_info" in container:
            metadata["query_info"] = container["query_info"]
    if empty_result:
        metadata["empty_result"] = True
    return NormalizedToolResult(
        body=body,
        ok=ok,
        status=status,
        error=error,
        artifacts=tuple(artifacts),
        provenance=tuple(provenance),
        metadata=metadata,
    )


class ResultAdapterRegistry:
    """Resolve catalog-declared result adapters with a generic fallback."""

    def __init__(self) -> None:
        self._adapters: dict[str, ResultAdapter] = {"generic": normalize_generic_result}

    def register(self, name: str, adapter: ResultAdapter) -> None:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("result adapter name must be non-empty")
        self._adapters[normalized] = adapter

    def adapt(self, payload: Any, name: str = "generic") -> NormalizedToolResult:
        adapter = self._adapters.get(str(name or "").strip(), self._adapters["generic"])
        return adapter(payload)
