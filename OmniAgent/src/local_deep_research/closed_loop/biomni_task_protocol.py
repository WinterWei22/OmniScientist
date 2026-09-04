from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any


PayloadParser = Callable[[Any], Any]
StatusInvoker = Callable[[dict[str, Any]], Awaitable[Any]]

_TERMINAL_STATUSES = {
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
    "manual_review",
    "dead_letter",
}
_WAITING_STATUSES = {
    "submitted",
    "queued",
    "running",
    "retry_wait",
    "pending",
    "in_progress",
    "waiting",
}
_REVIEW_STATUSES = {"manual_review", "dead_letter"}
_STATUS_ALIASES = {
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
}
_TASK_METADATA_FIELDS = (
    "protocol",
    "task_id",
    "request_id",
    "task_type",
    "tool_name",
    "status",
    "task_dir",
    "log_path",
    "worker_id",
    "resources",
    "attempt_count",
    "execution_attempt_count",
    "resource_wait_count",
    "max_attempts",
    "next_attempt_at",
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
    "failure_kind",
    "last_heartbeat_at",
    "execution_deadline_at",
    "timeout_requested_at",
)


@dataclass(slots=True)
class BiomniTaskResolution:
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    pending: bool = False
    poll_error: str = ""


def parse_mcp_payload(response: Any) -> Any:
    """Extract one JSON-compatible payload from common MCP client responses."""
    structured = getattr(response, "structuredContent", None)
    if structured:
        return structured
    content = getattr(response, "content", None)
    if content and len(content) == 1 and hasattr(content[0], "text"):
        response = content[0].text
    elif isinstance(response, list) and len(response) == 1:
        item = response[0]
        if hasattr(item, "text"):
            response = item.text
        elif isinstance(item, dict) and item.get("type") == "text":
            response = item.get("text")
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return response
    return response


def normalize_task_status(value: Any) -> str:
    """Normalize lifecycle aliases without collapsing waiting states."""
    normalized = str(value or "").strip().lower()
    return _STATUS_ALIASES.get(normalized, normalized)


def normalize_biomni_task_payload(payload: Any) -> Any:
    """Adapt ``biomni.task.v1`` envelopes to the existing result consumers.

    The Harness verifies tool data rather than transport metadata.  For the v1
    envelope, expose ``result.data`` as ``result`` while preserving provenance
    and artifacts alongside it for result adapters and traces.
    """
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    if str(normalized.get("protocol") or "") != "biomni.task.v1":
        return normalized

    status = normalize_task_status(normalized.get("status"))
    if status:
        normalized["status"] = status
    if status == "succeeded":
        normalized.setdefault("ok", True)
    elif status in _TERMINAL_STATUSES - {"succeeded"}:
        normalized["ok"] = False
    elif status in _WAITING_STATUSES:
        normalized["ok"] = True

    error = normalized.get("error")
    if isinstance(error, dict):
        normalized["error_details"] = dict(error)
        normalized["error"] = str(
            error.get("message") or error.get("code") or "Biomni task failed"
        )

    result = normalized.get("result")
    if not isinstance(result, dict) or "data" not in result:
        return normalized

    data = result.get("data")
    provenance = result.get("provenance")
    artifacts = result.get("artifacts")
    if isinstance(data, dict):
        data = dict(data)
        if isinstance(provenance, list):
            data.setdefault("provenance", provenance)
        if isinstance(artifacts, list):
            data.setdefault("artifacts", artifacts)
    normalized["result"] = data
    if isinstance(provenance, list):
        normalized["provenance"] = provenance
    if isinstance(artifacts, list):
        normalized["artifacts"] = artifacts
    return normalized


def begin_biomni_submission(submission: Any) -> BiomniTaskResolution:
    """Normalize a submission without waiting for an external worker."""
    submission = normalize_biomni_task_payload(submission)
    if not isinstance(submission, dict) or not submission.get("task_id"):
        return BiomniTaskResolution(payload=submission)

    metadata = _task_metadata(submission)
    trace = [_task_event("biomni_task_submitted", metadata)]
    initial_status = normalize_task_status(metadata.get("status"))
    if initial_status in _TERMINAL_STATUSES:
        terminal_payload = dict(submission)
        if initial_status == "succeeded":
            terminal_payload["ok"] = True
        else:
            terminal_payload["ok"] = False
            terminal_payload.setdefault(
                "error",
                "Biomni worker reported task timeout"
                if initial_status == "timed_out"
                else "Biomni worker reported task cancellation"
                if initial_status == "cancelled"
                else "Biomni task requires manual review"
                if initial_status == "manual_review"
                else "Biomni task moved to dead letter"
                if initial_status == "dead_letter"
                else "Biomni worker reported task failure",
            )
            if initial_status in _REVIEW_STATUSES:
                metadata["requires_review"] = True
        trace.append(_task_event("biomni_task_completed", metadata))
        return BiomniTaskResolution(terminal_payload, metadata, trace)
    if initial_status and initial_status not in _WAITING_STATUSES:
        metadata["remote_status"] = initial_status
        metadata["status"] = "unknown"
        metadata["requires_reconciliation"] = True
        trace.append(_task_event("biomni_task_status_unknown", metadata))
        return BiomniTaskResolution(
            payload=dict(submission) | {"status": "unknown", "pending": True},
            metadata=metadata,
            trace=trace,
            pending=True,
        )
    trace.append(_task_event("biomni_task_waiting", metadata))
    return BiomniTaskResolution(
        payload=dict(submission) | {"pending": True},
        metadata=metadata,
        trace=trace,
        pending=True,
    )


async def poll_biomni_task(
    metadata: dict[str, Any],
    *,
    status_tool: Any,
    parse_payload: PayloadParser,
    invoke_status: StatusInvoker | None = None,
) -> BiomniTaskResolution:
    """Poll an already-submitted Biomni task exactly once."""
    task_id = str(metadata.get("task_id") or "").strip()
    trace: list[dict[str, Any]] = [_task_event("biomni_task_polling", metadata)]
    if not task_id:
        return BiomniTaskResolution(
            payload={"ok": False, "error": "missing Biomni task ID"},
            metadata=dict(metadata),
            trace=trace,
        )
    if status_tool is None:
        payload = _failure_payload(
            metadata,
            "Biomni returned an asynchronous task but get_biomni_task is unavailable",
        )
        trace.append(_task_event("biomni_task_poll_unavailable", metadata))
        return BiomniTaskResolution(payload, dict(metadata), trace)
    try:
        arguments = {"task_id": task_id}
        raw_status = await (
            invoke_status(arguments)
            if invoke_status is not None
            else status_tool.ainvoke(arguments)
        )
        status_payload = normalize_biomni_task_payload(parse_payload(raw_status))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        trace.append(_task_event("biomni_task_poll_error", metadata))
        return BiomniTaskResolution(
            payload=dict(metadata) | {"ok": True, "pending": True},
            metadata=dict(metadata),
            trace=trace,
            pending=True,
            poll_error=error,
        )
    if not isinstance(status_payload, dict):
        payload = _failure_payload(
            metadata,
            "get_biomni_task returned a non-object response",
        )
        trace.append(_task_event("biomni_task_poll_failed", metadata))
        return BiomniTaskResolution(payload, dict(metadata), trace)

    updated = dict(metadata)
    updated.update(_task_metadata(status_payload))
    status = normalize_task_status(status_payload.get("status"))
    if status:
        updated["status"] = status
        status_payload = dict(status_payload)
        status_payload["status"] = status
    # Lifecycle status is authoritative. Some Biomni envelopes set ok=false for
    # a queued/retry-wait task because the task has not produced result data yet.
    if status in _WAITING_STATUSES:
        trace.append(_task_event("biomni_task_waiting", updated))
        return BiomniTaskResolution(
            payload=dict(status_payload) | {"ok": True, "pending": True},
            metadata=updated,
            trace=trace,
            pending=True,
        )
    if status in _TERMINAL_STATUSES:
        trace.append(_task_event("biomni_task_completed", updated))
        status_payload = dict(status_payload)
        if status == "succeeded":
            status_payload["ok"] = True
        else:
            status_payload["ok"] = False
            status_payload.setdefault(
                "error",
                "Biomni worker reported task timeout"
                if status == "timed_out"
                else "Biomni worker reported task cancellation"
                if status == "cancelled"
                else "Biomni worker reported task failure",
            )
        return BiomniTaskResolution(status_payload, updated, trace)
    if status_payload.get("ok") is False:
        trace.append(_task_event("biomni_task_completed", updated))
        return BiomniTaskResolution(status_payload, updated, trace)
    if status not in _WAITING_STATUSES:
        if status in _REVIEW_STATUSES:
            status_payload = dict(status_payload)
            status_payload["ok"] = False
            status_payload.setdefault(
                "error",
                "Biomni task requires manual review"
                if status == "manual_review"
                else "Biomni task moved to dead letter",
            )
            updated["requires_review"] = True
            trace.append(_task_event("biomni_task_requires_review", updated))
            return BiomniTaskResolution(status_payload, updated, trace)
        # Preserve the task identity for reconciliation. An unknown remote
        # state is not evidence that the external task failed.
        updated["remote_status"] = status
        updated["status"] = "unknown"
        updated["requires_reconciliation"] = True
        trace.append(_task_event("biomni_task_status_unknown", updated))
        return BiomniTaskResolution(
            payload=dict(status_payload)
            | {"status": "unknown", "ok": True, "pending": True},
            metadata=updated,
            trace=trace,
            pending=True,
        )
    payload = _failure_payload(
        status_payload,
        f"get_biomni_task returned unsupported status: {status or '<empty>'}",
    )
    trace.append(_task_event("biomni_task_poll_failed", updated))
    return BiomniTaskResolution(payload, updated, trace)


async def resolve_biomni_submission(
    submission: Any,
    *,
    status_tool: Any,
    parse_payload: PayloadParser,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 900.0,
    max_consecutive_poll_errors: int = 3,
) -> BiomniTaskResolution:
    """Legacy blocking compatibility wrapper around one-poll task primitives."""
    resolution = begin_biomni_submission(submission)
    if not resolution.pending:
        return resolution

    deadline = monotonic() + max(0.0, float(timeout_seconds))
    interval = max(0.0, float(poll_interval_seconds))
    consecutive_errors = 0
    while True:
        if monotonic() >= deadline:
            # This is only the caller's wait budget. The remote task may still
            # be queued/running, so retain its identity for resume/reconcile.
            pending_metadata = dict(resolution.metadata)
            pending_metadata["status"] = "unknown"
            pending_metadata["timeout_kind"] = "client_wait_deadline"
            pending_metadata["requires_reconciliation"] = True
            payload = dict(pending_metadata) | {
                "ok": True,
                "status": "unknown",
                "pending": True,
                "error": (
                    "Biomni task did not reach a terminal state within the local "
                    f"wait budget of {timeout_seconds:g} seconds"
                ),
            }
            resolution.trace.append(
                _task_event("biomni_task_wait_deadline_exceeded", pending_metadata)
            )
            return BiomniTaskResolution(
                payload, pending_metadata, resolution.trace, pending=True
            )

        if interval:
            await asyncio.sleep(interval)
        polled = await poll_biomni_task(
            resolution.metadata,
            status_tool=status_tool,
            parse_payload=parse_payload,
        )
        resolution.trace.extend(polled.trace)
        resolution.metadata = polled.metadata
        if polled.poll_error:
            consecutive_errors += 1
            if consecutive_errors < max(1, max_consecutive_poll_errors):
                continue
            payload = _failure_payload(
                resolution.metadata,
                "Biomni task status polling failed after "
                f"{consecutive_errors} attempts: {polled.poll_error}",
            )
            unknown_metadata = dict(resolution.metadata)
            unknown_metadata["status"] = "unknown"
            unknown_metadata["requires_reconciliation"] = True
            resolution.trace.append(_task_event("biomni_task_poll_failed", resolution.metadata))
            return BiomniTaskResolution(
                dict(payload)
                | {"status": "unknown", "ok": True, "pending": True},
                unknown_metadata,
                resolution.trace,
                pending=True,
                poll_error=polled.poll_error,
            )
        consecutive_errors = 0
        if not polled.pending:
            return BiomniTaskResolution(polled.payload, polled.metadata, resolution.trace)


def _task_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        field: payload[field]
        for field in _TASK_METADATA_FIELDS
        if field in payload and payload[field] is not None
    }
    error = payload.get("error_details")
    if isinstance(error, dict):
        for field in ("code", "message", "retryable"):
            if field in error:
                metadata[f"error_{field}"] = error[field]
    idempotency = payload.get("idempotency")
    if isinstance(idempotency, dict):
        metadata["idempotency"] = dict(idempotency)
        if idempotency.get("replayed") is True:
            metadata["idempotency_replayed"] = True
        if idempotency.get("conflict") is True:
            metadata["idempotency_conflict"] = True
    return metadata


def _task_event(event: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        **{
            field: metadata[field]
            for field in (
            "task_id",
            "request_id",
            "task_type",
                "tool_name",
                "status",
                "worker_id",
                "attempt_count",
            )
            if field in metadata
        },
    }


def _failure_payload(payload: dict[str, Any], error: str) -> dict[str, Any]:
    return dict(payload) | {"ok": False, "error": error}
