from __future__ import annotations

from typing import Any


_TASK_FIELDS = (
    "action_id",
    "idempotency_key",
    "action_retryable",
    "blocked_existing_status",
    "task_id",
    "request_id",
    "status",
    "remote_status",
    "rpc_status",
    "gateway_tool",
    "task_type",
    "tool_name",
    "worker_id",
    "attempt_count",
    "execution_attempt_count",
    "resource_wait_count",
    "created_at",
    "updated_at",
    "next_poll_at",
    "deadline_at",
    "parent_task_id",
    "workflow_id",
    "workflow_parent_request_id",
    "error_code",
    "error_message",
    "error_retryable",
    "submission_unknown",
    "requires_reconciliation",
    "idempotency_conflict",
)
_ROUTE_FIELDS = (
    "backend",
    "reason_code",
    "request_id",
    "route_signature",
    "execution_signature",
    "admitted_capability",
    "selected_capability",
    "execution_invoked",
    "binding_id",
    "catalog_revision",
)
_STAGE_FIELDS = (
    "phase",
    "workflow_id",
    "parent_request_id",
    "request_id",
)
_EFFECT_FIELDS = (
    "passed",
    "missing_paths",
    "missing_artifacts",
    "missing_value_matches",
)


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "...[truncated]"
    if isinstance(value, dict):
        return {
            str(key): _bounded(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
            if str(key) != "workflow_resume"
        }
    if isinstance(value, list | tuple):
        return [_bounded(item, depth=depth + 1) for item in value[:24]]
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "...[truncated]"
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)[:2000]


def external_task_summary(metadata: Any) -> dict[str, Any]:
    """Project provider metadata into bounded durable control-plane state."""
    if not isinstance(metadata, dict):
        return {}
    summary = {
        key: _bounded(metadata[key])
        for key in _TASK_FIELDS
        if key in metadata
    }
    for source_key, fields in (
        ("omniagent_route", _ROUTE_FIELDS),
        ("domain_workflow_stage", _STAGE_FIELDS),
        ("effect_verification", _EFFECT_FIELDS),
    ):
        value = metadata.get(source_key)
        if isinstance(value, dict):
            projected = {
                key: _bounded(value[key]) for key in fields if key in value
            }
            if projected:
                summary[source_key] = projected
    for key in (
        "idempotency",
        "method_provenance",
        "artifact_manifest",
        "artifact_manifests",
        "artifacts",
    ):
        if key in metadata:
            summary[key] = _bounded(metadata[key])
    previous = metadata.get("previous_external_task")
    if isinstance(previous, dict):
        summary["previous_external_task"] = external_task_summary(previous)
    children = metadata.get("child_tasks")
    if isinstance(children, list):
        compact_children: list[dict[str, Any]] = []
        for child in children:
            compact_children = upsert_external_task_snapshot(compact_children, child)
        if compact_children:
            summary["child_tasks"] = compact_children
    return summary


def upsert_external_task_snapshot(
    children: list[dict[str, Any]], metadata: Any
) -> list[dict[str, Any]]:
    """Retain only the latest bounded snapshot for an external task identity."""
    snapshot = external_task_summary(metadata)
    if not snapshot:
        return [external_task_summary(item) for item in children if isinstance(item, dict)]
    identity = (
        str(snapshot.get("task_id") or ""),
        str(snapshot.get("request_id") or ""),
    )
    projected = [external_task_summary(item) for item in children if isinstance(item, dict)]
    if identity == ("", ""):
        if snapshot not in projected:
            projected.append(snapshot)
        return projected
    retained: list[dict[str, Any]] = []
    replacement_index: int | None = None
    for item in projected:
        item_identity = (
            str(item.get("task_id") or ""),
            str(item.get("request_id") or ""),
        )
        if item_identity == identity:
            if replacement_index is None:
                replacement_index = len(retained)
                retained.append(snapshot)
            continue
        retained.append(item)
    if replacement_index is None:
        retained.append(snapshot)
    return retained
