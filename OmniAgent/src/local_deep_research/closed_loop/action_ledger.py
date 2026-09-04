from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any


class ActionStatus(str, Enum):
    PLANNED = "planned"
    ADMITTED = "admitted"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"
    MANUAL_REVIEW = "manual_review"
    DEAD_LETTER = "dead_letter"


ACTIVE_ACTION_STATUSES = frozenset(
    {
        ActionStatus.ADMITTED.value,
        ActionStatus.SUBMITTED.value,
        ActionStatus.QUEUED.value,
        ActionStatus.RUNNING.value,
        ActionStatus.RETRY_WAIT.value,
        ActionStatus.UNKNOWN.value,
    }
)


def canonical_action_key(payload: Any) -> str:
    """Return a stable identity for one logical external action."""
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def new_request_id(
    *, run_id: str, backend: str, step_id: str, attempt: int = 1
) -> str:
    """Create a protocol-safe ID for one logical Biomni submission."""
    value = (
        f"omniagent:{str(run_id).strip()}:{str(backend).strip()}"
        f":{str(step_id).strip()}:attempt-{max(1, int(attempt))}"
    )
    return value[:256]


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    idempotency_key: str
    iteration: int
    step_id: str
    backend: str
    capability_id: str = ""
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    route_signature: str = ""
    # Stable identity of the logical action. Child stage identities are separate.
    request_id: str = ""
    external_request_id: str = ""
    external_task_id: str = ""
    status: str = ActionStatus.PLANNED.value
    attempt: int = 0
    result_key: str = ""
    result_status: str = ""
    last_error: str = ""
    retryable: bool = True
    task_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_ACTION_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ActionStatus.SUCCEEDED.value,
            ActionStatus.FAILED.value,
            ActionStatus.TIMED_OUT.value,
            ActionStatus.CANCELLED.value,
        }

    def storage_key(self) -> str:
        return self.result_key or f"{self.action_id}:attempt:{self.attempt}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionLedger:
    """Persistent logical-action registry used before external side effects."""

    schema_version = "omniagent.action_ledger.v1"

    def __init__(self, records: dict[str, ActionRecord] | None = None) -> None:
        self.records: dict[str, ActionRecord] = dict(records or {})

    def find_by_idempotency_key(self, key: str) -> ActionRecord | None:
        key = str(key or "").strip()
        if not key:
            return None
        matches = [
            item for item in self.records.values() if item.idempotency_key == key
        ]
        return max(matches, key=lambda item: (item.updated_at, item.attempt), default=None)

    def admit(
        self,
        *,
        run_id: str = "",
        idempotency_key: str,
        iteration: int,
        step_id: str,
        backend: str,
        capability_id: str = "",
        normalized_arguments: dict[str, Any] | None = None,
        route_signature: str = "",
        request_id: str = "",
        max_attempts: int = 2,
    ) -> tuple[ActionRecord, str]:
        """Admit an action and return ``(record, disposition)``.

        Dispositions are ``new``, ``retry``, ``reuse``, ``attach``, ``replay``,
        ``conflict`` and ``blocked``.  UNKNOWN actions are deliberately never
        admitted again unless their submission response was explicitly unknown.
        """
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("an idempotency key is required to admit an action")
        existing = self.find_by_idempotency_key(key)
        if existing is not None:
            if (
                existing.backend
                and str(backend)
                and existing.backend != str(backend)
            ) or (
                existing.normalized_arguments
                and normalized_arguments is not None
                and canonical_action_key(existing.normalized_arguments)
                != canonical_action_key(normalized_arguments)
            ):
                return existing, "conflict"
            if existing.status == ActionStatus.SUCCEEDED.value:
                return existing, "reuse"
            if existing.status in {
                ActionStatus.ADMITTED.value,
                ActionStatus.SUBMITTED.value,
            }:
                return existing, "replay" if not existing.external_task_id else "attach"
            if existing.status in {
                ActionStatus.QUEUED.value,
                ActionStatus.RUNNING.value,
                ActionStatus.RETRY_WAIT.value,
            }:
                return existing, "attach"
            if existing.status == ActionStatus.UNKNOWN.value:
                if not existing.external_task_id and (
                    existing.task_metadata.get("submission_unknown")
                    or "submission" in existing.last_error.lower()
                    or "transport" in existing.last_error.lower()
                ):
                    return existing, "replay"
                return existing, "blocked"
            if existing.status == ActionStatus.FAILED.value:
                if (
                    not existing.retryable
                    or "idempotency_conflict" in existing.last_error.lower()
                    or existing.task_metadata.get("idempotency_conflict")
                ):
                    return existing, "blocked"
                if existing.attempt >= max(1, int(max_attempts)):
                    return existing, "blocked"
                existing.attempt += 1
                existing.status = ActionStatus.ADMITTED.value
                existing.external_request_id = ""
                existing.external_task_id = ""
                existing.result_key = ""
                existing.result_status = ""
                existing.last_error = ""
                existing.task_metadata = {}
                existing.request_id = new_request_id(
                    run_id=str(run_id).strip() or "unknown",
                    backend=existing.backend,
                    step_id=existing.step_id,
                    attempt=existing.attempt,
                )
                existing.updated_at = time()
                return existing, "retry"
            if existing.status in {
                ActionStatus.TIMED_OUT.value,
                ActionStatus.CANCELLED.value,
                ActionStatus.MANUAL_REVIEW.value,
                ActionStatus.DEAD_LETTER.value,
            }:
                return existing, "blocked"
            if existing.status == ActionStatus.PLANNED.value:
                existing.status = ActionStatus.ADMITTED.value
                existing.updated_at = time()
                return existing, "new"
            if existing.status == ActionStatus.ADMITTED.value:
                return existing, "attach"

        action_id = f"action-{key[:20]}"
        normalized_request_id = str(request_id).strip()
        if normalized_request_id and len(normalized_request_id) > 256:
            raise ValueError("request_id must be at most 256 characters")
        # A collision is extraordinarily unlikely, but retaining the key in the
        # ID makes a conflicting record visible instead of silently overwriting it.
        if action_id in self.records and self.records[action_id].idempotency_key != key:
            action_id = f"{action_id}-{canonical_action_key(key)[:8]}"
        record = ActionRecord(
            action_id=action_id,
            idempotency_key=key,
            iteration=int(iteration),
            step_id=str(step_id),
            backend=str(backend),
            capability_id=str(capability_id),
            normalized_arguments=dict(normalized_arguments or {}),
            route_signature=str(route_signature),
            request_id=(
                normalized_request_id
                or new_request_id(
                    run_id=str(run_id).strip() or "unknown",
                    backend=str(backend),
                    step_id=str(step_id),
                    attempt=1,
                )
            ),
            status=ActionStatus.ADMITTED.value,
            attempt=1,
        )
        self.records[record.action_id] = record
        return record, "new"

    def update(
        self,
        action_id: str,
        status: str | ActionStatus,
        *,
        external_task_id: str | None = None,
        request_id: str | None = None,
        external_request_id: str | None = None,
        result_key: str | None = None,
        result_status: str | None = None,
        error: str | None = None,
        retryable: bool | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> ActionRecord:
        record = self.records.get(str(action_id))
        if record is None:
            raise KeyError(f"unknown action ID: {action_id}")
        normalized = status.value if isinstance(status, ActionStatus) else str(status)
        if normalized not in {item.value for item in ActionStatus}:
            raise ValueError(f"unsupported action status: {normalized}")
        record.status = normalized
        if external_task_id is not None:
            record.external_task_id = str(external_task_id)
        if request_id is not None:
            record.request_id = str(request_id)
        if external_request_id is not None:
            record.external_request_id = str(external_request_id)
        if result_key is not None:
            record.result_key = str(result_key)
        if result_status is not None:
            record.result_status = str(result_status)
        if error is not None:
            record.last_error = str(error)
        if retryable is not None:
            record.retryable = bool(retryable)
        if task_metadata is not None:
            record.task_metadata = dict(task_metadata)
        record.updated_at = time()
        return record

    def unresolved(self) -> list[ActionRecord]:
        return sorted(
            (item for item in self.records.values() if item.is_active),
            key=lambda item: (item.updated_at, item.action_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [item.to_dict() for item in self.records.values()],
        }

    @classmethod
    def from_dict(cls, value: Any) -> ActionLedger:
        if not isinstance(value, dict):
            return cls()
        records: dict[str, ActionRecord] = {}
        raw_records = value.get("records", [])
        if isinstance(raw_records, dict):
            raw_records = list(raw_records.values())
        if not isinstance(raw_records, list):
            return cls()
        valid_statuses = {item.value for item in ActionStatus}
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            action_id = str(raw.get("action_id", "")).strip()
            key = str(raw.get("idempotency_key", "")).strip()
            if not action_id or not key:
                continue
            status = str(raw.get("status", ActionStatus.PLANNED.value))
            if status not in valid_statuses:
                status = ActionStatus.UNKNOWN.value
            try:
                record = ActionRecord(
                    action_id=action_id,
                    idempotency_key=key,
                    iteration=int(raw.get("iteration", 0)),
                    step_id=str(raw.get("step_id", "")),
                    backend=str(raw.get("backend", "unknown")),
                    capability_id=str(raw.get("capability_id", "")),
                    normalized_arguments=(
                        dict(raw.get("normalized_arguments", {}))
                        if isinstance(raw.get("normalized_arguments", {}), dict)
                        else {}
                    ),
                    route_signature=str(raw.get("route_signature", "")),
                    request_id=str(raw.get("request_id", "")),
                    external_request_id=str(raw.get("external_request_id", "")),
                    external_task_id=str(raw.get("external_task_id", "")),
                    status=status,
                    attempt=max(0, int(raw.get("attempt", 0))),
                    result_key=str(raw.get("result_key", "")),
                    result_status=str(raw.get("result_status", "")),
                    last_error=str(raw.get("last_error", "")),
                    retryable=bool(raw.get("retryable", True)),
                    task_metadata=(
                        dict(raw.get("task_metadata", {}))
                        if isinstance(raw.get("task_metadata", {}), dict)
                        else {}
                    ),
                    created_at=float(raw.get("created_at", 0.0) or 0.0),
                    updated_at=float(raw.get("updated_at", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                continue
            records[action_id] = record
        return cls(records)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
