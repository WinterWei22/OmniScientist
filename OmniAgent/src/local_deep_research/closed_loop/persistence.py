from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any

from .action_ledger import ActionLedger
from .contracts import (
    A1TaskResult,
    AnalysisResult,
    Critique,
    Decision,
    ExperimentPlan,
    ExperimentStep,
    IterationRecord,
    LoopPhase,
    PendingExecution,
    ResearchState,
    RunStatus,
    WorkflowEvaluation,
)
from .scientific_state import (
    AttemptRecord,
    AttemptStatus,
    CanonicalEntityRecord,
    ClaimRecord,
    ClaimStatus,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    ScientificState,
)


EVENT_SCHEMA_VERSION = "omniagent.events.v1"
CHECKPOINT_SCHEMA_VERSION = "omniagent.checkpoint.v1"
PENDING_SNAPSHOT_SCHEMA_VERSION = "omniagent.pending.v1"
ACTION_LEDGER_SCHEMA_VERSION = "omniagent.action_ledger.v1"
STATE_SCHEMA_VERSION = "omniagent.research_state.v1"


class CheckpointValidationError(ValueError):
    pass


class RunPersistence:
    """Append-only run artifacts plus validated JSON checkpoint/resume."""

    def __init__(self, root: str | Path, *, run_id: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = self._safe_component(run_id)
        self.events_path = self.root / "events.jsonl"
        self.executions_dir = self.root / "executions"
        self.artifacts_dir = self.root / "artifacts"
        self.checkpoints_dir = self.root / "checkpoints"
        self.pending_snapshot_path = self.root / "pending.json"
        self.action_ledger_path = self.root / "actions.json"
        for directory in (
            self.root,
            self.executions_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._sequence = self._read_last_sequence()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> RunPersistence:
        path = Path(checkpoint).expanduser().resolve()
        if path.is_dir():
            root = path
        elif path.parent.name == "checkpoints":
            root = path.parent.parent
        else:
            root = path.parent
        envelope = cls._read_checkpoint_envelope(path if path.is_file() else root)
        run_id = str(envelope.get("run_id", "")).strip()
        if not run_id:
            raise CheckpointValidationError("checkpoint does not contain a run ID")
        return cls(root, run_id=run_id)

    def append_event(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        record = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": str(event),
            "payload": _jsonable(payload),
        }
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def write_artifact(self, kind: str, payload: Any) -> str:
        body = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": str(kind),
            "payload": _jsonable(payload),
        }
        encoded = _canonical_bytes(body)
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.artifacts_dir / f"{self._safe_component(kind)}-{digest}.json"
        self._write_once(path, encoded + b"\n")
        self.append_event(
            "artifact_recorded",
            {"kind": kind, "path": str(path), "sha256": digest},
        )
        return str(path)

    def save_execution_result(self, attempt_key: str, result: A1TaskResult) -> str:
        key_digest = hashlib.sha256(attempt_key.encode("utf-8")).hexdigest()
        path = self.executions_dir / f"execution-{key_digest}.json"
        body = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_key": attempt_key,
            "result": _encode_result(result),
        }
        encoded = _canonical_bytes(body) + b"\n"
        if path.exists():
            existing = path.read_bytes()
            if existing != encoded:
                raise CheckpointValidationError(
                    "an execution result already exists for this attempt key"
                )
            return str(path)
        self._write_once(path, encoded)
        self.append_event(
            "execution_result_recorded",
            {
                "attempt_key": attempt_key,
                "path": str(path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
        )
        return str(path)

    def load_execution_result(self, attempt_key: str) -> tuple[A1TaskResult, str] | None:
        key_digest = hashlib.sha256(attempt_key.encode("utf-8")).hexdigest()
        path = self.executions_dir / f"execution-{key_digest}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported execution artifact schema")
        if value.get("run_id") != self.run_id or value.get("attempt_key") != attempt_key:
            raise CheckpointValidationError("execution artifact identity mismatch")
        result = _decode_result(_mapping(value.get("result")))
        if str(path) not in result.artifacts:
            result.artifacts.append(str(path))
        return result, str(path)

    def load_reusable_mcp_result(
        self,
        execution_signature: str,
        catalog_revision: str,
    ) -> tuple[A1TaskResult, str, str] | None:
        """Reuse only verified successful MCP work from the same immutable catalog."""
        if not execution_signature or not catalog_revision:
            return None
        paths = sorted(
            self.executions_dir.glob("execution-*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise CheckpointValidationError("unsupported execution artifact schema")
            if value.get("run_id") != self.run_id:
                raise CheckpointValidationError("execution artifact identity mismatch")
            result = _decode_result(_mapping(value.get("result")))
            metadata = result.task_metadata
            route = metadata.get("omniagent_route", {}) if isinstance(metadata, dict) else {}
            verification = (
                metadata.get("effect_verification", {})
                if isinstance(metadata, dict)
                else {}
            )
            if not (
                result.success
                and isinstance(route, dict)
                and route.get("backend") == "mcp"
                and route.get("execution_signature") == execution_signature
                and route.get("catalog_revision") == catalog_revision
                and isinstance(verification, dict)
                and verification.get("passed") is True
            ):
                continue
            if str(path) not in result.artifacts:
                result.artifacts.append(str(path))
            return result, str(path), str(value.get("attempt_key", ""))
        return None

    def save_checkpoint(self, state: ResearchState, *, safe_point: str) -> str:
        # Keep the small action index durable even when the process stops between
        # two full scientific-state checkpoints.
        self.save_action_ledger(state.action_ledger)
        state_payload = _encode_state(state)
        body = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "safe_point": str(safe_point),
            "last_event_sequence": self._sequence,
            "state": state_payload,
        }
        digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        envelope = dict(body)
        envelope["sha256"] = digest
        safe_point_name = self._safe_component(safe_point)
        path = self.checkpoints_dir / (
            f"checkpoint-{len(state.iterations):04d}-{state.scientific_state.state_version:06d}-"
            f"{safe_point_name}-{digest[:12]}.json"
        )
        self._write_once(path, _canonical_bytes(envelope) + b"\n")
        pointer = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "checkpoint": path.name,
            "sha256": digest,
        }
        self._atomic_write_json(self.root / "latest.json", pointer)
        return str(path)

    def save_pending_snapshot(self, state: ResearchState) -> str:
        """Persist only mutable external-task poll state between full checkpoints."""
        pending = state.pending_execution
        body = {
            "schema_version": PENDING_SNAPSHOT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "active_plan_id": state.active_plan.plan_id if state.active_plan else "",
            "pending_execution": _jsonable(pending) if pending is not None else None,
            "last_event_sequence": self._sequence,
        }
        envelope = dict(body)
        envelope["sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        self._atomic_write_json(self.pending_snapshot_path, envelope)
        return str(self.pending_snapshot_path)

    def save_action_ledger(self, ledger: ActionLedger) -> str:
        """Persist the logical-action index independently of full checkpoints."""
        body = ledger.to_dict()
        body["schema_version"] = ACTION_LEDGER_SCHEMA_VERSION
        envelope = dict(body)
        envelope["sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        self._atomic_write_json(self.action_ledger_path, envelope)
        return str(self.action_ledger_path)

    def load_action_ledger(self) -> ActionLedger:
        if not self.action_ledger_path.is_file():
            return ActionLedger()
        try:
            envelope = json.loads(self.action_ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError("action ledger is unreadable") from exc
        if not isinstance(envelope, dict):
            raise CheckpointValidationError("action ledger must be a JSON object")
        if envelope.get("schema_version") != ACTION_LEDGER_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported action ledger schema")
        expected = str(envelope.get("sha256", ""))
        body = dict(envelope)
        body.pop("sha256", None)
        if not expected or expected != hashlib.sha256(_canonical_bytes(body)).hexdigest():
            raise CheckpointValidationError("action ledger digest validation failed")
        return ActionLedger.from_dict(body)

    def load_checkpoint(self, checkpoint: str | Path | None = None) -> ResearchState:
        envelope = self._read_checkpoint_envelope(checkpoint or self.root)
        if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported checkpoint schema")
        if envelope.get("state_schema_version") != STATE_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported research-state schema")
        if envelope.get("run_id") != self.run_id:
            raise CheckpointValidationError("checkpoint run ID does not match store")
        expected = str(envelope.get("sha256", ""))
        body = dict(envelope)
        body.pop("sha256", None)
        actual = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if not expected or expected != actual:
            raise CheckpointValidationError("checkpoint digest validation failed")
        state = _decode_state(_mapping(envelope.get("state")))
        # Older checkpoints did not embed the ledger.  A separately persisted
        # ledger may also contain a newer dispatch transition, so merge it by
        # record timestamp rather than silently losing the external-task identity.
        if self.action_ledger_path.is_file():
            separate = self.load_action_ledger()
            for action_id, record in separate.records.items():
                current = state.action_ledger.records.get(action_id)
                if current is None or record.updated_at >= current.updated_at:
                    state.action_ledger.records[action_id] = record
        self._apply_pending_snapshot(state)
        self._validate_state(state)
        return state

    def _apply_pending_snapshot(self, state: ResearchState) -> None:
        """Merge a newer compact poll snapshot into a pending full checkpoint."""
        if state.pending_execution is None or not self.pending_snapshot_path.is_file():
            return
        try:
            envelope = json.loads(self.pending_snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError("pending snapshot is unreadable") from exc
        if not isinstance(envelope, dict):
            raise CheckpointValidationError("pending snapshot must be a JSON object")
        if envelope.get("schema_version") != PENDING_SNAPSHOT_SCHEMA_VERSION:
            raise CheckpointValidationError("unsupported pending snapshot schema")
        if envelope.get("run_id") != self.run_id:
            raise CheckpointValidationError("pending snapshot run ID does not match store")
        expected = str(envelope.get("sha256", ""))
        body = dict(envelope)
        body.pop("sha256", None)
        if not expected or expected != hashlib.sha256(_canonical_bytes(body)).hexdigest():
            raise CheckpointValidationError("pending snapshot digest validation failed")
        pending_raw = envelope.get("pending_execution")
        if pending_raw is None:
            return
        snapshot = _decode_pending_execution(_mapping(pending_raw))
        checkpoint_pending = state.pending_execution
        if (
            snapshot.task_id != checkpoint_pending.task_id
            or snapshot.request_id != checkpoint_pending.request_id
            or snapshot.step_id != checkpoint_pending.step_id
            or snapshot.iteration != checkpoint_pending.iteration
        ):
            raise CheckpointValidationError(
                "pending snapshot does not match the checkpoint task identity"
            )
        active_plan_id = str(envelope.get("active_plan_id", ""))
        if active_plan_id and (
            state.active_plan is None or state.active_plan.plan_id != active_plan_id
        ):
            raise CheckpointValidationError(
                "pending snapshot does not match the checkpoint active plan"
            )
        state.pending_execution = snapshot

    @staticmethod
    def validate_resume_request(
        state: ResearchState,
        *,
        goal: str,
        workspace: str,
        task_manifest: dict[str, Any],
    ) -> None:
        replayable_submission = any(
            action.status in {"admitted", "submitted"}
            or action.status == "unknown"
            and bool(action.task_metadata.get("submission_unknown"))
            for action in state.action_ledger.records.values()
        )
        if state.status is not RunStatus.RUNNING and not (
            state.status is RunStatus.NEEDS_REVIEW
            and (state.pending_execution is not None or replayable_submission)
        ):
            raise CheckpointValidationError(
                "cannot resume a terminal run without a pending external task"
            )
        if state.goal != goal:
            raise CheckpointValidationError("resume goal does not match checkpoint")
        if Path(state.workspace).resolve() != Path(workspace).resolve():
            raise CheckpointValidationError("resume workspace does not match checkpoint")
        if _canonical_bytes(state.task_manifest) != _canonical_bytes(task_manifest):
            raise CheckpointValidationError("resume task manifest does not match checkpoint")

    @staticmethod
    def _validate_state(state: ResearchState) -> None:
        scientific = state.scientific_state
        if scientific.state_version != len(scientific.transition_ids):
            raise CheckpointValidationError(
                "scientific state version does not match transition history"
            )
        if len(scientific.transition_ids) != len(set(scientific.transition_ids)):
            raise CheckpointValidationError("scientific transition IDs are not unique")
        if state.active_plan is None and state.active_executions:
            raise CheckpointValidationError(
                "checkpoint contains active executions without an active plan"
            )
        if state.active_plan is not None and len(state.active_executions) > len(
            state.active_plan.steps
        ):
            raise CheckpointValidationError(
                "checkpoint has more active executions than planned steps"
            )
        if state.pending_execution is not None:
            pending = state.pending_execution
            if state.active_plan is None:
                raise CheckpointValidationError(
                    "checkpoint contains a pending task without an active plan"
                )
            if not pending.task_id or not pending.request_id or not pending.step_id:
                raise CheckpointValidationError(
                    "checkpoint pending task identity is incomplete"
                )
            if pending.step_id not in {
                step.step_id for step in state.active_plan.steps
            }:
                raise CheckpointValidationError(
                    "checkpoint pending task does not belong to the active plan"
                )
        for action in state.action_ledger.records.values():
            if not action.idempotency_key or not action.action_id:
                raise CheckpointValidationError("checkpoint contains an invalid action record")
            if action.status in {"submitted", "queued", "running", "unknown"}:
                if not action.request_id and not action.external_task_id:
                    raise CheckpointValidationError(
                        "active action is missing request or external task identity"
                    )
        completed_calls = sum(len(item.executions) for item in state.iterations) + len(
            state.active_executions
        )
        if state.a1_call_count < completed_calls:
            raise CheckpointValidationError("checkpoint A1 call count is inconsistent")

    @classmethod
    def _read_checkpoint_envelope(cls, checkpoint: str | Path) -> dict[str, Any]:
        path = Path(checkpoint).expanduser().resolve()
        if path.is_dir():
            pointer_path = path / "latest.json"
            if not pointer_path.is_file():
                raise CheckpointValidationError("latest checkpoint pointer is missing")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            name = str(pointer.get("checkpoint", ""))
            if not name or Path(name).name != name:
                raise CheckpointValidationError("invalid latest checkpoint pointer")
            path = path / "checkpoints" / name
        if not path.is_file():
            raise CheckpointValidationError(f"checkpoint file is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CheckpointValidationError("checkpoint must be a JSON object")
        return value

    def _read_last_sequence(self) -> int:
        if not self.events_path.is_file():
            return 0
        last = 0
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CheckpointValidationError(
                        f"invalid event log record at line {line_number}"
                    ) from exc
                sequence = int(value.get("sequence", 0))
                if sequence != last + 1:
                    raise CheckpointValidationError("event log sequence is not contiguous")
                if value.get("run_id") != self.run_id:
                    raise CheckpointValidationError(
                        "event log belongs to a different run ID"
                    )
                last = sequence
        return last

    @staticmethod
    def _write_once(path: Path, data: bytes) -> None:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != data:
                raise CheckpointValidationError(
                    f"append-only artifact already exists with different content: {path}"
                )
            return
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _safe_component(value: str) -> str:
        result = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value)).strip(".-")
        if not result:
            raise ValueError("persistence identifier has no safe filename characters")
        return result


def _encode_state(state: ResearchState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "constraints": state.constraints,
        "workspace": state.workspace,
        "task_manifest": state.task_manifest,
        "run_id": state.run_id,
        "phase": state.phase.value,
        "status": state.status.value,
        "iterations": [_encode_iteration(item) for item in state.iterations],
        "best_score": state.best_score,
        "stalled_iterations": state.stalled_iterations,
        "a1_call_count": state.a1_call_count,
        "finish_reason": state.finish_reason,
        "working_memory_token_budget": state.working_memory_token_budget,
        "active_plan": _encode_plan(state.active_plan) if state.active_plan else None,
        "active_executions": [_encode_result(item) for item in state.active_executions],
        "pending_execution": _jsonable(state.pending_execution),
        "route_retry_state": _jsonable(state.route_retry_state),
        "action_ledger": state.action_ledger.to_dict(),
        "final_output_materialized": state.final_output_materialized,
        "scientific_state": _jsonable(state.scientific_state),
    }


def _decode_state(value: dict[str, Any]) -> ResearchState:
    state = ResearchState(
        goal=str(value["goal"]),
        constraints=_strings(value.get("constraints")),
        workspace=str(value["workspace"]),
        task_manifest=_mapping(value.get("task_manifest")),
        run_id=str(value["run_id"]),
        phase=LoopPhase(str(value.get("phase", LoopPhase.INITIALIZE.value))),
        status=RunStatus(str(value.get("status", RunStatus.RUNNING.value))),
        iterations=[_decode_iteration(_mapping(item)) for item in value.get("iterations", [])],
        best_score=float(value.get("best_score", 0.0)),
        stalled_iterations=int(value.get("stalled_iterations", 0)),
        a1_call_count=int(value.get("a1_call_count", 0)),
        finish_reason=str(value.get("finish_reason", "")),
        working_memory_token_budget=int(value.get("working_memory_token_budget", 2400)),
        active_plan=(
            _decode_plan(_mapping(value["active_plan"]))
            if value.get("active_plan") is not None
            else None
        ),
        active_executions=[
            _decode_result(_mapping(item)) for item in value.get("active_executions", [])
        ],
        pending_execution=(
            _decode_pending_execution(_mapping(value.get("pending_execution")))
            if value.get("pending_execution") is not None
            else None
        ),
        route_retry_state={
            str(key): _mapping(item)
            for key, item in _mapping(value.get("route_retry_state")).items()
            if isinstance(item, dict)
        },
        action_ledger=ActionLedger.from_dict(value.get("action_ledger")),
        final_output_materialized=bool(value.get("final_output_materialized", False)),
    )
    state.scientific_state = _decode_scientific_state(
        _mapping(value.get("scientific_state"))
    )
    return state


def _encode_iteration(value: IterationRecord) -> dict[str, Any]:
    return {
        "iteration": value.iteration,
        "plan": _encode_plan(value.plan),
        "executions": [
            _encode_result(item, historical=True) for item in value.executions
        ],
        "analysis": _jsonable(value.analysis),
        "evaluation": _jsonable(value.evaluation),
        "critique": _jsonable(value.critique),
        "score_improvement": value.score_improvement,
    }


def _decode_iteration(value: dict[str, Any]) -> IterationRecord:
    analysis = _mapping(value.get("analysis"))
    evaluation = _mapping(value.get("evaluation"))
    critique = _mapping(value.get("critique"))
    return IterationRecord(
        iteration=int(value["iteration"]),
        plan=_decode_plan(_mapping(value["plan"])),
        executions=[_decode_result(_mapping(item)) for item in value.get("executions", [])],
        analysis=AnalysisResult(
            summary=str(analysis.get("summary", "")),
            observations=_records(analysis.get("observations")),
            metrics=_float_mapping(analysis.get("metrics")),
            anomalies=_strings(analysis.get("anomalies")),
            evidence_gaps=_strings(analysis.get("evidence_gaps")),
            final_output=(
                _mapping(analysis.get("final_output"))
                if isinstance(analysis.get("final_output"), dict)
                else None
            ),
            supporting_evidence_ids=_strings(analysis.get("supporting_evidence_ids")),
        ),
        evaluation=WorkflowEvaluation(
            evaluator_id=str(evaluation["evaluator_id"]),
            score=float(evaluation["score"]),
            metrics=_float_mapping(evaluation.get("metrics")),
            satisfied_criteria=_strings(evaluation.get("satisfied_criteria")),
            failed_criteria=_strings(evaluation.get("failed_criteria")),
            evidence=_records(evaluation.get("evidence")),
            errors=_strings(evaluation.get("errors")),
            retryable=bool(evaluation.get("retryable", True)),
            evaluation_id=str(evaluation["evaluation_id"]),
        ),
        critique=Critique(
            decision=Decision(str(critique["decision"])),
            score=float(critique["score"]),
            summary=str(critique.get("summary", "")),
            satisfied_criteria=_strings(critique.get("satisfied_criteria")),
            failed_criteria=_strings(critique.get("failed_criteria")),
            evidence_gaps=_strings(critique.get("evidence_gaps")),
            required_changes=_strings(critique.get("required_changes")),
            next_experiment=str(critique.get("next_experiment", "")),
            retryable=bool(critique.get("retryable", True)),
            source_evaluation_id=str(critique.get("source_evaluation_id", "")),
            feedback_id=str(critique["feedback_id"]),
        ),
        score_improvement=float(value.get("score_improvement", 0.0)),
    )


def _encode_plan(value: ExperimentPlan) -> dict[str, Any]:
    return _jsonable(value)


def _decode_plan(value: dict[str, Any]) -> ExperimentPlan:
    return ExperimentPlan(
        hypothesis=str(value.get("hypothesis", "")),
        rationale=str(value.get("rationale", "")),
        steps=[
            ExperimentStep(
                step_id=str(item["step_id"]),
                objective=str(item["objective"]),
                inputs=_mapping(item.get("inputs")),
                constraints=_strings(item.get("constraints")),
                expected_outputs=_strings(item.get("expected_outputs")),
                success_criteria=_strings(item.get("success_criteria")),
            )
            for item in (_mapping(raw) for raw in value.get("steps", []))
        ],
        plan_id=str(value["plan_id"]),
        feedback_ids_consumed=_strings(value.get("feedback_ids_consumed")),
        evidence_refs_consumed=_strings(value.get("evidence_refs_consumed")),
        adaptation_summary=str(value.get("adaptation_summary", "")),
        planner_contract_version=str(value.get("planner_contract_version", "")),
        planner_contract_violations=_strings(value.get("planner_contract_violations")),
    )


def _bounded_json_value(value: Any, limit: int) -> Any:
    normalized = _jsonable(value)
    encoded = json.dumps(normalized, ensure_ascii=False, default=str)
    if len(encoded) <= limit:
        return normalized
    return {
        "truncated": True,
        "json_preview": encoded[:limit] + "...[truncated]",
    }


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _historical_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "event",
        "backend",
        "workflow_id",
        "workflow_step_id",
        "tool_name",
        "task_id",
        "request_id",
        "success",
        "result_status",
    )
    return [
        {key: item[key] for key in fields if key in item}
        for item in trace[:64]
        if isinstance(item, dict)
    ]


def _encode_result(
    value: A1TaskResult,
    *,
    historical: bool = False,
) -> dict[str, Any]:
    if historical:
        return {
            "success": value.success,
            "result_status": value.result_status,
            "answer": _bounded_text(value.answer, 4000),
            "output": _bounded_json_value(value.output, 4000),
            "tool_trace": _historical_trace(value.tool_trace),
            "observations": [
                _bounded_json_value(item, 2000) for item in value.observations[:8]
            ],
            "metrics": value.metrics,
            "artifacts": value.artifacts,
            "errors": [_bounded_text(item, 2000) for item in value.errors[:8]],
            "task_metadata": _bounded_json_value(value.task_metadata, 4000),
        }
    return {
        "success": value.success,
        "result_status": value.result_status,
        "answer": value.answer,
        "output": _jsonable(value.output),
        "tool_trace": _records(value.tool_trace),
        "observations": _records(value.observations),
        "metrics": value.metrics,
        "artifacts": value.artifacts,
        "errors": value.errors,
        "task_metadata": _jsonable(value.task_metadata),
    }


def _decode_result(value: dict[str, Any]) -> A1TaskResult:
    return A1TaskResult(
        success=bool(value.get("success", False)),
        result_status=str(value.get("result_status", "error")),
        answer=str(value.get("answer", "")),
        output=value.get("output"),
        tool_trace=_records(value.get("tool_trace")),
        observations=_records(value.get("observations")),
        metrics=_float_mapping(value.get("metrics")),
        artifacts=_strings(value.get("artifacts")),
        errors=_strings(value.get("errors")),
        task_metadata=_mapping(value.get("task_metadata")),
        raw=None,
    )


def _decode_pending_execution(value: dict[str, Any]) -> PendingExecution:
    return PendingExecution(
        step_id=str(value["step_id"]),
        iteration=int(value["iteration"]),
        task_id=str(value["task_id"]),
        request_id=str(value["request_id"]),
        gateway_tool=str(value["gateway_tool"]),
        backend=str(value["backend"]),
        status=str(value.get("status", "submitted")),
        next_poll_at=float(value.get("next_poll_at", 0.0)),
        deadline_at=float(value.get("deadline_at", 0.0)),
        consecutive_poll_errors=int(value.get("consecutive_poll_errors", 0)),
        last_poll_error=str(value.get("last_poll_error", "")),
        task_metadata=_mapping(value.get("task_metadata")),
        remote_status=str(value.get("remote_status", "")),
        rpc_status=str(value.get("rpc_status", "not_started")),
        unknown_reason=str(value.get("unknown_reason", "")),
        wait_started_at=float(value.get("wait_started_at", 0.0) or 0.0),
        action_id=str(value.get("action_id", "")),
        idempotency_key=str(value.get("idempotency_key", "")),
    )


def _decode_scientific_state(value: dict[str, Any]) -> ScientificState:
    return ScientificState(
        task_id=str(value["task_id"]),
        goal=str(value["goal"]),
        state_version=int(value.get("state_version", 0)),
        canonical_entities={
            str(key): CanonicalEntityRecord(
                entity_id=str(item["entity_id"]),
                entity_type=str(item["entity_type"]),
                query_name=str(item["query_name"]),
                preferred_name=str(item["preferred_name"]),
                gene_symbol=str(item.get("gene_symbol", "")),
                aliases=tuple(_strings(item.get("aliases"))),
                uniprot_accession=str(item.get("uniprot_accession", "")),
                organism=str(item.get("organism", "")),
                tax_id=(
                    int(item["tax_id"])
                    if isinstance(item.get("tax_id"), int | str)
                    and str(item.get("tax_id")).isdigit()
                    else None
                ),
                source=str(item.get("source", "")),
                source_url=str(item.get("source_url", "")),
            )
            for key, raw in _mapping(value.get("canonical_entities")).items()
            for item in [_mapping(raw)]
        },
        entity_corrections={
            str(key): str(item)
            for key, item in _mapping(value.get("entity_corrections")).items()
        },
        evidence={
            str(key): EvidenceRecord(
                evidence_id=str(item["evidence_id"]),
                evidence_type=str(item["evidence_type"]),
                summary=str(item["summary"]),
                source_attempt_id=str(item["source_attempt_id"]),
                source_backend=str(item["source_backend"]),
                source_capability_id=str(item.get("source_capability_id", "")),
                provenance=_mapping(item.get("provenance")),
                payload=_mapping(item.get("payload")),
                artifact_refs=tuple(_strings(item.get("artifact_refs"))),
                verifier_id=str(item.get("verifier_id", "")),
            )
            for key, raw in _mapping(value.get("evidence")).items()
            for item in [_mapping(raw)]
        },
        claims={
            str(key): ClaimRecord(
                claim_id=str(item["claim_id"]),
                statement=str(item["statement"]),
                evidence_ids=tuple(_strings(item.get("evidence_ids"))),
                status=ClaimStatus(str(item["status"])),
                source_attempt_id=str(item["source_attempt_id"]),
                verifier_id=str(item["verifier_id"]),
            )
            for key, raw in _mapping(value.get("claims")).items()
            for item in [_mapping(raw)]
        },
        hypotheses={
            str(key): HypothesisRecord(
                hypothesis_id=str(item["hypothesis_id"]),
                statement=str(item["statement"]),
                status=HypothesisStatus(str(item.get("status", "proposed"))),
                supporting_evidence_ids=tuple(
                    _strings(item.get("supporting_evidence_ids"))
                ),
                contradicting_evidence_ids=tuple(
                    _strings(item.get("contradicting_evidence_ids"))
                ),
                uncertainty=str(item.get("uncertainty", "")),
                next_test=str(item.get("next_test", "")),
                version=int(item.get("version", 1)),
            )
            for key, raw in _mapping(value.get("hypotheses")).items()
            for item in [_mapping(raw)]
        },
        attempts={
            str(key): AttemptRecord(
                attempt_id=str(item["attempt_id"]),
                iteration=int(item["iteration"]),
                step_id=str(item["step_id"]),
                objective=str(item["objective"]),
                backend=str(item["backend"]),
                capability_id=str(item["capability_id"]),
                status=AttemptStatus(str(item["status"])),
                result_status=str(item["result_status"]),
                verifier_id=str(item["verifier_id"]),
                reason=str(item.get("reason", "")),
                evidence_ids=tuple(_strings(item.get("evidence_ids"))),
                artifact_refs=tuple(_strings(item.get("artifact_refs"))),
            )
            for key, raw in _mapping(value.get("attempts")).items()
            for item in [_mapping(raw)]
        },
        unresolved_questions=_strings(value.get("unresolved_questions")),
        conflicts=_strings(value.get("conflicts")),
        failed_directions=_strings(value.get("failed_directions")),
        artifact_refs=_strings(value.get("artifact_refs")),
        transition_ids=_strings(value.get("transition_ids")),
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_mapping(item) if isinstance(item, dict) else {"value": str(item)} for item in items]


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }
