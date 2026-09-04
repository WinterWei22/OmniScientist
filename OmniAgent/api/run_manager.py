from __future__ import annotations

import asyncio
import json
import os
import sys
import re
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RUN_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.-]{0,127}$")
DEFAULT_MODELS = {"qwen-plus", "qwen3.7-plus", "qwen3.8-max"}
REDACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "ground_truth",
    "canary",
}
PATH_KEYS = {
    "workspace",
    "workspace_path",
    "manifest_path",
    "output_path",
    "events_path",
    "trajectory_path",
    "task_dir",
    "log_path",
    "benchmark_root",
    "dataset_path",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def public_json(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Project internal records into a bounded response-safe JSON value."""
    normalized_key = key.lower() if key else ""
    if normalized_key in REDACT_KEYS:
        return "[redacted]"
    if normalized_key in PATH_KEYS:
        return Path(str(value)).name if value is not None else None
    if depth > 8:
        return "[depth limited]"
    if isinstance(value, dict):
        return {
            str(k): public_json(v, key=str(k), depth=depth + 1)
            for k, v in value.items()
            if str(k).lower() not in REDACT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [public_json(item, depth=depth + 1) for item in value[:500]]
    if isinstance(value, str) and len(value) > 20000:
        return value[:20000] + "...[truncated]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    task: dict[str, Any]
    model: str
    workspace: Path
    log_path: Path
    process: asyncio.subprocess.Process | None = None
    process_pid: int | None = None
    process_group_id: int | None = None
    process_uid: int | None = None
    harness_run_id: str | None = None
    status: str = "queued"
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    stop_requested_at: str | None = None
    stop_reason: str | None = None


class RunManager:
    """Starts existing DDB runners and exposes only their durable artifacts."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        biomni_canceller: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        self.python = Path(
            os.getenv(
                "OMNIAGENT_API_PYTHON",
                sys.executable,
            )
        )
        self.tasks_path = Path(
            os.getenv(
                "DDB_TASKS_PATH",
                self.project_root / "data" / "DrugDiscoveryBench" / "tasks.jsonl",
            )
        ).expanduser().resolve()
        self.run_root = Path(
            os.getenv("OMNIAGENT_RUN_ROOT", self.project_root / "runs" / "ddb")
        ).expanduser().resolve()
        self.log_root = self.project_root / "logs"
        self.binding_root = self.run_root / ".api_bindings"
        self.records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()
        self._biomni_canceller = biomni_canceller or self._cancel_biomni_task

    def _allowed_models(self) -> set[str]:
        configured = os.getenv("OMNIAGENT_ALLOWED_MODELS", "")
        return {
            item.strip()
            for item in configured.split(",")
            if item.strip()
        } or DEFAULT_MODELS

    def _validate_run_id(self, run_id: str) -> str:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Invalid run_id")
        return run_id

    def _binding_path(self, run_id: str) -> Path:
        return self.binding_root / f"{run_id}.json"

    def _write_binding(self, record: RunRecord) -> None:
        self.binding_root.mkdir(parents=True, exist_ok=True)
        path = self._binding_path(record.run_id)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            "api_run_id": record.run_id,
            "harness_run_id": record.harness_run_id,
            "task_id": record.task_id,
            "model": record.model,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "status": record.status,
            "process_pid": record.process_pid,
            "process_group_id": record.process_group_id,
            "process_uid": record.process_uid,
            "stop_requested_at": record.stop_requested_at,
            "stop_reason": record.stop_reason,
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _read_harness_run_id(self, workspace: Path) -> str | None:
        try:
            with (workspace / "events.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") not in {"run_started", "run_resumed"}:
                        continue
                    payload = event.get("payload")
                    if isinstance(payload, dict) and payload.get("run_id"):
                        return str(payload["run_id"])
        except OSError:
            return None
        return None

    def _read_event_metadata(self, workspace: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        try:
            with (workspace / "events.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_name = event.get("event")
                    timestamp = event.get("timestamp")
                    payload = event.get("payload")
                    if event_name == "runtime_configured" and isinstance(payload, dict):
                        model = payload.get("model")
                        if model:
                            metadata["model"] = str(model)
                        if timestamp:
                            metadata["started_at"] = str(timestamp)
                    elif event_name == "run_finished" and timestamp:
                        metadata["finished_at"] = str(timestamp)
        except OSError:
            return metadata
        return metadata

    def _refresh_harness_run_id(self, record: RunRecord) -> None:
        if record.harness_run_id is None:
            record.harness_run_id = self._read_harness_run_id(record.workspace)
            if record.harness_run_id:
                self._write_binding(record)

    def _task_view(self, item: dict[str, Any]) -> dict[str, Any]:
        task_id = str(item.get("task_id", item.get("id", "")))
        return {
            "task_id": task_id,
            "short_title": str(item.get("short_title", task_id)),
            "capability": item.get("capability"),
            "prompt": str(item.get("prompt", item.get("instruction", ""))).strip(),
        }

    def list_tasks(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if offset < 0 or limit < 1 or limit > 100:
            raise ValueError("offset must be non-negative and limit must be 1..100")
        items: list[dict[str, Any]] = []
        has_more = False
        valid_count = 0
        try:
            handle = self.tasks_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ValueError("DDB tasks.jsonl is unavailable") from exc
        with handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                view = self._task_view(item)
                if not view["task_id"] or not view["prompt"]:
                    continue
                valid_count += 1
                if valid_count <= offset:
                    continue
                if len(items) < limit:
                    items.append(view)
                    continue
                has_more = True
                break
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
        }

    def resolve_task(
        self,
        *,
        task_id: str | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        if (task_id is None) == (index is None):
            raise ValueError("Provide exactly one of task_id or index")
        if index is not None and index < 1:
            raise ValueError("index must be 1-based")
        try:
            handle = self.tasks_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ValueError("DDB tasks.jsonl is unavailable") from exc
        valid_count = 0
        with handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                view = self._task_view(item)
                if not view["task_id"] or not view["prompt"]:
                    continue
                valid_count += 1
                if (task_id is not None and view["task_id"] == task_id) or (
                    index is not None and valid_count == index
                ):
                    return view
        raise ValueError("DDB task was not found")

    def _child_environment(self, model: str, api_key: str | None) -> dict[str, str]:
        env = os.environ.copy()
        for name in ("PYTHONHOME", "LD_PRELOAD", "OMNIAGENT_API_AUTH_TOKEN"):
            env.pop(name, None)
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "OMNIAGENT_PROJECT_ROOT": str(self.project_root),
                "OMNIAGENT_RUN_ROOT": str(self.run_root),
                "DDB_TASKS_PATH": str(self.tasks_path),
                "DDB_ROOT": os.getenv(
                    "DDB_ROOT", str(self.project_root / "data" / "DrugDiscoveryBench")
                ),
                "PYTHONPATH": os.pathsep.join(
                    [str(self.project_root / "src"), str(self.project_root)]
                ),
                "QWEN_PROVIDER": "bailian",
                "BAILIAN_MODEL": model,
                "BIOMNI_MCP_TRANSPORT": os.getenv(
                    "BIOMNI_MCP_TRANSPORT", "streamable_http"
                ),
                "BIOMNI_MCP_URL": os.getenv(
                    "BIOMNI_MCP_URL", "http://127.0.0.1:18000/mcp"
                ),
            }
        )
        if api_key:
            env["DASHSCOPE_API_KEY"] = api_key
        return env

    async def start(
        self,
        *,
        task_id: str | None,
        index: int | None,
        model: str | None,
        api_key: str | None,
    ) -> RunRecord:
        task = self.resolve_task(task_id=task_id, index=index)
        selected_model = (model or os.getenv("BAILIAN_MODEL", "qwen3.8-max")).strip()
        if selected_model not in self._allowed_models():
            raise ValueError("Model is not allowed")
        if not api_key and not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("DashScope API key is required")
        run_id = f"api-{task['task_id']}-{uuid.uuid4().hex[:10]}"
        run_id = self._validate_run_id(run_id)
        workspace = self.run_root / run_id
        log_path = self.log_root / f"{run_id}.console.log"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        command = [
            str(self.python),
            str(self.project_root / "scripts" / "run_ddb_case.py"),
            task["task_id"],
            run_id,
        ]
        try:
            async with self._lock:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.project_root),
                    env=self._child_environment(selected_model, api_key),
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception:
            log_handle.close()
            raise
        record = RunRecord(
            run_id=run_id,
            task_id=task["task_id"],
            task=task,
            model=selected_model,
            workspace=workspace,
            log_path=log_path,
            process=process,
            process_pid=process.pid,
            process_group_id=process.pid,
            process_uid=os.getuid(),
            status="running",
        )
        self.records[run_id] = record
        self._write_binding(record)
        asyncio.create_task(self._watch(record, log_handle))
        return record

    async def _watch(self, record: RunRecord, log_handle: Any) -> None:
        assert record.process is not None
        try:
            record.returncode = await record.process.wait()
            record.finished_at = utc_now()
            record.status = (
                "cancelled" if record.stop_requested_at else self._outcome_status(record)
            )
            self._refresh_harness_run_id(record)
            self._write_binding(record)
        except Exception as exc:
            record.status = "failed"
            record.error = type(exc).__name__
            record.finished_at = utc_now()
        finally:
            log_handle.close()

    def _outcome_status(self, record: RunRecord) -> str:
        if record.returncode != 0:
            return "failed"
        state = read_json(record.workspace / "final_state.json") or {}
        state_status = str(state.get("status", "")).lower()
        return "succeeded" if state_status in {"completed", "succeeded"} else "failed"

    def _record_or_existing(self, run_id: str) -> RunRecord:
        self._validate_run_id(run_id)
        if run_id in self.records:
            return self.records[run_id]
        workspace = self.run_root / run_id
        if not workspace.is_dir():
            raise KeyError(run_id)
        manifest = read_json(workspace / "task_manifest.json") or {}
        binding = read_json(self._binding_path(run_id)) or {}
        event_metadata = self._read_event_metadata(workspace)
        task_id = str(manifest.get("id", "unknown"))
        task = {
            "task_id": task_id,
            "short_title": str(manifest.get("name", task_id)),
            "capability": None,
            "prompt": str(manifest.get("description", "")),
        }
        record = RunRecord(
            run_id=run_id,
            task_id=task_id,
            task=task,
            model=str(binding.get("model") or event_metadata.get("model") or "unknown"),
            workspace=workspace,
            log_path=self.log_root / f"{run_id}.console.log",
            harness_run_id=(
                str(binding["harness_run_id"])
                if binding.get("harness_run_id")
                else self._read_harness_run_id(workspace)
            ),
            process_pid=(
                int(binding["process_pid"]) if binding.get("process_pid") else None
            ),
            process_group_id=(
                int(binding["process_group_id"])
                if binding.get("process_group_id")
                else None
            ),
            process_uid=(
                int(binding["process_uid"])
                if binding.get("process_uid") is not None
                else None
            ),
            started_at=str(
                binding.get("started_at")
                or event_metadata.get("started_at")
                or utc_now()
            ),
            finished_at=(
                str(binding["finished_at"])
                if binding.get("finished_at")
                else event_metadata.get("finished_at")
            ),
            status=(
                str(binding.get("status"))
                if binding.get("status") in {"stopping", "cancelled", "needs_review"}
                else self._outcome_status_from_files(workspace)
            ),
            stop_requested_at=(
                str(binding["stop_requested_at"])
                if binding.get("stop_requested_at")
                else None
            ),
            stop_reason=(
                str(binding["stop_reason"]) if binding.get("stop_reason") else None
            ),
        )
        if record.harness_run_id or binding:
            self._write_binding(record)
        return record

    def _outcome_status_from_files(self, workspace: Path) -> str:
        if not (workspace / "final_state.json").exists():
            return "running" if (workspace / "events.jsonl").exists() else "queued"
        state = read_json(workspace / "final_state.json") or {}
        return (
            "succeeded"
            if str(state.get("status", "")).lower() in {"completed", "succeeded"}
            else "failed"
        )

    def status(self, run_id: str) -> dict[str, Any]:
        record = self._record_or_existing(run_id)
        self._refresh_harness_run_id(record)
        if (
            record.process is not None
            and record.process.returncode is not None
            and not record.stop_requested_at
        ):
            record.status = self._outcome_status(record)
        return {
            "run_id": record.run_id,
            "harness_run_id": record.harness_run_id,
            "task_id": record.task_id,
            "case": public_json(record.task),
            "model": record.model,
            "status": record.status,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "process_exit_code": record.returncode,
            "error": record.error,
            "stop_requested_at": record.stop_requested_at,
            "stop_reason": record.stop_reason,
            "events_url": f"/v1/runs/{record.run_id}/events",
            "result_url": f"/v1/runs/{record.run_id}/result",
        }

    def events(self, run_id: str, cursor: int, limit: int) -> dict[str, Any]:
        record = self._record_or_existing(run_id)
        self._refresh_harness_run_id(record)
        if cursor < 0 or limit < 1 or limit > 200:
            raise ValueError("cursor must be non-negative and limit must be 1..200")
        path = record.workspace / "events.jsonl"
        items: list[dict[str, Any]] = []
        next_cursor = cursor
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    if line_number < cursor:
                        continue
                    next_cursor = line_number + 1
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    projected = public_json(event)
                    if isinstance(projected, dict):
                        projected["api_run_id"] = run_id
                        projected["harness_run_id"] = record.harness_run_id
                    items.append({"cursor": line_number, "event": projected})
                    if len(items) >= limit:
                        break
        except FileNotFoundError:
            pass
        return {
            "run_id": run_id,
            "harness_run_id": record.harness_run_id,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "events": items,
            "has_more": len(items) >= limit,
        }

    def result(self, run_id: str) -> dict[str, Any]:
        record = self._record_or_existing(run_id)
        self._refresh_harness_run_id(record)
        final_state = read_json(record.workspace / "final_state.json")
        submission = read_json(record.workspace / "final_submission.json")
        evaluation = read_json(record.workspace / "benchmark_evaluation.json")
        state_summary: dict[str, Any] | None = None
        if final_state is not None:
            state_summary = {
                "phase": final_state.get("phase"),
                "status": final_state.get("status"),
                "finish_reason": final_state.get("finish_reason"),
                "best_score": final_state.get("best_score"),
                "iterations": len(final_state.get("iterations", [])),
                "a1_call_count": final_state.get("a1_call_count"),
            }
        return {
            "run_id": run_id,
            "harness_run_id": record.harness_run_id,
            "task_id": record.task_id,
            "status": record.status,
            "final_state": public_json(state_summary),
            "final_submission": public_json(submission),
            "benchmark_evaluation": public_json(evaluation),
            "artifacts": [
                {"name": name, "url": f"/v1/runs/{run_id}/result/{name}"}
                for name, present in (
                    ("final_submission.json", submission is not None),
                    ("final_state.json", final_state is not None),
                    ("benchmark_evaluation.json", evaluation is not None),
                )
                if present
            ],
        }

    def artifact(self, run_id: str, name: str) -> dict[str, Any]:
        record = self._record_or_existing(run_id)
        allowed = {
            "final_submission.json",
            "final_state.json",
            "benchmark_evaluation.json",
        }
        if name not in allowed:
            raise KeyError(name)
        value = read_json(record.workspace / name)
        if value is None:
            raise FileNotFoundError(name)
        return public_json(value)

    async def _cancel_biomni_task(
        self, task_id: str, reason: str
    ) -> dict[str, Any]:
        from local_deep_research.closed_loop.biomni_gateway import BiomniGatewayClient

        client = BiomniGatewayClient(
            {
                "transport": os.getenv("BIOMNI_MCP_TRANSPORT", "streamable_http"),
                "url": os.getenv(
                    "BIOMNI_MCP_URL", "http://127.0.0.1:18000/mcp"
                ),
            }
        )
        return await client.cancel_task(task_id, reason)

    def _append_api_event(
        self, record: RunRecord, event: str, payload: dict[str, Any]
    ) -> None:
        record.workspace.mkdir(parents=True, exist_ok=True)
        item = {
            "schema_version": "omniagent.v1",
            "timestamp": utc_now(),
            "event": event,
            "payload": payload,
        }
        with (record.workspace / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _active_biomni_task_ids(self, workspace: Path) -> list[str]:
        terminal = {
            "succeeded",
            "completed",
            "failed",
            "cancelled",
            "dead_letter",
            "manual_review",
            "timed_out",
        }
        active: dict[str, None] = {}

        def add(value: Any) -> None:
            task_id = str(value or "").strip()
            if task_id:
                active[task_id] = None

        def remove(value: Any) -> None:
            active.pop(str(value or "").strip(), None)

        try:
            with (workspace / "events.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event = str(item.get("event") or "")
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if event == "external_task_submitted":
                        add(payload.get("task_id"))
                    elif event == "external_task_stage_changed":
                        remove(payload.get("previous_task_id"))
                        add(payload.get("task_id"))
                    elif event in {"external_task_completed", "execution_completed"}:
                        remove(payload.get("task_id"))
                        biomni_task = payload.get("biomni_task")
                        if isinstance(biomni_task, dict):
                            remove(biomni_task.get("task_id"))
                    elif event == "action_state_changed":
                        status = str(payload.get("status") or "").lower()
                        if status in terminal or payload.get("terminal") is True:
                            remove(payload.get("task_id"))
                        else:
                            add(payload.get("task_id"))

        except OSError:
            pass

        state = read_json(workspace / "final_state.json") or {}
        pending = state.get("pending_execution")
        if isinstance(pending, dict):
            status = str(pending.get("status") or "").lower()
            if status not in terminal:
                add(pending.get("task_id"))
        ledger = state.get("action_ledger")
        records = ledger.get("records") if isinstance(ledger, dict) else None
        if isinstance(records, list):
            for item in records:
                if not isinstance(item, dict):
                    continue
                task_id = item.get("external_task_id")
                if str(item.get("status") or "").lower() in terminal:
                    remove(task_id)
                else:
                    add(task_id)
        return list(active)

    def _persisted_process_identity(self, record: RunRecord) -> tuple[bool, str]:
        pid = record.process_pid
        if not pid:
            return False, "process PID was not persisted"
        process_path = Path("/proc") / str(pid)
        try:
            owner_uid = process_path.stat().st_uid
            command = (process_path / "cmdline").read_bytes().replace(b"\x00", b" ")
        except FileNotFoundError:
            return False, "process is no longer running"
        except OSError:
            return False, "process identity could not be read"
        expected_uid = record.process_uid
        if expected_uid is not None and owner_uid != expected_uid:
            return False, "persisted process owner no longer matches the run binding"
        if owner_uid != os.getuid():
            return False, (
                f"process belongs to uid {owner_uid}, but OmniAgent API runs as "
                f"uid {os.getuid()}"
            )
        if b"scripts/run_ddb_case.py" not in command or record.run_id.encode() not in command:
            return False, "persisted PID does not belong to this OmniAgent run"
        return True, ""

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _wait_for_process_group_exit(
        self, process_group_id: int, timeout: float
    ) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while self._process_group_exists(process_group_id):
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.1)
        return True

    async def stop(self, run_id: str, reason: str) -> dict[str, Any]:
        reason = str(reason or "user requested stop").strip()
        if not reason:
            raise ValueError("Stop reason must not be empty")
        async with self._lock:
            record = self._record_or_existing(run_id)
            terminal = {"succeeded", "failed", "cancelled"}
            if record.status in terminal:
                return {
                    "run_id": record.run_id,
                    "harness_run_id": record.harness_run_id,
                    "status": record.status,
                    "already_terminal": True,
                    "reason": reason,
                    "omniagent": {"stop_requested": False, "forced": False},
                    "biomni": {
                        "task_ids": [],
                        "requested": 0,
                        "cancelled": 0,
                        "tasks": [],
                        "errors": [],
                        "remote_cancellation_complete": True,
                    },
                    "stopped_at": record.finished_at,
                }

            record.stop_requested_at = utc_now()
            record.stop_reason = reason
            record.status = "stopping"
            self._write_binding(record)
            self._append_api_event(
                record,
                "api_stop_requested",
                {"run_id": run_id, "reason": reason},
            )

            signal_sent = False
            signal_name: str | None = None
            local_stop_error: str | None = None
            pid = record.process.pid if record.process is not None else None
            pgid = record.process_group_id or pid
            process_already_exited = (
                record.process is not None and record.process.returncode is not None
            )
            may_signal = record.process is not None and not process_already_exited
            if record.process is None:
                identity_valid, identity_error = self._persisted_process_identity(record)
                may_signal = identity_valid
                if identity_error == "process is no longer running":
                    process_already_exited = True
                elif not identity_valid:
                    local_stop_error = identity_error
            if pgid and may_signal:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    signal_sent = True
                    signal_name = "SIGTERM"
                except ProcessLookupError:
                    process_already_exited = True
                except PermissionError as exc:
                    local_stop_error = f"PermissionError: {exc}"

            # Scan after SIGTERM as well, so the last durable task identity is included.
            task_ids = self._active_biomni_task_ids(record.workspace)
            task_results: list[dict[str, Any]] = []
            errors: list[dict[str, str]] = []
            cancel_timeout = max(
                0.1,
                float(os.getenv("OMNIAGENT_BIOMNI_CANCEL_TIMEOUT_SECONDS", "15")),
            )
            for task_id in task_ids:
                try:
                    result = await asyncio.wait_for(
                        self._biomni_canceller(task_id, reason),
                        timeout=cancel_timeout,
                    )
                    task_results.append(public_json(result))
                except Exception as exc:
                    errors.append(
                        {
                            "task_id": task_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            forced = False
            local_stop_complete = process_already_exited
            if signal_sent and pgid:
                grace = max(0.1, float(os.getenv("OMNIAGENT_STOP_GRACE_SECONDS", "10")))
                loop = asyncio.get_running_loop()
                graceful_deadline = loop.time() + grace
                if record.process is not None and record.process.returncode is None:
                    try:
                        await asyncio.wait_for(record.process.wait(), timeout=grace)
                    except TimeoutError:
                        pass
                remaining_grace = max(0.0, graceful_deadline - loop.time())
                local_stop_complete = await self._wait_for_process_group_exit(
                    pgid, remaining_grace
                )
                if not local_stop_complete:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        forced = True
                        signal_name = "SIGKILL"
                    except ProcessLookupError:
                        local_stop_complete = True
                    except PermissionError as exc:
                        local_stop_error = f"PermissionError: {exc}"
                    if not local_stop_error:
                        local_stop_complete = await self._wait_for_process_group_exit(
                            pgid, 2.0
                        )
            if record.process is not None and record.process.returncode is None:
                try:
                    await asyncio.wait_for(record.process.wait(), timeout=2.0)
                except TimeoutError:
                    local_stop_complete = False
                    local_stop_error = local_stop_error or "runner did not exit after signals"
            record.returncode = (
                record.process.returncode
                if record.process is not None
                else record.returncode
            )
            if not local_stop_complete and local_stop_error is None:
                local_stop_error = "runner process group is still active"
            record.status = "cancelled" if local_stop_complete else "needs_review"
            record.finished_at = utc_now()
            self._refresh_harness_run_id(record)
            self._append_api_event(
                record,
                "api_stop_completed",
                {
                    "run_id": run_id,
                    "reason": reason,
                    "biomni_task_ids": task_ids,
                    "biomni_cancel_errors": errors,
                    "local_stop_complete": local_stop_complete,
                    "local_stop_error": local_stop_error,
                },
            )
            self._write_binding(record)
            cancelled_count = sum(
                str(item.get("status") or "").lower() == "cancelled"
                for item in task_results
                if isinstance(item, dict)
            )
            return {
                "run_id": record.run_id,
                "harness_run_id": record.harness_run_id,
                "status": record.status,
                "already_terminal": False,
                "reason": reason,
                "omniagent": {
                    "stop_requested": True,
                    "signal_sent": signal_sent,
                    "signal": signal_name,
                    "forced": forced,
                    "process_exit_code": record.returncode,
                    "local_stop_complete": local_stop_complete,
                    "error": local_stop_error,
                },
                "biomni": {
                    "task_ids": task_ids,
                    "requested": len(task_ids),
                    "cancelled": cancelled_count,
                    "tasks": task_results,
                    "errors": errors,
                    "remote_cancellation_complete": not errors,
                },
                "stopped_at": record.finished_at,
            }
