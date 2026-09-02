"""Security boundary for running the external ProteinMD workflow.

This module deliberately accepts declarative data only.  It never accepts a
command, arbitrary argv, environment variables, or user-selected filesystem
roots.  Public Biomni tools call :func:`prepare_request`, :func:`submit_request`
and :func:`get_request`; the queue worker calls :func:`execute_request` only
after acquiring the exact GPU resource recorded in the task.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

INTERNAL_TOOL_NAME = "biomni.tool._proteinmd_task._execute_proteinmd_inference"
_SYSTEM_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
_TOKEN_RE = re.compile(r"^[a-f0-9]{64}$")
_TERMINAL_TASK_STATES = {"succeeded", "failed", "dead_letter", "manual_review"}


class ProteinMDExecutionCancelledError(RuntimeError):
    """A controlled cancellation that must not be treated as a retryable timeout."""


@dataclass(frozen=True)
class ExecutionProfile:
    frames: int
    max_frames: int
    inference_steps: int
    timeout_seconds: int
    benchmark: bool


PROFILES = {
    "smoke": ExecutionProfile(1, 2, 1, 600, False),
    "standard": ExecutionProfile(250, 250, 20, 3600, True),
    "production": ExecutionProfile(250, 250, 50, 7200, True),
}


class ProteinMDRequest(BaseModel):
    """Strict request model populated from A1-generated tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: Literal["atlas", "misato"]
    system_id: str
    execution_profile: Literal["smoke", "standard", "production"] = "smoke"
    replica: Literal[1, 2, 3] = 1
    pdb_only: bool = False
    seed: int = Field(default=0, ge=0, le=2_147_483_647)

    @field_validator("system_id")
    @classmethod
    def validate_system_id(cls, value: str) -> str:
        if not _SYSTEM_ID_RE.fullmatch(value) or value in {".", ".."} or value.startswith("-"):
            raise ValueError("system_id may contain only letters, digits, '_' and '-', and cannot begin with '-'")
        return value


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_token: str
    gpu_policy: Literal["auto"] = "auto"

    @field_validator("validation_token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if not _TOKEN_RE.fullmatch(value):
            raise ValueError("invalid ProteinMD validation token")
        return value


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_token: str
    gpu_device: int = Field(ge=0, le=1024)

    @field_validator("validation_token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if not _TOKEN_RE.fullmatch(value):
            raise ValueError("invalid ProteinMD validation token")
        return value


@dataclass(frozen=True)
class ProteinMDSettings:
    proteinmd_root: Path
    protenix_root: Path
    protenix_checkpoint: Path
    checkpoint_root: Path
    ccd_cif: Path
    ccd_rdkit: Path
    atlas_input_root: Path
    misato_input_root: Path
    legacy_data_root: Path
    state_root: Path
    gpu_devices: tuple[int, ...]
    token_ttl_seconds: int
    termination_grace_seconds: int
    max_input_bytes: int
    max_output_bytes: int
    sandbox_mode: Literal["required", "disabled"]
    bwrap_path: Path

    @classmethod
    def from_env(cls) -> ProteinMDSettings:
        required_paths = {
            name: os.getenv(name, "").strip()
            for name in (
                "BIOMNI_PROTEINMD_ROOT",
                "BIOMNI_PROTENIX_ROOT",
                "BIOMNI_PROTENIX_CHECKPOINT",
                "BIOMNI_PROTEINMD_CHECKPOINT_ROOT",
                "BIOMNI_PROTEINMD_CCD_CIF",
                "BIOMNI_PROTEINMD_CCD_RDKIT",
                "BIOMNI_PROTEINMD_ATLAS_INPUT_ROOT",
                "BIOMNI_PROTEINMD_MISATO_INPUT_ROOT",
                "BIOMNI_PROTEINMD_LEGACY_DATA_ROOT",
                "BIOMNI_PROTEINMD_STATE_ROOT",
                "BIOMNI_PROTEINMD_BWRAP",
            )
        }
        missing = [name for name, value in required_paths.items() if not value]
        if missing:
            raise ValueError("missing required ProteinMD deployment variables: " + ", ".join(missing))
        gpu_text = os.getenv("BIOMNI_PROTEINMD_GPU_DEVICES", "").strip()
        if not gpu_text:
            raise ValueError("BIOMNI_PROTEINMD_GPU_DEVICES is required")
        try:
            gpu_devices = tuple(sorted({int(item.strip()) for item in gpu_text.split(",") if item.strip()}))
        except ValueError as exc:
            raise ValueError("BIOMNI_PROTEINMD_GPU_DEVICES must be comma-separated integers") from exc
        if not gpu_devices or any(device < 0 or device > 1024 for device in gpu_devices):
            raise ValueError("at least one ProteinMD GPU in the range 0..1024 must be configured")
        sandbox_mode = os.getenv("BIOMNI_PROTEINMD_SANDBOX_MODE", "required").strip().lower()
        if sandbox_mode not in {"required", "disabled"}:
            raise ValueError("BIOMNI_PROTEINMD_SANDBOX_MODE must be 'required' or 'disabled'")
        return cls(
            proteinmd_root=Path(required_paths["BIOMNI_PROTEINMD_ROOT"]),
            protenix_root=Path(required_paths["BIOMNI_PROTENIX_ROOT"]),
            protenix_checkpoint=Path(required_paths["BIOMNI_PROTENIX_CHECKPOINT"]),
            checkpoint_root=Path(required_paths["BIOMNI_PROTEINMD_CHECKPOINT_ROOT"]),
            ccd_cif=Path(required_paths["BIOMNI_PROTEINMD_CCD_CIF"]),
            ccd_rdkit=Path(required_paths["BIOMNI_PROTEINMD_CCD_RDKIT"]),
            atlas_input_root=Path(required_paths["BIOMNI_PROTEINMD_ATLAS_INPUT_ROOT"]),
            misato_input_root=Path(required_paths["BIOMNI_PROTEINMD_MISATO_INPUT_ROOT"]),
            legacy_data_root=Path(required_paths["BIOMNI_PROTEINMD_LEGACY_DATA_ROOT"]),
            state_root=Path(required_paths["BIOMNI_PROTEINMD_STATE_ROOT"]),
            gpu_devices=gpu_devices,
            token_ttl_seconds=int(os.getenv("BIOMNI_PROTEINMD_TOKEN_TTL_SECONDS", "900")),
            termination_grace_seconds=int(os.getenv("BIOMNI_PROTEINMD_TERMINATION_GRACE_SECONDS", "60")),
            max_input_bytes=int(os.getenv("BIOMNI_PROTEINMD_MAX_INPUT_BYTES", str(20 * 1024**3))),
            max_output_bytes=int(os.getenv("BIOMNI_PROTEINMD_MAX_OUTPUT_BYTES", str(50 * 1024**3))),
            sandbox_mode=sandbox_mode,
            bwrap_path=Path(required_paths["BIOMNI_PROTEINMD_BWRAP"]),
        )


def _utc_now() -> datetime:
    # datetime.UTC is unavailable in the supported Python 3.10 deployment.
    return datetime.now(timezone.utc)  # noqa: UP017


def _caller_id() -> str:
    return os.getenv("BIOMNI_TASK_ID", "local-direct-call")


def _ensure_regular_under_root(path: Path, root: Path, max_bytes: int) -> Path:
    expanded_root = root.expanduser()
    expanded_path = path.expanduser()
    try:
        relative = expanded_path.relative_to(expanded_root)
    except ValueError as exc:
        raise ValueError(f"input path is not lexically under the managed root: {path}") from exc
    cursor = expanded_root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"input path contains a forbidden symbolic link: {cursor}")
    root_resolved = expanded_root.resolve(strict=True)
    path_resolved = expanded_path.resolve(strict=True)
    if not path_resolved.is_relative_to(root_resolved):
        raise ValueError(f"input path escapes managed root: {path}")
    info = path_resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"input is not a regular file: {path_resolved}")
    if info.st_size > max_bytes:
        raise ValueError(f"input exceeds the configured size limit: {path_resolved}")
    return path_resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"managed state path is not a real directory: {path}")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError(f"managed state directory must be owned by the worker and mode 0700: {path}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"managed state file is not a regular file: {path}")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise PermissionError(f"managed state file must be owned by the worker and mode 0600: {path}")
    if info.st_size > 1024 * 1024:
        raise ValueError(f"managed state file exceeds the size limit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"managed state file must contain a JSON object: {path}")
    return payload


def _token_path(token: str, settings: ProteinMDSettings) -> Path:
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError("invalid ProteinMD validation token")
    return settings.state_root.expanduser().resolve() / "tokens" / f"{token}.json"


def _load_token(
    token: str,
    settings: ProteinMDSettings,
    *,
    allow_consumed_expired: bool = False,
) -> dict[str, Any]:
    path = _token_path(token, settings)
    try:
        payload = _read_private_json(path)
    except FileNotFoundError as exc:
        raise ValueError("ProteinMD validation token was not found") from exc
    expires_at = datetime.fromisoformat(str(payload["expires_at"]))
    if _utc_now() >= expires_at and not (allow_consumed_expired and payload.get("consumed")):
        raise ValueError("ProteinMD validation token has expired")
    return payload


def _task_binding_path(task_id: str, settings: ProteinMDSettings) -> Path:
    canonical = str(uuid.UUID(str(task_id)))
    return settings.state_root.expanduser().resolve() / "tasks" / f"{canonical}.json"


def _gpu_device_available(device: int) -> bool:
    path = Path(f"/dev/nvidia{device}")
    return path.exists() and stat.S_ISCHR(path.stat().st_mode)


def _validate_installation(settings: ProteinMDSettings, request: ProteinMDRequest) -> None:
    required = [
        settings.proteinmd_root / "scripts/infer_local_case.sh",
        settings.proteinmd_root / ".venv/bin/python",
        settings.protenix_root,
        settings.protenix_checkpoint,
        settings.checkpoint_root / request.dataset / "coarse.pt",
        settings.checkpoint_root / request.dataset / "fine.pt",
    ]
    if request.dataset == "atlas" and not request.pdb_only:
        required.extend([settings.ccd_cif, settings.ccd_rdkit])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing ProteinMD runtime assets: " + ", ".join(missing))
    for executable in (
        settings.proteinmd_root / "scripts/infer_local_case.sh",
        settings.proteinmd_root / ".venv/bin/python",
    ):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise PermissionError(f"ProteinMD executable is not a regular executable file: {executable}")
    if settings.sandbox_mode == "required":
        if (
            settings.bwrap_path.is_symlink()
            or not settings.bwrap_path.is_file()
            or not os.access(settings.bwrap_path, os.X_OK)
        ):
            raise PermissionError(
                f"ProteinMD sandbox executable must be a non-symlink executable file: {settings.bwrap_path}"
            )

    missing_gpu_devices = [device for device in settings.gpu_devices if not _gpu_device_available(device)]
    if missing_gpu_devices:
        raise FileNotFoundError(f"configured ProteinMD GPU device nodes are unavailable: {missing_gpu_devices}")

    workspace_text = os.getenv("BIOMNI_TASK_DIR", "").strip() or os.getenv("BIOMNI_TASK_ROOT", "").strip()
    if not workspace_text:
        raise ValueError("BIOMNI_TASK_DIR or BIOMNI_TASK_ROOT is required for ProteinMD disk preflight")
    workspace = Path(workspace_text).expanduser()
    while not workspace.exists() and workspace != workspace.parent:
        workspace = workspace.parent
    free_bytes = shutil.disk_usage(workspace.resolve(strict=True)).free
    if free_bytes < settings.max_output_bytes:
        raise OSError(
            f"ProteinMD task filesystem has {free_bytes} free bytes; at least {settings.max_output_bytes} are required"
        )


def _input_manifest(request: ProteinMDRequest, settings: ProteinMDSettings) -> dict[str, Any]:
    files: dict[str, Path]
    if request.dataset == "atlas":
        system_root = settings.atlas_input_root / request.system_id
        files = {"pdb": system_root / f"{request.system_id}.pdb"}
        if not request.pdb_only:
            files["xtc"] = system_root / f"{request.system_id}_prod_R{request.replica}_fit.xtc"
        root = settings.atlas_input_root
    else:
        raw_root = settings.misato_input_root / request.system_id
        if (raw_root / "complex.pdb").is_file() and (raw_root / "ligand.sdf").is_file():
            files = {"complex_pdb": raw_root / "complex.pdb", "ligand_sdf": raw_root / "ligand.sdf"}
            root = settings.misato_input_root
        else:
            cache = settings.legacy_data_root / "misato_public_compatible_v3/systems" / request.system_id
            files = {
                "positions": cache / "positions.npy",
                "static_features": cache / "static_features.pt",
                "metadata": cache / "metadata.json",
            }
            root = settings.legacy_data_root
    manifest: dict[str, Any] = {}
    for name, candidate in files.items():
        path = _ensure_regular_under_root(candidate, root, settings.max_input_bytes)
        manifest[f"{name}_path"] = str(path)
        manifest[f"{name}_size_bytes"] = path.stat().st_size
        manifest[f"{name}_sha256"] = _sha256(path)
    return manifest


def prepare_request(**arguments: Any) -> dict[str, Any]:
    try:
        settings = ProteinMDSettings.from_env()
        request = ProteinMDRequest.model_validate(arguments)
        _validate_installation(settings, request)
        manifest = _input_manifest(request, settings)
        token = secrets.token_hex(32)
        now = _utc_now()
        payload = {
            "token": token,
            "caller_id": _caller_id(),
            "request": request.model_dump(),
            "profile": asdict(PROFILES[request.execution_profile]),
            "input_manifest": manifest,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=settings.token_ttl_seconds)).isoformat(),
            "consumed": False,
            "child_task_id": None,
        }
        _atomic_json(_token_path(token, settings), payload)
        return {
            "success": True,
            "execution_spec": request.model_dump(),
            "input_manifest": manifest,
            "validation_token": token,
            "expires_at": payload["expires_at"],
        }
    except (ValidationError, ValueError, FileNotFoundError, OSError) as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def _task_service() -> Any:
    try:
        from tutorials.examples.expose_biomni_server.task_queue import TaskService
    except ImportError as exc:
        raise RuntimeError("Biomni task backend is unavailable in this deployment") from exc
    return TaskService.from_env()


def _select_gpu(token: str, devices: tuple[int, ...]) -> int:
    # Deterministic distribution; the queue's exact gpu:N lease is the authority.
    return devices[int(token[:16], 16) % len(devices)]


def submit_request(validation_token: str, gpu_policy: str = "auto") -> dict[str, Any]:
    try:
        settings = ProteinMDSettings.from_env()
        submission = SubmitRequest.model_validate({"validation_token": validation_token, "gpu_policy": gpu_policy})
        token_path = _token_path(submission.validation_token, settings)
        lock_path = token_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            token_record = _load_token(submission.validation_token, settings)
            if token_record["caller_id"] != _caller_id():
                raise PermissionError("validation token belongs to another caller")
            if token_record.get("consumed"):
                raise ValueError("validation token has already been consumed")
            gpu_device = _select_gpu(submission.validation_token, settings.gpu_devices)
            token_record["consumed"] = True
            token_record["submission_state"] = "submitting"
            token_record["gpu_device"] = gpu_device
            _atomic_json(token_path, token_record)
            service = _task_service()
            try:
                submitted = service.submit_tool(
                    INTERNAL_TOOL_NAME,
                    {"validation_token": submission.validation_token, "gpu_device": gpu_device},
                    None,
                    f"proteinmd:{submission.validation_token}",
                )
            except Exception as exc:
                token_record["consumed"] = False
                token_record["submission_state"] = "failed"
                _atomic_json(token_path, token_record)
                raise RuntimeError(f"ProteinMD task submission failed: {type(exc).__name__}: {exc}") from exc
            if not submitted.get("ok"):
                token_record["consumed"] = False
                token_record["submission_state"] = "failed"
                _atomic_json(token_path, token_record)
                return {"success": False, "error": submitted.get("error", "task submission failed")}
            token_record["child_task_id"] = str(submitted["task_id"])
            token_record["submission_state"] = "submitted"
            token_record["submitted_at"] = _utc_now().isoformat()
            _atomic_json(token_path, token_record)
            _atomic_json(
                _task_binding_path(str(submitted["task_id"]), settings),
                {
                    "task_id": str(submitted["task_id"]),
                    "caller_id": token_record["caller_id"],
                    "validation_token": submission.validation_token,
                },
            )
        return {
            "success": True,
            "task_id": str(submitted["task_id"]),
            "status": submitted.get("status", "queued"),
            "gpu_policy": "auto",
        }
    except (ValidationError, ValueError, PermissionError, RuntimeError, OSError) as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def get_request(task_id: str) -> dict[str, Any]:
    try:
        canonical = str(uuid.UUID(str(task_id)))
        settings = ProteinMDSettings.from_env()
        binding_path = _task_binding_path(canonical, settings)
        try:
            binding = _read_private_json(binding_path)
        except FileNotFoundError as exc:
            raise PermissionError("task is not a ProteinMD child task visible to this caller") from exc
        if binding.get("caller_id") != _caller_id():
            raise PermissionError("ProteinMD task belongs to another caller")
        service = _task_service()
        result = service.get(canonical)
        if not result.get("ok"):
            return {"success": False, "task_id": canonical, "error": result.get("error", "task lookup failed")}
        task_result = result.get("result")
        if (
            isinstance(task_result, dict)
            and task_result.get("protocol") == "biomni.result.v1"
            and isinstance(task_result.get("output"), dict)
        ):
            task_result = task_result["output"]
        # The task directory is parent-scoped by the backend; do not return arbitrary
        # binary content, only the normalised task status and result references.
        return {
            "success": True,
            "task_id": canonical,
            "status": result.get("status"),
            "result": task_result if result.get("status") in _TERMINAL_TASK_STATES else None,
            "error": result.get("error"),
            "log_path": result.get("log_path"),
        }
    except (ValueError, PermissionError, RuntimeError) as exc:
        return {"success": False, "task_id": str(task_id), "error": f"{type(exc).__name__}: {exc}"}


def _allocated_gpu() -> int:
    value = os.getenv("BIOMNI_ALLOCATED_GPU")
    if value is None or not value.isdigit():
        raise PermissionError("ProteinMD execution requires a worker-provided GPU lease")
    return int(value)


def _attempt_directories() -> tuple[Path, Path, Path, Path]:
    raw = os.getenv("BIOMNI_ATTEMPT_DIR")
    if not raw:
        raise RuntimeError("ProteinMD execution requires a task attempt directory")
    attempt = Path(raw).expanduser().resolve(strict=True)
    work = attempt / "work"
    output = attempt / "output"
    temporary = attempt / "tmp"
    for path in (work, output, temporary):
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
        if path.is_symlink() or not path.resolve().is_relative_to(attempt):
            raise ValueError("attempt path escapes the managed task directory")
    if any(output.iterdir()):
        raise ValueError("ProteinMD attempt output directory must be empty before execution")
    return attempt, work, output, temporary


def _runtime_environment(
    settings: ProteinMDSettings,
    request: ProteinMDRequest,
    profile: ExecutionProfile,
    gpu_device: int,
    work: Path,
    output: Path,
    temporary: Path,
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(temporary),
        "TMPDIR": str(temporary),
        "PYTHON_ENV": str(settings.proteinmd_root / ".venv"),
        "PROTENIX_ROOT": str(settings.protenix_root),
        "PROTENIX_CHECKPOINT": str(settings.protenix_checkpoint),
        "CHECKPOINT_ROOT": str(settings.checkpoint_root),
        "CCD_CIF": str(settings.ccd_cif),
        "CCD_RDKIT": str(settings.ccd_rdkit),
        "ATLAS_INPUT_ROOT": str(settings.atlas_input_root),
        "MISATO_INPUT_ROOT": str(settings.misato_input_root),
        "DATA_ROOT": str(settings.legacy_data_root),
        "WORK_ROOT": str(work),
        "OUTPUT_ROOT": str(output),
        "GPU_ID": "0",
        "CUDA_VISIBLE_DEVICES": str(gpu_device),
        "FRAMES": str(profile.frames),
        "MAX_FRAMES": str(profile.max_frames),
        "INFERENCE_STEPS": str(profile.inference_steps),
        "SEED": str(request.seed),
        "REPLICA": str(request.replica),
        "PDB_ONLY": "1" if request.pdb_only else "0",
        "FINE_SEGMENT_BATCH_SIZE": "1",
        "MPLCONFIGDIR": str(temporary / "matplotlib"),
    }
    if not profile.benchmark:
        environment["MAX_TOKENS"] = "64"
    return environment


def _gpu_device_nodes(gpu_device: int) -> list[Path]:
    candidates = [
        Path(f"/dev/nvidia{gpu_device}"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev/nvidia-uvm-tools"),
        Path("/dev/nvidia-modeset"),
    ]
    nodes: list[Path] = []
    for path in candidates:
        if path.exists() and stat.S_ISCHR(path.stat().st_mode):
            nodes.append(path)
    caps_root = Path("/dev/nvidia-caps")
    if caps_root.is_dir():
        nodes.extend(
            path
            for path in sorted(caps_root.iterdir())
            if stat.S_ISCHR(path.stat().st_mode)
        )
    if not nodes or nodes[0] != candidates[0]:
        raise FileNotFoundError(f"allocated GPU device node is unavailable: {candidates[0]}")
    return nodes


def _sandboxed_command(
    settings: ProteinMDSettings,
    command: list[str],
    gpu_device: int,
    work: Path,
    output: Path,
    temporary: Path,
) -> list[str]:
    if settings.sandbox_mode == "disabled":
        return command
    sandbox = [
        str(settings.bwrap_path.resolve(strict=True)),
        "--die-with-parent",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
    ]
    device_nodes = _gpu_device_nodes(gpu_device)
    for parent in sorted({device.parent for device in device_nodes if device.parent != Path("/dev")}):
        sandbox.extend(("--dir", str(parent)))
    for device in device_nodes:
        sandbox.extend(("--dev-bind", str(device), str(device)))
    for writable in (work, output, temporary):
        resolved = writable.resolve(strict=True)
        sandbox.extend(("--bind", str(resolved), str(resolved)))
    sandbox.extend(("--chdir", str(settings.proteinmd_root.resolve(strict=True)), "--", *command))
    return sandbox


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: int) -> dict[str, Any]:
    """Terminate and reap a whole ProteinMD process group.

    The function is idempotent and is also used by tests with a synthetic child
    tree.  A successful return guarantees that the process leader was reaped and
    no member of its process group remains observable.
    """

    pgid = process.pid
    signals_sent: list[str] = []
    if process.poll() is None or _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
            signals_sent.append("SIGTERM")
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0, grace_seconds)
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(pgid):
            break
        time.sleep(0.05)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            signals_sent.append("SIGKILL")
        except ProcessLookupError:
            pass
    # Do not return (and therefore do not release the enclosing GPU lease)
    # until the leader is reaped and the process group is actually gone. If a
    # kernel-uninterruptible process cannot die, the worker intentionally stays
    # inside the lease; systemd/cgroup shutdown is the final containment layer.
    process.wait()
    while _process_group_exists(pgid):
        time.sleep(0.05)
    return {
        "signals_sent": signals_sent,
        "returncode": process.returncode,
        "process_group_exited": not _process_group_exists(pgid),
    }


def _run_process(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
    cancel_files: tuple[Path, ...],
    grace_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        reason: str | None = None
        try:
            deadline = started + timeout_seconds
            while process.poll() is None:
                if any(path.exists() for path in cancel_files):
                    reason = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                time.sleep(0.25)
        except BaseException:
            terminate_process_group(process, grace_seconds)
            raise
        termination = None
        if reason:
            termination = terminate_process_group(process, grace_seconds)
        else:
            process.wait()
            if _process_group_exists(process.pid):
                reason = "straggler_processes"
                termination = terminate_process_group(process, grace_seconds)
        return {
            "exit_code": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "termination_reason": reason,
            "termination": termination,
        }


def _directory_size(root: Path) -> int:
    size = 0
    resolved_root = root.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"ProteinMD output contains a forbidden symbolic link: {path}")
        if path.is_file():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(resolved_root):
                raise ValueError(f"ProteinMD output escapes the attempt directory: {path}")
            size += path.stat().st_size
    return size


def _find_atlas_result(output: Path, system_id: str, replica: int) -> tuple[Path, Path]:
    candidates = list(output.rglob(f"{system_id}/R{replica}/seed_*.json"))
    if not candidates:
        raise ValueError("ProteinMD Atlas output JSON was not produced for the requested system and replica")
    if len(candidates) != 1:
        raise ValueError("ProteinMD Atlas output is ambiguous for the requested system and replica")
    metadata = candidates[0]
    coordinates = metadata.with_suffix(".npz")
    if not coordinates.is_file() or coordinates.stat().st_size == 0:
        raise ValueError("ProteinMD Atlas output NPZ was not produced")
    return metadata, coordinates


def _validate_outputs(
    output: Path,
    request: ProteinMDRequest,
    profile: ExecutionProfile,
) -> dict[str, Any]:
    if request.dataset == "atlas":
        metadata_path, npz_path = _find_atlas_result(output, request.system_id, request.replica)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "system_id": request.system_id,
            "replica": request.replica,
            "seed": request.seed,
            "requested_frames": profile.frames,
            "inference_steps": profile.inference_steps,
        }
        mismatched = {
            name: {"expected": expected, "actual": metadata.get(name)}
            for name, expected in expected_metadata.items()
            if metadata.get(name) != expected
        }
        if mismatched or metadata.get("valid") is not True:
            raise ValueError(f"Atlas output metadata does not match the validated request: {mismatched}")
        with np.load(npz_path, allow_pickle=False) as archive:
            positions = archive["positions"]
            if (
                positions.ndim != 3
                or positions.shape[2] != 3
                or positions.shape[0] != profile.frames
                or positions.dtype != np.float32
                or not np.isfinite(positions).all()
            ):
                raise ValueError("Atlas output positions violate the profile frame/dtype/finite-value contract")
            frame_count, atom_count = int(positions.shape[0]), int(positions.shape[1])
        result = {
            "metadata_path": str(metadata_path),
            "npz_path": str(npz_path),
            "frame_count": frame_count,
            "atom_count": atom_count,
        }
        if profile.benchmark:
            benchmark = output / "atlas" / request.system_id / "trajectory"
            pdb = benchmark / f"{request.system_id}.pdb"
            xtc = benchmark / f"{request.system_id}.xtc"
            if not pdb.is_file() or not xtc.is_file():
                raise ValueError("Atlas benchmark PDB/XTC output is incomplete")
            result.update({"pdb_path": str(pdb), "xtc_path": str(xtc)})
        return result
    result_root = output / "misato" / request.system_id
    required = (
        "COMPLETE",
        "trajectory.pdb",
        "trajectory.xtc",
        "trajectory.npz",
        "identity.npz",
        "metadata.json",
        "topology.pdb",
        "internal_anchors.npz",
    )
    missing = [name for name in required if not (result_root / name).is_file()]
    if missing:
        raise ValueError("MISATO output is incomplete: " + ", ".join(missing))
    empty = [name for name in required if name != "COMPLETE" and (result_root / name).stat().st_size == 0]
    if empty:
        raise ValueError("MISATO output contains empty artifacts: " + ", ".join(empty))
    with np.load(result_root / "trajectory.npz", allow_pickle=False) as archive:
        positions = archive["positions"]
        if (
            positions.ndim != 3
            or positions.shape[2] != 3
            or positions.shape[0] != profile.frames
            or positions.dtype != np.float32
            or not np.isfinite(positions).all()
        ):
            raise ValueError("MISATO positions violate the profile frame/shape/dtype/finite-value contract")
        frame_count, atom_count = int(positions.shape[0]), int(positions.shape[1])
    return {
        "pdb_path": str(result_root / "trajectory.pdb"),
        "xtc_path": str(result_root / "trajectory.xtc"),
        "metadata_path": str(result_root / "metadata.json"),
        "frame_count": frame_count,
        "atom_count": atom_count,
    }


def execute_request(validation_token: str, gpu_device: int) -> dict[str, Any]:
    settings = ProteinMDSettings.from_env()
    execution = ExecuteRequest.model_validate({"validation_token": validation_token, "gpu_device": gpu_device})
    allocated = _allocated_gpu()
    if execution.gpu_device != allocated:
        raise PermissionError(f"gpu_device={execution.gpu_device} does not match worker allocation gpu:{allocated}")
    token_record = _load_token(
        execution.validation_token,
        settings,
        allow_consumed_expired=True,
    )
    binding_deadline = time.monotonic() + 10
    while token_record.get("submission_state") == "submitting" and time.monotonic() < binding_deadline:
        time.sleep(0.1)
        token_record = _load_token(
            execution.validation_token,
            settings,
            allow_consumed_expired=True,
        )
    if not token_record.get("consumed"):
        raise PermissionError("ProteinMD validation token was not submitted")
    if token_record.get("submission_state") != "submitted":
        raise PermissionError("ProteinMD validation token submission is not complete")
    if token_record.get("child_task_id") != os.getenv("BIOMNI_TASK_ID"):
        raise PermissionError("ProteinMD token is not bound to this execution task")
    if token_record.get("gpu_device") != execution.gpu_device:
        raise PermissionError("ProteinMD token GPU does not match the execution task")
    request = ProteinMDRequest.model_validate(token_record["request"])
    profile = PROFILES[request.execution_profile]
    current_manifest = _input_manifest(request, settings)
    if current_manifest != token_record["input_manifest"]:
        raise ValueError("ProteinMD input files changed after validation")
    attempt, work, output, temporary = _attempt_directories()
    manifest_path = attempt / "input_manifest.json"
    _atomic_json(manifest_path, current_manifest)
    log_path = attempt / "proteinmd.log"
    command = [
        str((settings.proteinmd_root / "scripts/infer_local_case.sh").resolve(strict=True)),
        request.dataset,
        request.system_id,
    ]
    environment = _runtime_environment(settings, request, profile, allocated, work, output, temporary)
    launch_command = _sandboxed_command(settings, command, allocated, work, output, temporary)
    cancel_files = (
        attempt / "CANCEL_REQUESTED",
        Path(os.getenv("BIOMNI_TASK_DIR", str(attempt))) / "CANCEL_REQUESTED",
    )
    if any(path.exists() for path in cancel_files):
        raise ProteinMDExecutionCancelledError("ProteinMD execution was cancelled before process start")
    execution_info = _run_process(
        launch_command,
        settings.proteinmd_root.resolve(strict=True),
        environment,
        log_path,
        profile.timeout_seconds,
        cancel_files,
        settings.termination_grace_seconds,
    )
    if execution_info["termination_reason"]:
        reason = str(execution_info["termination_reason"])
        message = f"ProteinMD process tree terminated: {reason}; details={execution_info['termination']}"
        if reason == "timeout":
            raise TimeoutError(message)
        if reason == "cancelled":
            raise ProteinMDExecutionCancelledError(message)
        raise RuntimeError(message)
    if execution_info["exit_code"] != 0:
        raise RuntimeError(f"ProteinMD exited with code {execution_info['exit_code']}; log={log_path}")
    if _directory_size(output) > settings.max_output_bytes:
        raise ValueError("ProteinMD output exceeds the configured size limit")
    result = _validate_outputs(output, request, profile)
    payload = {
        "success": True,
        "status": "complete",
        "dataset": request.dataset,
        "system_id": request.system_id,
        "execution_profile": request.execution_profile,
        "result": result,
        "execution": {
            **execution_info,
            "gpu_device": allocated,
            "logical_device": "cuda:0",
            "log_path": str(log_path),
            "command_argv": command,
            "sandbox_mode": settings.sandbox_mode,
        },
    }
    _atomic_json(attempt / "result_manifest.json", payload)
    return payload
