from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


class LeakageError(ValueError):
    """Raised when a benchmark asset could escape the isolated workspace."""


@dataclass(frozen=True, slots=True)
class StagedTask:
    task_id: str
    workspace: Path
    manifest_path: Path
    output_path: Path
    input_paths: tuple[Path, ...]
    manifest: dict[str, Any]

    @property
    def allowed_paths(self) -> list[str]:
        return [str(self.workspace.resolve())]


class SmddWorkspaceStager:
    """Copy only declared SMDD inputs into an agent-visible run directory."""

    _SENSITIVE_KEY_MARKERS = (
        "evaluation",
        "ground_truth",
        "witness",
        "hidden",
        "answer",
    )
    _BLOCKED_FILE_MARKERS = (
        "eval_",
        "ground_truth",
        "witness",
        "gt_dock",
    )

    def stage(
        self,
        task_directory: str | Path,
        run_root: str | Path,
        *,
        run_id: str | None = None,
    ) -> StagedTask:
        source = Path(task_directory).resolve(strict=True)
        if not source.is_dir():
            raise LeakageError(f"Task directory is not a directory: {source}")

        config_path = source / "task.yaml"
        if not config_path.is_file():
            raise LeakageError(f"Missing task.yaml: {config_path}")
        config = self._read_yaml(config_path)

        root = Path(run_root).resolve()
        if self._is_relative_to(root, source):
            raise LeakageError("run_root must not be inside the source benchmark task")
        root.mkdir(parents=True, exist_ok=True)

        safe_run_id = self._safe_name(run_id or f"run-{uuid4().hex[:12]}")
        workspace = root / safe_run_id
        workspace.mkdir(mode=0o700)

        input_names = config.get("input_files", [])
        if not isinstance(input_names, list) or not input_names:
            raise LeakageError("task.yaml must declare at least one input file")

        copied_inputs: list[Path] = []
        hashes: dict[str, str] = {}
        for raw_name in input_names:
            source_file, relative = self._resolve_declared_file(source, raw_name)
            self._reject_hidden_name(relative)
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            copied_inputs.append(destination)
            hashes[relative.as_posix()] = self._sha256(destination)

        output_config = config.get("output_config", {})
        if not isinstance(output_config, dict):
            raise LeakageError("output_config must be an object")
        output_name = output_config.get("file_path")
        if not isinstance(output_name, str) or not output_name.strip():
            raise LeakageError("output_config.file_path must be a non-empty string")
        output_relative = self._validate_relative_path(output_name)
        output_path = (workspace / output_relative).resolve()
        if not self._is_relative_to(output_path, workspace.resolve()):
            raise LeakageError("Output path escapes the isolated workspace")
        if output_path in {path.resolve() for path in copied_inputs}:
            raise LeakageError("Output path cannot overwrite an input file")

        manifest = {
            "id": str(config.get("id") or source.name),
            "name": str(config.get("name") or source.name),
            "description": str(config.get("description") or ""),
            "input_files": [path.relative_to(workspace).as_posix() for path in copied_inputs],
            "output_config": {
                "file_path": output_relative.as_posix(),
                "format": output_config.get("format"),
            },
            "task_parameters": self._sanitize_mapping(
                {
                    key: value
                    for key, value in config.items()
                    if key
                    not in {
                        "id",
                        "name",
                        "description",
                        "input_files",
                        "output_config",
                        "evaluation",
                    }
                }
            ),
            "input_sha256": hashes,
            "isolation": {
                "source_path_exposed": False,
                "only_declared_inputs_copied": True,
                "private_assets_excluded": True,
            },
        }
        manifest_path = workspace / "task_manifest.json"
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
        if str(source) in serialized:
            raise LeakageError("Source benchmark path leaked into the agent manifest")
        self._reject_hidden_reference(serialized)
        manifest_path.write_text(serialized, encoding="utf-8")

        staged = StagedTask(
            task_id=manifest["id"],
            workspace=workspace,
            manifest_path=manifest_path,
            output_path=output_path,
            input_paths=tuple(copied_inputs),
            manifest=manifest,
        )
        self.assert_isolated(staged)
        return staged

    def assert_isolated(self, staged: StagedTask) -> None:
        allowed = {staged.manifest_path.resolve(), *(path.resolve() for path in staged.input_paths)}
        if staged.output_path.exists():
            allowed.add(staged.output_path.resolve())
        actual = {path.resolve() for path in staged.workspace.rglob("*") if path.is_file()}
        extras = actual - allowed
        if extras:
            raise LeakageError(f"Unexpected files in isolated workspace: {sorted(map(str, extras))}")
        for path in actual:
            self._reject_hidden_name(path.relative_to(staged.workspace))

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise LeakageError("task.yaml must contain a YAML object")
        return value

    def _resolve_declared_file(self, source: Path, raw_name: Any) -> tuple[Path, Path]:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise LeakageError("Every input file must be a non-empty relative path")
        relative = self._validate_relative_path(raw_name)
        unresolved = source / relative
        current = source
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise LeakageError(f"Symlink inputs are not allowed: {relative}")
        resolved = unresolved.resolve(strict=True)
        if not self._is_relative_to(resolved, source):
            raise LeakageError(f"Input path escapes task directory: {relative}")
        if not resolved.is_file():
            raise LeakageError(f"Declared input is not a file: {relative}")
        return resolved, relative

    @staticmethod
    def _validate_relative_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise LeakageError(f"Unsafe relative path: {raw_path}")
        return path

    def _reject_hidden_name(self, path: Path) -> None:
        lowered = path.as_posix().lower()
        if any(marker in lowered for marker in self._BLOCKED_FILE_MARKERS):
            raise LeakageError(f"Hidden evaluation asset cannot be agent-visible: {path}")

    def _reject_hidden_reference(self, serialized_manifest: str) -> None:
        lowered = serialized_manifest.lower()
        leaked = [
            marker for marker in self._BLOCKED_FILE_MARKERS if marker in lowered
        ]
        if leaked:
            raise LeakageError(
                "Hidden evaluation reference leaked into the agent manifest: "
                + ", ".join(leaked)
            )

    def _sanitize_mapping(self, value: Any) -> Any:
        if isinstance(value, dict):
            clean: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in self._SENSITIVE_KEY_MARKERS):
                    continue
                clean[str(key)] = self._sanitize_mapping(item)
            return clean
        if isinstance(value, list):
            return [self._sanitize_mapping(item) for item in value]
        return value

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip(".-")
        if not safe:
            raise LeakageError("run_id does not contain a safe filename character")
        return safe

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True
