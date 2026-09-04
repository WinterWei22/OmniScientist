from __future__ import annotations

import fnmatch
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuntimeContextError(ValueError):
    """Raised when the Biomni runtime build context is not competition-safe."""


DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "closed_loop_runs",
        "evaluation",
        "benchmarks",
        "smdd_tasks",
        "output",
        "outputs",
    }
)

DEFAULT_EXCLUDED_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.egg-info/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*eval_*",
    "*ground_truth*",
    "*gt_dock*",
    "*witness*",
)

DEFAULT_DATA_PARTS = frozenset({"data", "data_lake", "datasets", "cache", "models"})


@dataclass(frozen=True, slots=True)
class RuntimeContextManifest:
    source_name: str
    context_path: str
    files_copied: int
    paths_excluded: int
    excluded_parts: tuple[str, ...]
    excluded_patterns: tuple[str, ...]
    data_excluded: bool
    sensitive_assets_excluded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "context_path": self.context_path,
            "files_copied": self.files_copied,
            "paths_excluded": self.paths_excluded,
            "excluded_parts": list(self.excluded_parts),
            "excluded_patterns": list(self.excluded_patterns),
            "data_excluded": self.data_excluded,
            "sensitive_assets_excluded": self.sensitive_assets_excluded,
        }


@dataclass(slots=True)
class BiomniRuntimeContextBuilder:
    """Create a Docker build context that excludes hidden benchmark assets."""

    excluded_parts: frozenset[str] = DEFAULT_EXCLUDED_PARTS
    excluded_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_PATTERNS
    exclude_large_data: bool = True
    data_parts: frozenset[str] = DEFAULT_DATA_PARTS
    manifest_filename: str = "runtime_context_manifest.json"

    def stage(
        self,
        source_root: str | Path,
        context_root: str | Path,
        *,
        clean: bool = False,
    ) -> RuntimeContextManifest:
        source = Path(source_root).resolve(strict=True)
        context = Path(context_root).resolve()
        if not source.is_dir():
            raise RuntimeContextError(f"Biomni source is not a directory: {source}")
        if source == context or self._is_relative_to(context, source):
            raise RuntimeContextError("Build context must not be inside Biomni source")

        if context.exists():
            if not clean:
                raise RuntimeContextError(
                    f"Build context already exists; pass clean=True to replace it: {context}"
                )
            if context.anchor == str(context):
                raise RuntimeContextError(f"Refuse to delete filesystem root: {context}")
            shutil.rmtree(context)
        context.mkdir(parents=True)

        files_copied = 0
        paths_excluded = 0
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise RuntimeContextError(f"Symlink paths are not allowed: {relative}")
            if self._should_exclude(relative, path.is_dir()):
                paths_excluded += 1
                continue
            if path.is_dir():
                (context / relative).mkdir(parents=True, exist_ok=True)
                continue
            destination = context / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files_copied += 1

        self.assert_context_is_safe(context)
        manifest = RuntimeContextManifest(
            source_name=source.name,
            context_path=str(context),
            files_copied=files_copied,
            paths_excluded=paths_excluded,
            excluded_parts=tuple(sorted(self.excluded_parts)),
            excluded_patterns=tuple(self.excluded_patterns),
            data_excluded=self.exclude_large_data,
        )
        (context / self.manifest_filename).write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def assert_context_is_safe(self, context_root: str | Path) -> None:
        context = Path(context_root).resolve(strict=True)
        for path in context.rglob("*"):
            relative = path.relative_to(context)
            if self._is_sensitive_path(relative):
                raise RuntimeContextError(
                    f"Sensitive benchmark asset entered runtime context: {relative}"
                )

    def _should_exclude(self, relative: Path, is_dir: bool) -> bool:
        parts = {part.lower() for part in relative.parts}
        if parts & {part.lower() for part in self.excluded_parts}:
            return True
        if self.exclude_large_data and parts & {part.lower() for part in self.data_parts}:
            return True
        normalized = relative.as_posix().lower()
        if is_dir:
            normalized = normalized.rstrip("/") + "/"
        return any(fnmatch.fnmatch(normalized, pattern.lower()) for pattern in self.excluded_patterns)

    def _is_sensitive_path(self, relative: Path) -> bool:
        lowered = relative.as_posix().lower()
        sensitive_parts = {"evaluation", "benchmarks", "smdd_tasks", "hidden"}
        if {part.lower() for part in relative.parts} & sensitive_parts:
            return True
        return any(
            marker in lowered
            for marker in ("eval_", "ground_truth", "gt_dock", "witness")
        )

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

