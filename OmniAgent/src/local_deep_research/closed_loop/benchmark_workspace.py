from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .workspace import LeakageError, StagedTask


class DrugDiscoveryBenchWorkspaceStager:
    """Expose only a selected public prompt, never evaluator-only task fields."""

    def stage(
        self,
        benchmark_root: str | Path,
        task_id: str,
        run_root: str | Path,
        *,
        run_id: str | None = None,
    ) -> StagedTask:
        source = Path(benchmark_root).expanduser().resolve(strict=True)
        tasks_path = source if source.is_file() else source / "tasks.jsonl"
        if not tasks_path.is_file():
            raise LeakageError(
                f"DrugDiscoveryBench tasks.jsonl is missing: {tasks_path}"
            )

        selected: dict[str, Any] | None = None
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LeakageError(
                        f"Invalid DrugDiscoveryBench JSONL line {line_number}"
                    ) from exc
                item_id = str(item.get("task_id", item.get("id", "")))
                if item_id == str(task_id):
                    selected = item
                    break
        if selected is None:
            raise LeakageError(f"DrugDiscoveryBench task is missing: {task_id}")

        prompt = str(
            selected.get("prompt", selected.get("instruction", ""))
        ).strip()
        if not prompt:
            raise LeakageError(f"DrugDiscoveryBench task has no public prompt: {task_id}")

        root = Path(run_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        safe_run_id = self._safe_name(
            run_id or f"drugdiscovery-{task_id}-{uuid4().hex[:8]}"
        )
        workspace = root / safe_run_id
        workspace.mkdir(mode=0o700)
        output_path = workspace / "final_submission.json"
        manifest = {
            "id": str(task_id),
            "name": str(selected.get("short_title", task_id)),
            "description": prompt,
            "input_files": [],
            "output_config": {
                "file_path": output_path.name,
                "format": "json",
            },
            "task_parameters": {
                "benchmark": "DrugDiscoveryBench",
                "benchmark_release": "tasks.jsonl",
                "evaluation_visibility": "evaluator_only",
            },
            "isolation": {
                "source_path_exposed": False,
                "public_prompt_only": True,
                "ground_truth_excluded": True,
                "rubrics_excluded": True,
                "canary_excluded": True,
            },
        }
        manifest_path = workspace / "task_manifest.json"
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
        forbidden_values = (
            str(selected.get("ground_truth", "")),
            str(selected.get("canary", "")),
        )
        for value in forbidden_values:
            if value and value in serialized:
                raise LeakageError("Evaluator-only DrugDiscoveryBench data leaked")
        manifest_path.write_text(serialized, encoding="utf-8")
        actual = {
            path.resolve()
            for path in workspace.rglob("*")
            if path.is_file()
        }
        if actual != {manifest_path.resolve()}:
            raise LeakageError("Unexpected file entered the benchmark workspace")
        return StagedTask(
            task_id=str(task_id),
            workspace=workspace,
            manifest_path=manifest_path,
            output_path=output_path,
            input_paths=(),
            manifest=manifest,
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip(".-")
        if not safe:
            raise LeakageError("run_id does not contain a safe filename character")
        return safe
