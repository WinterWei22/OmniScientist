from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluation:
    benchmark_status: str
    benchmark_score: float | None
    evaluator: str
    reason: str
    task_id: str
    harness_status: str
    harness_score: float
    schema_version: str = "omniagent.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_ddb_after_run(
    *,
    task_id: str,
    workspace: Path,
    benchmark_root: Path,
    answer_path: Path,
    trajectory_path: Path,
    harness_status: str,
    harness_score: float,
    dataset_path: Path | None = None,
) -> BenchmarkEvaluation:
    """Run the hidden-reference judge only after the agent runtime has stopped."""
    bundled_rubrics_path = (
        benchmark_root
        / "benchmark"
        / "tasks"
        / task_id
        / "tests"
        / "rubrics.json"
    )
    judge_path = bundled_rubrics_path.with_name("judge.py")
    result_path = workspace / "benchmark_evaluation.json"

    bundle = _load_task_bundle(dataset_path, task_id)
    if bundle is None and bundled_rubrics_path.is_file():
        bundle = _load_json(bundled_rubrics_path)
    if bundle is None:
        return _persist(
            result_path,
            BenchmarkEvaluation(
                benchmark_status="not_evaluated",
                benchmark_score=None,
                evaluator="drugdiscoverybench.official_judge",
                reason="Official DrugDiscoveryBench rubric bundle is unavailable.",
                task_id=task_id,
                harness_status=harness_status,
                harness_score=harness_score,
            ),
        )

    if not isinstance(bundle, dict):
        return _persist(
            result_path,
            _not_evaluated(
                task_id,
                harness_status,
                harness_score,
                "Official DrugDiscoveryBench rubric bundle is not a JSON object.",
            ),
        )
    if not bundle.get("ground_truth") or not bundle.get("outcome_rubrics"):
        return _persist(
            result_path,
            _not_evaluated(
                task_id,
                harness_status,
                harness_score,
                "Official outcome ground truth or rubrics are not populated.",
            ),
        )
    if not judge_path.is_file():
        return _persist(
            result_path,
            _not_evaluated(
                task_id,
                harness_status,
                harness_score,
                "Rubric loaded, but the official DrugDiscoveryBench judge is unavailable.",
            ),
        )
    judge_base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("BAILIAN_BASE_URL")
    judge_api_key = os.getenv("JUDGE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not judge_base_url:
        judge_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not judge_api_key:
        return _persist(
            result_path,
            _not_evaluated(
                task_id,
                harness_status,
                harness_score,
                "Official judge credentials are not configured for post-run evaluation.",
            ),
        )

    evaluation_dir = workspace.parent / ".benchmark_evaluations" / workspace.name
    evaluation_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rubrics_path = evaluation_dir / f"rubrics-{task_id}.json"
    rubrics_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reward_path = evaluation_dir / "reward.json"
    command = [
        sys.executable,
        str(judge_path),
        "--answer-file",
        str(answer_path),
        "--rubrics-file",
        str(rubrics_path),
        "--trajectory-file",
        str(trajectory_path),
        "--output",
        str(reward_path),
    ]
    judge_env = os.environ.copy()
    judge_env.update(
        {
            "JUDGE_MODEL": os.getenv("JUDGE_MODEL") or os.getenv("BAILIAN_MODEL") or "qwen3.8-max",
            "JUDGE_BASE_URL": judge_base_url,
            "JUDGE_API_KEY": judge_api_key,
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            env=judge_env,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0 or not reward_path.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(detail[-2000:] or "official judge produced no reward")
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        score = float(reward["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("official judge score is outside [0, 1]")
        threshold = float(os.getenv("DDB_BENCHMARK_PASS_SCORE", "1.0"))
        evaluation = BenchmarkEvaluation(
            benchmark_status="passed" if score >= threshold else "failed",
            benchmark_score=score,
            evaluator="drugdiscoverybench.official_judge",
            reason="Official post-run outcome evaluation completed.",
            task_id=task_id,
            harness_status=harness_status,
            harness_score=harness_score,
        )
    except Exception as exc:
        evaluation = BenchmarkEvaluation(
            benchmark_status="evaluation_error",
            benchmark_score=None,
            evaluator="drugdiscoverybench.official_judge",
            reason=f"Official post-run evaluation failed: {type(exc).__name__}: {exc}",
            task_id=task_id,
            harness_status=harness_status,
            harness_score=harness_score,
        )
    return _persist(result_path, evaluation)


def _not_evaluated(
    task_id: str,
    harness_status: str,
    harness_score: float,
    reason: str,
) -> BenchmarkEvaluation:
    return BenchmarkEvaluation(
        benchmark_status="not_evaluated",
        benchmark_score=None,
        evaluator="drugdiscoverybench.official_judge",
        reason=reason,
        task_id=task_id,
        harness_status=harness_status,
        harness_score=harness_score,
    )


def _load_task_bundle(dataset_path: Path | None, task_id: str) -> dict[str, Any] | None:
    if dataset_path is None or not dataset_path.is_file():
        return None
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if str(item.get("task_id", item.get("id", ""))) == task_id:
                    return item
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _persist(path: Path, evaluation: BenchmarkEvaluation) -> BenchmarkEvaluation:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return evaluation
