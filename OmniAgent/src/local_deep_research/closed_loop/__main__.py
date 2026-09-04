from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

from .a1_tool import BiomniA1Tool
from .contracts import LoopPolicy
from .roles import LLMAnalyzer, LLMPlanner, LLMVerifier
from .runtime import ClosedLoopRuntime
from .workspace import SmddWorkspaceStager


def _build_model(name: str):
    from local_deep_research.config import get_qwen_siliconflow

    if name != "qwen":
        raise ValueError("The competition closed loop is Qwen-only")
    return get_qwen_siliconflow()


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


async def _run(args: argparse.Namespace) -> int:
    staged = SmddWorkspaceStager().stage(args.task_dir, args.run_root)
    model = _build_model(args.model)
    a1_tool = BiomniA1Tool.from_settings()
    await a1_tool.initialize()

    events_path = staged.workspace / "events.jsonl"

    def record_event(event: str, payload: dict) -> None:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": event, "payload": payload},
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )

    runtime = ClosedLoopRuntime(
        planner=LLMPlanner(model),
        a1_tool=a1_tool,
        analyzer=LLMAnalyzer(model),
        verifier=LLMVerifier(model),
        policy=LoopPolicy(
            max_iterations=args.max_iterations,
            max_a1_calls=args.max_a1_calls,
            target_score=args.target_score,
            min_improvement=args.min_improvement,
            max_stalled_iterations=args.max_stalled_iterations,
            max_steps_per_iteration=args.max_steps_per_iteration,
        ),
        event_sink=record_event,
    )
    state = await runtime.run(
        goal=staged.manifest["description"],
        constraints=[
            "Use only files in the isolated workspace",
            "Do not inspect benchmark source directories",
        ],
        workspace=str(staged.workspace),
        task_manifest=staged.manifest,
        allowed_paths=staged.allowed_paths,
    )
    final_path = staged.workspace / "final_state.json"
    final_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run status: {state.status.value}")
    print(f"Finish reason: {state.finish_reason}")
    print(f"Workspace: {staged.workspace}")
    print(f"Final state: {final_path}")
    return 0 if state.status.value == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OmniAgent's closed-loop SMDD workflow"
    )
    parser.add_argument("task_dir", type=Path)
    parser.add_argument(
        "--run-root", type=Path, default=Path("closed_loop_runs")
    )
    parser.add_argument(
        "--model",
        choices=["qwen"],
        default="qwen",
    )
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-a1-calls", type=int, default=12)
    parser.add_argument("--target-score", type=float, default=0.8)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--max-stalled-iterations", type=int, default=2)
    parser.add_argument("--max-steps-per-iteration", type=int, default=1)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
