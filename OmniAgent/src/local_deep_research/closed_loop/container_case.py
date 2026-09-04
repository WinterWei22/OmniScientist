from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from .a1_tool import BiomniA1Tool
from .biomni_runtime_context import BiomniRuntimeContextBuilder
from .container_runtime import A1ContainerRuntime, A1ContainerSpec, A1ImageBuildSpec
from .contracts import LoopPolicy
from .roles import LLMAnalyzer, LLMPlanner, LLMVerifier
from .runtime import ClosedLoopRuntime
from .workspace import SmddWorkspaceStager


def _build_model(name: str):
    from local_deep_research.config import (
        get_deepseek_r1,
        get_gpt4_1,
        get_gpt4_1_mini,
        get_qwen_siliconflow,
    )

    factories = {
        "qwen": get_qwen_siliconflow,
        "gpt4_1": get_gpt4_1,
        "gpt4_1_mini": get_gpt4_1_mini,
        "deepseek": get_deepseek_r1,
    }
    return factories[name]()


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _split_container_command(command: str | None) -> tuple[str, ...]:
    if not command:
        return ()
    return tuple(shlex.split(command, posix=True))


async def _run_closed_loop(args: argparse.Namespace, biomni_url: str, staged) -> int:
    model = _build_model(args.model)
    a1_tool = BiomniA1Tool({"transport": "sse", "url": biomni_url})
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
        ),
        event_sink=record_event,
    )
    state = await runtime.run(
        goal=staged.manifest["description"],
        constraints=[
            "Use only files in the isolated workspace.",
            "Do not inspect benchmark source directories.",
            "Use Biomni A1 as the only environment-facing tool.",
        ],
        workspace=str(staged.workspace),
        task_manifest=staged.manifest,
        allowed_paths=staged.allowed_paths,
    )
    final_path = staged.workspace / "final_state.json"
    final_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Run status: {state.status.value}")
    print(f"Finish reason: {state.finish_reason}")
    print(f"Workspace: {staged.workspace}")
    print(f"Events: {events_path}")
    print(f"Final state: {final_path}")
    return 0 if state.status.value == "completed" else 1


def _write_phase1_plan(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"phase1-{uuid4().hex[:10]}"
    staged = SmddWorkspaceStager().stage(args.task_dir, args.run_root, run_id=run_id)

    context_dir = args.context_dir or (
        staged.workspace.parent / "_a1_runtime_contexts" / run_id
    )
    dockerfile = args.dockerfile or (
        Path(__file__).resolve().parents[3] / "deploy" / "a1_runtime" / "Dockerfile"
    )
    context_manifest = None
    if not args.skip_context:
        context_manifest = BiomniRuntimeContextBuilder(
            exclude_large_data=not args.include_data_in_image
        ).stage(args.biomni_source, context_dir, clean=args.clean_context)

    build_spec = A1ImageBuildSpec(
        context_dir=context_dir,
        dockerfile=dockerfile,
        image_tag=args.image_tag,
        runtime=args.container_runtime,
        pull=args.pull,
    )
    container_spec = A1ContainerSpec(
        workspace=staged.workspace,
        image_tag=args.image_tag,
        container_name=args.container_name or f"omniagent-biomni-a1-{run_id}",
        host_port=args.host_port,
        container_port=args.container_port,
        runtime=args.container_runtime,
        command=_split_container_command(args.a1_command),
        public_data=args.public_data,
        cpus=args.cpus,
        memory=args.memory,
        readonly_rootfs=not args.disable_readonly_rootfs,
        extra_env={
            "BIOMNI_MCP_TRANSPORT": "sse",
            "BIOMNI_MCP_HOST": "0.0.0.0",
            "BIOMNI_MCP_PORT": str(args.container_port),
        },
    )
    phase1_plan = {
        "run_id": run_id,
        "task_manifest": str(staged.manifest_path),
        "isolated_workspace": str(staged.workspace),
        "context_manifest": context_manifest.to_dict() if context_manifest else None,
        "build_command": build_spec.command().to_dict(),
        "run_command": container_spec.run_command(detach=True).to_dict(),
        "biomni_url": container_spec.endpoint_url,
        "dry_run": args.dry_run,
    }
    plan_path = staged.workspace / "phase1_container_plan.json"
    _write_phase1_plan(plan_path, phase1_plan)

    if args.dry_run:
        print(f"Dry run plan: {plan_path}")
        print("Build command:")
        print(build_spec.command().to_powershell())
        print("Run command:")
        print(container_spec.run_command(detach=True).to_powershell())
        return 0

    runtime = A1ContainerRuntime()
    if not args.skip_build:
        runtime.run_checked(build_spec.command())
    runtime.start_detached(container_spec)
    try:
        runtime.wait_for_port("127.0.0.1", args.host_port, timeout_seconds=args.wait_seconds)
        return await _run_closed_loop(args, container_spec.endpoint_url, staged)
    finally:
        if not args.keep_container:
            runtime.stop(container_spec)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a phase-1 isolated OmniAgent -> Biomni A1 SMDD case."
    )
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--biomni-source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("closed_loop_runs"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--context-dir", type=Path)
    parser.add_argument("--clean-context", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--include-data-in-image", action="store_true")
    parser.add_argument("--dockerfile", type=Path)
    parser.add_argument("--container-runtime", default=os.getenv("CONTAINER_RUNTIME", "docker"))
    parser.add_argument("--image-tag", default="omniagent-biomni-a1-runtime:phase1")
    parser.add_argument("--container-name", default="")
    parser.add_argument("--host-port", type=int, default=18100)
    parser.add_argument("--container-port", type=int, default=18000)
    parser.add_argument("--public-data", type=Path)
    parser.add_argument("--a1-command", default="")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--disable-readonly-rootfs", action="store_true")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--memory", default="16g")
    parser.add_argument("--wait-seconds", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        choices=["qwen", "gpt4_1", "gpt4_1_mini", "deepseek"],
        default="qwen",
    )
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-a1-calls", type=int, default=12)
    parser.add_argument("--target-score", type=float, default=0.8)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--max-stalled-iterations", type=int, default=2)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

