from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from time import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .action_ledger import ActionRecord, ActionStatus, canonical_action_key
from .contracts import (
    A1TaskRequest,
    A1TaskResult,
    Decision,
    ExperimentPlan,
    ExperimentStep,
    IterationRecord,
    LoopPhase,
    LoopPolicy,
    PendingExecution,
    PlannerContractError,
    ResearchState,
    RunStatus,
)
from .finalization import FinalizationDecision, FinalizationGate
from .answer_validation import (
    infer_answer_semantic_contract,
    synthesize_grounded_final_output,
    validate_final_answer_semantics,
)
from .persistence import RunPersistence
from .result_payload import (
    material_result_leaves,
    merge_envelope_answer,
)
from .result_verifier import TaskResultVerifier, build_task_result_verifier
from .scientific_state import ScientificStateReducer
from .artifact_contract import verify_artifacts
from .execution_validation import validate_schema_instance
from .failure_policy import execution_failure_retryable
from .biomni_task_protocol import normalize_task_status
from .entity_resolution import EntityResolutionError, ProteinEntityResolver
from .execution_models import (
    BoundCapabilityCall,
    BoundCapabilityWorkflow,
    ConditionCoverage,
    EffectContract,
    ExecutionBackend,
    PathValueRequirement,
    ResourceCandidate,
    RouteDecision,
    SemanticCapabilityIntent,
    WorkflowCallTemplate,
)
from .state_graph import ExecutionStateGraph
from .task_metadata import external_task_summary

if TYPE_CHECKING:
    from .evaluators import WorkflowEvaluator
    from .roles import Analyzer, Planner, Verifier


class FeedbackNotConsumedError(ValueError):
    """Raised when a revised plan ignores mandatory Verifier feedback."""


class FinalArtifactPlanError(ValueError):
    """Raised when a tool plan tries to handle Harness-owned final output."""


EventSink = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class _IterationContext:
    state: ResearchState
    evaluator: WorkflowEvaluator
    result_verifier: TaskResultVerifier
    visible_paths: list[str]
    plan: ExperimentPlan | None = None
    executions: list[A1TaskResult] = field(default_factory=list)
    cursor: int = 0
    request: A1TaskRequest | None = None
    step: ExperimentStep | None = None
    attempt_key: str = ""
    binding: Any = None
    binding_failed: bool = False
    skip_dispatch: bool = False
    pending: bool = False
    result: A1TaskResult | None = None
    artifact_path: str = ""
    verification: Any = None
    analysis: Any = None
    evaluation: Any = None
    critique: Any = None
    finalization: Any = None
    action: ActionRecord | None = None
    action_disposition: str = ""
    terminal: bool = False


class ClosedLoopRuntime:
    """Runtime-owned PLAN -> BIND -> DISPATCH -> VERIFY -> FINALIZE graph."""

    _NON_TERMINAL_EXTERNAL_STATUSES = frozenset(
        {
            "submitted",
            "queued",
            "running",
            "retry_wait",
            "pending",
            "in_progress",
            "waiting",
        }
    )
    _TERMINAL_EXTERNAL_STATUS_ALIASES = {
        "success": "succeeded",
        "completed": "succeeded",
        "complete": "succeeded",
        "timeout": "timed_out",
        "timedout": "timed_out",
        "canceled": "cancelled",
        "retrying": "retry_wait",
        "retrying_wait": "retry_wait",
        "manual": "manual_review",
        "review": "manual_review",
        "dead-letter": "dead_letter",
        "deadletter": "dead_letter",
    }

    def __init__(
        self,
        *,
        planner: Planner,
        a1_tool: Any,
        analyzer: Analyzer,
        verifier: Verifier,
        evaluator: WorkflowEvaluator | None = None,
        result_verifier: TaskResultVerifier | None = None,
        finalization_gate: FinalizationGate | None = None,
        policy: LoopPolicy | None = None,
        event_sink: EventSink | None = None,
        persistence_root: str | Path | None = None,
        entity_resolver: ProteinEntityResolver | None = None,
    ) -> None:
        self.planner = planner
        self.a1_tool = a1_tool
        self.analyzer = analyzer
        self.verifier = verifier
        self.evaluator = evaluator
        self.result_verifier = result_verifier
        self.finalization_gate = finalization_gate or FinalizationGate()
        self.state_reducer = ScientificStateReducer()
        self.policy = policy or LoopPolicy()
        self.event_sink = event_sink or (lambda _event, _payload: None)
        self.entity_resolver = entity_resolver or ProteinEntityResolver()
        self.persistence_root = (
            Path(persistence_root).expanduser().resolve()
            if persistence_root is not None
            else None
        )
        self.persistence: RunPersistence | None = None

    async def run(
        self,
        *,
        goal: str,
        constraints: list[str],
        workspace: str,
        task_manifest: dict[str, Any],
        allowed_paths: list[str] | None = None,
        resume_from: str | Path | None = None,
    ) -> ResearchState:
        """Run the explicit PLAN-to-FINALIZE graph for one research task."""
        if resume_from is not None:
            self.persistence = RunPersistence.from_checkpoint(resume_from)
            state = self.persistence.load_checkpoint(resume_from)
            self._restore_pending_from_action_ledger(state)
            self.persistence.validate_resume_request(
                state,
                goal=goal,
                workspace=workspace,
                task_manifest=task_manifest,
            )
            replayable_submission = any(
                action.status in {"admitted", "submitted"}
                or action.status == "unknown"
                and bool(action.task_metadata.get("submission_unknown"))
                for action in state.action_ledger.records.values()
            )
            if state.status is RunStatus.NEEDS_REVIEW and (
                state.pending_execution is not None or replayable_submission
            ):
                # A local wait budget is a pause for reconciliation, not a
                # terminal scientific decision. Resume the same external task.
                state.status = RunStatus.RUNNING
                state.phase = (
                    LoopPhase.WAIT_EXTERNAL
                    if state.pending_execution is not None
                    else LoopPhase.BIND
                    if state.active_plan is not None
                    else LoopPhase.PLAN
                )
                state.finish_reason = ""
            if state.pending_execution is not None:
                pending = state.pending_execution
                if pending.rpc_status == "recovered" and not pending.deadline_at:
                    pending.deadline_at = time() + self._task_timeout_seconds()
                    pending.next_poll_at = time()
                if (
                    pending.rpc_status == "deadline_exceeded"
                    and pending.deadline_at
                    and pending.deadline_at <= time()
                ):
                    pending.deadline_at = time() + self._task_timeout_seconds()
                    pending.rpc_status = "reconciling"
                    pending.unknown_reason = ""
                    pending.last_poll_error = ""
                    pending.next_poll_at = time()
                    pending.task_metadata.update(
                        {
                            "deadline_at": pending.deadline_at,
                            "next_poll_at": pending.next_poll_at,
                            "rpc_status": pending.rpc_status,
                        }
                    )
                    self._save_pending_snapshot(state)
            self._emit(
                "run_resumed",
                {
                    "run_id": state.run_id,
                    "phase": state.phase,
                    "iteration": len(state.iterations),
                    "state_version": state.scientific_state.state_version,
                    "active_plan_id": (
                        state.active_plan.plan_id if state.active_plan else None
                    ),
                    "active_execution_count": len(state.active_executions),
                },
            )
        else:
            state = ResearchState(
                goal=goal,
                constraints=list(constraints),
                workspace=workspace,
                task_manifest=task_manifest,
                working_memory_token_budget=self.policy.working_memory_token_budget,
            )
            root = self.persistence_root or (
                Path(workspace).resolve() / ".omniagent" / "runs" / state.run_id
            )
            self.persistence = RunPersistence(root, run_id=state.run_id)

        visible_paths = list(allowed_paths or [workspace])
        evaluator = self.evaluator
        if evaluator is None:
            from .evaluators import BenchmarkWorkflowEvaluator

            evaluator = BenchmarkWorkflowEvaluator()
        result_verifier = self.result_verifier or build_task_result_verifier(
            task_manifest
        )
        graph = ExecutionStateGraph(self)
        if resume_from is None:
            self._emit("run_started", state.to_dict())
            self._save_checkpoint(state, "initialized")

        while state.status is RunStatus.RUNNING:
            if state.active_plan is None and len(state.iterations) >= self.policy.max_iterations:
                self._finish_at_limit(state, "max_iterations_reached")
                break
            if state.active_plan is None and state.a1_call_count >= self.policy.max_a1_calls:
                self._finish_at_limit(state, "a1_call_budget_exhausted")
                break

            context = _IterationContext(
                state=state,
                evaluator=evaluator,
                result_verifier=result_verifier,
                visible_paths=visible_paths,
                plan=state.active_plan,
                executions=list(state.active_executions),
                cursor=len(state.active_executions),
            )
            self._emit(
                "state_graph_started",
                {
                    "run_id": state.run_id,
                    "iteration": len(state.iterations),
                    "entry_phase": (
                        LoopPhase.WAIT_EXTERNAL.value
                        if state.pending_execution is not None
                        else LoopPhase.PLAN.value
                        if state.active_plan is None
                        else LoopPhase.BIND.value
                    ),
                },
            )
            await graph.run(context)

        self._emit("run_finished", self._run_finished_summary(state))
        self._save_checkpoint(state, "run_finished")
        return state

    def _graph_enter(self, context: _IterationContext, node: str, phase: LoopPhase) -> None:
        context.state.phase = phase
        self._emit(
            "state_graph_node_entered",
            {
                "node": node,
                "phase": phase.value,
                "iteration": len(context.state.iterations),
                "step_id": context.step.step_id if context.step else None,
            },
        )

    def _graph_exit(self, context: _IterationContext, node: str) -> None:
        self._emit(
            "state_graph_node_exited",
            {
                "node": node,
                "phase": context.state.phase.value,
                "iteration": len(context.state.iterations),
                "step_id": context.step.step_id if context.step else None,
                "state_version": context.state.scientific_state.state_version,
                "status": context.state.status.value,
            },
        )

    async def _graph_plan(self, context: _IterationContext) -> None:
        state = context.state
        self._graph_enter(context, "plan", LoopPhase.PLAN)
        if state.active_plan is None:
            planner_context = state.planner_context()
            try:
                plan = await self._plan_with_contract_retry(state)
            except (PlannerContractError, FeedbackNotConsumedError) as exc:
                state.constraints.append(
                    "Planner contract feedback: " + str(exc)
                )
                self._needs_review(state, "planner_contract_violation")
                self._emit(
                    "plan_contract_failed",
                    {
                        "iteration": len(state.iterations),
                        "reason": str(exc),
                    },
                )
                self._save_checkpoint(state, "planner_contract_failed")
                context.terminal = True
                self._graph_exit(context, "plan")
                return
            except FinalArtifactPlanError as exc:
                self._fail(state, "planner_final_artifact_contract_violation")
                self._emit(
                    "run_failed",
                    {
                        "iteration": len(state.iterations),
                        "reason": state.finish_reason,
                        "error": str(exc),
                    },
                )
                self._save_checkpoint(state, "planner_final_artifact_contract_violation")
                context.terminal = True
                self._graph_exit(context, "plan")
                return
            state.active_plan = plan
            state.final_output_materialized = False
            state.active_executions.clear()
            context.executions.clear()
            self._emit(
                "plan_created",
                {
                    "iteration": len(state.iterations),
                    "planner_context": planner_context,
                    "plan": plan,
                },
            )
            self._save_checkpoint(state, "plan_created")
        else:
            plan = state.active_plan
            try:
                self._validate_plan(state, plan)
            except (PlannerContractError, FeedbackNotConsumedError) as exc:
                self._needs_review(state, "planner_contract_violation")
                self._emit(
                    "plan_contract_failed",
                    {
                        "iteration": len(state.iterations),
                        "reason": str(exc),
                    },
                )
                self._save_checkpoint(state, "planner_contract_failed")
                context.terminal = True
                self._graph_exit(context, "plan")
                return
            self._emit(
                "plan_resumed",
                {
                    "iteration": len(state.iterations),
                    "plan_id": plan.plan_id,
                    "completed_execution_count": len(state.active_executions),
                },
            )
        context.plan = state.active_plan
        context.cursor = len(state.active_executions)
        context.executions = list(state.active_executions)
        self._graph_exit(context, "plan")

    def _graph_prepare_request(self, context: _IterationContext) -> None:
        if context.plan is None:
            raise ValueError("state graph has no active plan")
        if context.cursor >= len(context.plan.steps):
            raise ValueError("state graph cursor is beyond the active plan")
        context.step = context.plan.steps[context.cursor]
        state = context.state
        execution_memory = state.working_memory(
            purpose="execute", focus=context.step.objective
        )
        context.request = A1TaskRequest(
            run_id=state.run_id,
            iteration=len(state.iterations),
            research_goal=state.goal,
            hypothesis=context.plan.hypothesis,
            step=context.step,
            global_constraints=state.constraints,
            prior_observations=list(execution_memory.get("evidence", [])),
            prior_evaluations=list(execution_memory.get("prior_evaluations", [])),
            allowed_paths=context.visible_paths,
            state_version=state.scientific_state.state_version,
            route_retry_state=state.route_retry_state,
        )
        request_prompt = context.request.to_prompt()
        self._emit(
            "execution_request_submitted",
            {
                "iteration": len(state.iterations),
                "step_id": context.step.step_id,
                "request": context.request.to_dict(),
                "prompt": request_prompt,
                "prompt_chars": len(request_prompt),
                "prompt_utf8_bytes": len(request_prompt.encode("utf-8")),
                "estimated_input_tokens": max(1, len(request_prompt) // 4),
            },
        )
        context.attempt_key = self._attempt_key(state, context.step.step_id)

    async def _graph_bind(self, context: _IterationContext) -> None:
        self._graph_enter(context, "bind", LoopPhase.BIND)
        state = context.state
        if context.plan is None:
            raise ValueError("bind node has no active plan")
        current_step = context.plan.steps[context.cursor]
        if context.step is None or context.step.step_id != current_step.step_id:
            context.request = None
            context.step = None
            context.attempt_key = ""
            context.binding = None
            context.binding_failed = False
            context.skip_dispatch = False
            context.pending = False
            context.result = None
            context.artifact_path = ""
            context.verification = None
            context.action = None
            context.action_disposition = ""
        if context.request is None:
            self._graph_prepare_request(context)
        assert context.request is not None
        try:
            await self._resolve_request_entity(context)
        except EntityResolutionError as exc:
            # Previous strict behavior is intentionally retained here as
            # commented code for easy rollback/reference:
            # context.binding_failed = True
            # context.skip_dispatch = True
            # context.result = A1TaskResult(
            #     success=False,
            #     result_status="entity_resolution_failed",
            #     errors=[f"ENTITY_RESOLUTION_FAILED: {exc}"],
            # )
            # self._emit(
            #     "entity_resolution_failed",
            #     {
            #         "iteration": len(state.iterations),
            #         "step_id": context.step.step_id if context.step else "",
            #         "error": str(exc),
            #     },
            # )
            # self._admit_action(context, result_hint=context.result)
            # self._mark_action_result(state, context, context.result)
            # self._graph_exit(context, "bind")
            # return

            if self._a1_only_entity_resolution_is_best_effort():
                self._emit(
                    "entity_resolution_deferred_to_a1",
                    {
                        "iteration": len(state.iterations),
                        "step_id": context.step.step_id if context.step else "",
                        "error": str(exc),
                        "reason": "a1_only_best_effort_resolution",
                    },
                )
            else:
                context.binding_failed = True
                context.skip_dispatch = True
                context.result = A1TaskResult(
                    success=False,
                    result_status="entity_resolution_failed",
                    errors=[f"ENTITY_RESOLUTION_FAILED: {exc}"],
                )
                self._emit(
                    "entity_resolution_failed",
                    {
                        "iteration": len(state.iterations),
                        "step_id": context.step.step_id if context.step else "",
                        "error": str(exc),
                    },
                )
                self._admit_action(context, result_hint=context.result)
                self._mark_action_result(state, context, context.result)
                self._graph_exit(context, "bind")
                return
        context.pending = state.pending_execution is not None
        context.skip_dispatch = False
        context.binding_failed = False
        if context.pending:
            pending = state.pending_execution
            assert pending is not None
            if (
                pending.step_id != context.request.step.step_id
                or pending.iteration != len(state.iterations)
            ):
                raise ValueError("checkpoint pending task does not match the current execution step")
            self._ensure_pending_action(state, pending, context.request)
            context.action = self._find_action_for_pending(state, pending)
            context.action_disposition = "attach"
            self._graph_exit(context, "bind")
            return

        if self._restore_action_for_request(context):
            self._graph_exit(context, "bind")
            return

        persisted = (
            self.persistence.load_execution_result(context.attempt_key)
            if self.persistence
            else None
        )
        if persisted is not None:
            context.result, context.artifact_path = persisted
            context.skip_dispatch = True
            self._admit_action(context, result_hint=context.result)
            self._mark_action_result(
                state,
                context,
                context.result,
                result_key=context.attempt_key,
            )
            self._emit(
                "execution_result_reused",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id,
                    "attempt_key": context.attempt_key,
                    "artifact": context.artifact_path,
                },
            )
            self._graph_exit(context, "bind")
            return

        bind = getattr(self.a1_tool, "bind", None)
        if bind is None:
            self._emit(
                "execution_binding_deferred",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id,
                    "reason": "execution backend does not expose a bind interface",
                },
            )
            self._admit_action(context)
            self._graph_exit(context, "bind")
            return

        self._emit(
            "execution_binding_started",
            {
                "iteration": len(state.iterations),
                "step_id": context.step.step_id,
                "state_version": state.scientific_state.state_version,
            },
        )
        try:
            context.binding = await bind(context.request)
        except Exception as exc:
            context.binding_failed = True
            context.skip_dispatch = True
            context.result = A1TaskResult(
                success=False,
                result_status="capability_binding_failed",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            self._emit(
                "execution_binding_failed",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id,
                    "error": context.result.errors[0],
                },
            )
        else:
            bound_backend = self._enum_value(
                getattr(context.binding, "backend", "")
            )
            if bound_backend == "unavailable":
                context.binding_failed = True
                context.skip_dispatch = True
                context.result = A1TaskResult(
                    success=False,
                    result_status="capability_unavailable",
                    errors=[
                        str(
                            getattr(context.binding, "rationale", "")
                            or "No admitted execution capability is available"
                        )
                    ],
                )
            self._emit(
                "execution_binding_completed",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id,
                    "binding": context.binding.to_dict(),
                },
            )
            self._save_checkpoint(state, "execution_bound")
            reusable = self._load_reusable_mcp_result(context.binding)
            if reusable is not None:
                context.result, context.artifact_path, source_attempt_key = reusable
                context.skip_dispatch = True
                self._admit_action(context, result_hint=context.result)
                self._mark_action_result(
                    state,
                    context,
                    context.result,
                    result_key=source_attempt_key,
                )
                self._emit(
                    "execution_result_reused",
                    {
                        "iteration": len(state.iterations),
                        "step_id": context.step.step_id,
                        "attempt_key": context.attempt_key,
                        "source_attempt_key": source_attempt_key,
                        "execution_signature": getattr(
                            context.binding, "execution_signature", ""
                        ),
                        "artifact": context.artifact_path,
                    },
                )
        if context.action is None and context.result is None:
            disposition = self._admit_action(context)
            if disposition == "attach":
                context.pending = state.pending_execution is not None
            elif disposition in {"reuse", "blocked"}:
                context.skip_dispatch = True
        elif context.action is None and context.result is not None:
            self._admit_action(context, result_hint=context.result)
            self._mark_action_result(state, context, context.result)
        self._graph_exit(context, "bind")

    def _a1_only_entity_resolution_is_best_effort(self) -> bool:
        """A1 owns entity resolution when Harness pre-resolution cannot resolve it."""
        routing_policy = getattr(self.a1_tool, "routing_policy", None)
        routing_mode = getattr(routing_policy, "mode", None)
        routing_mode = getattr(routing_mode, "value", routing_mode)
        return str(routing_mode or "").strip().casefold() == "a1_only"

    async def _resolve_request_entity(self, context: _IterationContext) -> None:
        request = context.request
        plan = context.plan
        if request is None or plan is None:
            return
        semantic_intent = request.step.inputs.get("semantic_intent")
        if not isinstance(semantic_intent, dict):
            return
        raw_entity_context = semantic_intent.get("entity_context")
        if not isinstance(raw_entity_context, dict):
            return

        def context_value(key: str) -> str:
            value = str(raw_entity_context.get(key, "")).strip()
            if value.casefold() in {"", "none", "null", "unknown", "n/a"}:
                return ""
            return value

        strong_identifier = next(
            (
                context_value(key)
                for key in (
                    "uniprot_accession",
                    "uniprot_id",
                    "accession",
                    "gene_symbol",
                    "gene_name",
                )
                if context_value(key)
            ),
            "",
        )
        local_source = next(
            (
                context_value(key)
                for key in (
                    "file_path",
                    "local_structure_path",
                    "structure_path",
                    "pdb_path",
                    "input_path",
                )
                if context_value(key)
            ),
            "",
        )
        if local_source and not strong_identifier:
            return
        query_name = strong_identifier or next(
            (
                context_value(key)
                for key in ("protein_name", "target_protein")
                if context_value(key)
            ),
            "",
        )
        if not query_name:
            return
        scientific_state = context.state.scientific_state
        existing = self.entity_resolver.find_existing(
            scientific_state.canonical_entities,
            query_name,
        )
        if existing is None:
            resolution = await self.entity_resolver.resolve(
                query_name,
                organism=str(
                    raw_entity_context.get("organism")
                    or raw_entity_context.get("species")
                    or ""
                ),
                entity_context=raw_entity_context,
            )
            scientific_state.canonical_entities[resolution.record.entity_id] = (
                resolution.record
            )
        else:
            resolution = self.entity_resolver.reuse(existing, raw_entity_context)

        corrections = resolution.corrections
        scientific_state.entity_corrections.update(corrections)
        if corrections:
            plan.hypothesis = self.entity_resolver.apply_corrections(
                plan.hypothesis, corrections
            )
            plan.rationale = self.entity_resolver.apply_corrections(
                plan.rationale, corrections
            )
            for step in plan.steps:
                step.objective = self.entity_resolver.apply_corrections(
                    step.objective, corrections
                )
                step.inputs = self.entity_resolver.apply_corrections(
                    step.inputs, corrections
                )
                step.constraints = self.entity_resolver.apply_corrections(
                    step.constraints, corrections
                )
                step.expected_outputs = self.entity_resolver.apply_corrections(
                    step.expected_outputs, corrections
                )
                step.success_criteria = self.entity_resolver.apply_corrections(
                    step.success_criteria, corrections
                )

        corrected_intent = request.step.inputs.get("semantic_intent")
        if isinstance(corrected_intent, dict):
            corrected_context = corrected_intent.get("entity_context")
            corrected_intent["entity_context"] = self.entity_resolver.canonical_context(
                corrected_context if isinstance(corrected_context, dict) else {},
                resolution,
            )
        request.hypothesis = plan.hypothesis
        request.canonical_entities = [resolution.record]
        request.entity_corrections = dict(scientific_state.entity_corrections)
        self._emit(
            "entity_resolution_completed",
            {
                "iteration": len(context.state.iterations),
                "step_id": request.step.step_id,
                "canonical_entity": resolution.record,
                "entity_corrections": corrections,
            },
        )

    def _admit_action(
        self,
        context: _IterationContext,
        *,
        result_hint: A1TaskResult | None = None,
    ) -> str:
        """Record one logical action before any external side effect."""
        if context.action is not None:
            return context.action_disposition or "existing"
        state = context.state
        request = context.request
        if request is None or context.step is None:
            raise ValueError("cannot admit an action without a request")

        binding = context.binding
        route = (
            binding.to_dict()
            if binding is not None and callable(getattr(binding, "to_dict", None))
            else {}
        )
        backend = self._enum_value(getattr(binding, "backend", "a1")) or "a1"
        capability_id = str(
            getattr(binding, "selected_capability", "")
            or getattr(binding, "admitted_capability", "")
            or route.get("selected_capability")
            or route.get("admitted_capability")
            or route.get("binding_id")
            or "call_biomni"
        )
        bound_call = getattr(binding, "bound_call", None)
        bound_workflow = getattr(binding, "bound_workflow", None)
        execution_payload = getattr(binding, "execution_payload", None)
        if not isinstance(execution_payload, dict) or not execution_payload:
            execution_payload = route.get("execution_payload")
        if isinstance(execution_payload, dict) and execution_payload:
            normalized_arguments = dict(execution_payload)
        elif bound_call is not None and callable(getattr(bound_call, "to_dict", None)):
            normalized_arguments = dict(bound_call.arguments)
        elif bound_workflow is not None and callable(
            getattr(bound_workflow, "to_dict", None)
        ):
            normalized_arguments = {"workflow": bound_workflow.to_dict()}
        else:
            normalized_arguments = {
                str(key): value
                for key, value in request.step.inputs.items()
                if key not in {"request_id", "retrieved_biomni_capabilities"}
            }
        execution_signature = str(
            getattr(binding, "execution_signature", "")
            or route.get("execution_signature", "")
        ).strip()
        if not execution_signature:
            execution_signature = canonical_action_key(
                {
                    "backend": backend,
                    "capability_id": capability_id,
                    "arguments": normalized_arguments,
                    "semantic_intent": request.step.inputs.get("semantic_intent"),
                    "objective": " ".join(request.step.objective.split()).casefold(),
                    "expected_outputs": sorted(
                        " ".join(str(item).split()).casefold()
                        for item in request.step.expected_outputs
                    ),
                }
            )
        catalog_revision = str(
            getattr(binding, "catalog_revision", "")
            or route.get("catalog_revision", "")
        ).strip()
        idempotency_key = str(
            getattr(binding, "idempotency_key", "")
            or route.get("idempotency_key", "")
        ).strip()
        if not idempotency_key:
            idempotency_key = canonical_action_key(
                {
                    "backend": backend,
                    "capability_id": capability_id,
                    "execution_signature": execution_signature,
                    "catalog_revision": catalog_revision,
                    "arguments": normalized_arguments,
                }
            )
        request_id = str(
            request.request_id
            or getattr(binding, "request_id", "")
            or route.get("request_id", "")
        ).strip()
        record, disposition = state.action_ledger.admit(
            run_id=state.run_id,
            idempotency_key=idempotency_key,
            iteration=request.iteration,
            step_id=request.step.step_id,
            backend=backend,
            capability_id=capability_id,
            normalized_arguments=normalized_arguments,
            route_signature=str(
                getattr(binding, "route_signature", "")
                or route.get("route_signature", "")
            ),
            request_id=request_id,
            max_attempts=2,
        )
        context.action = record
        context.action_disposition = disposition
        request.request_id = record.request_id
        binding_snapshot = {
            "backend": backend,
            "reason_code": str(getattr(binding, "reason_code", "")),
            "rationale": str(getattr(binding, "rationale", "")),
            "query": str(getattr(binding, "query", "") or request.step.objective),
            "requested_backend": str(getattr(binding, "requested_backend", "")),
            "admitted_capability": capability_id,
            "selected_capability": str(
                getattr(binding, "selected_capability", "")
            ),
            "route_signature": str(
                getattr(binding, "route_signature", "")
                or route.get("route_signature", "")
            ),
            "execution_signature": execution_signature,
            "execution_payload": dict(execution_payload or {}),
            "idempotency_key": idempotency_key,
            "binding_id": str(getattr(binding, "binding_id", "")),
            "catalog_revision": catalog_revision,
            "semantic_intent": request.step.inputs.get("semantic_intent"),
            "candidates": [item.to_dict() for item in getattr(binding, "candidates", [])],
            "condition_coverage": (
                binding.condition_coverage.to_dict()
                if binding is not None
                and callable(getattr(getattr(binding, "condition_coverage", None), "to_dict", None))
                else {}
            ),
            "output_contract": (
                dict(getattr(binding, "output_contract", {}))
                if isinstance(getattr(binding, "output_contract", {}), dict)
                else {}
            ),
            "evidence_purpose": str(
                getattr(binding, "evidence_purpose", "claim_evidence")
            ),
            "policy_metadata": (
                dict(getattr(binding, "policy_metadata", {}))
                if isinstance(getattr(binding, "policy_metadata", {}), dict)
                else {}
            ),
            "domain_workflow": (
                dict(getattr(binding, "domain_workflow", {}))
                if isinstance(getattr(binding, "domain_workflow", {}), dict)
                else {}
            ),
            "bound_call": (
                bound_call.to_dict()
                if bound_call is not None and callable(getattr(bound_call, "to_dict", None))
                else None
            ),
            "bound_workflow": (
                bound_workflow.to_dict()
                if bound_workflow is not None
                and callable(getattr(bound_workflow, "to_dict", None))
                else None
            ),
        }
        record.task_metadata = dict(record.task_metadata)
        record.task_metadata.setdefault("binding_snapshot", binding_snapshot)
        if binding is not None:
            try:
                binding.idempotency_key = record.idempotency_key
                binding.request_id = record.request_id
            except (AttributeError, TypeError):
                pass
        self._save_action_ledger(state)
        self._emit(
            "action_admitted",
            {
                "action_id": record.action_id,
                "idempotency_key": record.idempotency_key,
                "disposition": disposition,
                "status": record.status,
                "attempt": record.attempt,
                "backend": record.backend,
                "capability_id": record.capability_id,
                "request_id": record.request_id,
            },
        )
        self._save_checkpoint(state, "action_admitted")

        if disposition == "reuse":
            loaded = None
            if self.persistence is not None and record.result_key:
                loaded = self.persistence.load_execution_result(record.result_key)
            if loaded is None:
                self._mark_action_unknown(
                    state,
                    record,
                    "successful action has no durable result artifact",
                )
                context.result = A1TaskResult(
                    success=False,
                    result_status="action_reconciliation_required",
                    errors=[
                        "ACTION_RECONCILIATION_REQUIRED: successful action result is missing"
                    ],
                )
            else:
                context.result, context.artifact_path = loaded
                context.skip_dispatch = True
        elif disposition == "attach":
            if state.pending_execution is None and record.external_task_id:
                state.pending_execution = self._pending_from_action(record)
            if state.pending_execution is not None:
                context.pending = True
            else:
                self._mark_action_unknown(
                    state,
                    record,
                    "submitted action has no task ID to reconcile",
                )
                context.result = A1TaskResult(
                    success=False,
                    result_status="action_reconciliation_required",
                    errors=[
                        "ACTION_RECONCILIATION_REQUIRED: submitted action has no task ID"
                    ],
                )
                context.skip_dispatch = True
        elif disposition == "replay":
            # The external response may have been lost after submission. Reuse
            # the persisted request_id and exact bound request; Biomni decides
            # whether this is a replay and returns the original task_id.
            context.skip_dispatch = False
            self._emit(
                "action_replay_requested",
                {
                    "action_id": record.action_id,
                    "request_id": record.request_id,
                    "backend": record.backend,
                },
            )
        elif disposition == "conflict":
            context.result = A1TaskResult(
                success=False,
                result_status="idempotency_conflict",
                errors=[
                    "IDEMPOTENCY_CONFLICT: the same idempotency key maps to different "
                    "backend or request content"
                ],
            )
            context.skip_dispatch = True
            self._mark_action_unknown(
                state,
                record,
                "idempotency_conflict",
            )
        elif disposition == "blocked":
            context.result = A1TaskResult(
                success=False,
                result_status="duplicate_action_blocked",
                errors=[
                    "DUPLICATE_ACTION_BLOCKED: existing action requires reconciliation "
                    "or has exhausted its retry policy"
                ],
                task_metadata={
                    "action_id": record.action_id,
                    "idempotency_key": record.idempotency_key,
                    "action_retryable": record.retryable,
                    "blocked_existing_status": record.status,
                },
            )
            context.skip_dispatch = True
        return disposition

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "").strip().lower()

    def _save_action_ledger(self, state: ResearchState) -> None:
        if self.persistence is not None:
            self.persistence.save_action_ledger(state.action_ledger)

    def _find_action_for_pending(
        self,
        state: ResearchState,
        pending: PendingExecution,
    ) -> ActionRecord | None:
        if pending.action_id:
            record = state.action_ledger.records.get(pending.action_id)
            if record is not None:
                return record
        for record in state.action_ledger.records.values():
            if pending.idempotency_key and record.idempotency_key == pending.idempotency_key:
                return record
            if pending.task_id and record.external_task_id == pending.task_id:
                return record
        return None

    def _ensure_pending_action(
        self,
        state: ResearchState,
        pending: PendingExecution,
        request: A1TaskRequest,
    ) -> ActionRecord:
        existing = self._find_action_for_pending(state, pending)
        if existing is not None:
            pending.action_id = existing.action_id
            pending.idempotency_key = existing.idempotency_key
            return existing
        key = pending.idempotency_key or canonical_action_key(
            {
                "run_id": state.run_id,
                "backend": pending.backend,
                "request_id": pending.request_id,
                "task_id": pending.task_id,
                "step_id": pending.step_id,
            }
        )
        record, _ = state.action_ledger.admit(
            run_id=state.run_id,
            idempotency_key=key,
            iteration=pending.iteration,
            step_id=pending.step_id,
            backend=pending.backend,
            capability_id=str(pending.task_metadata.get("tool_name", "")),
            normalized_arguments=dict(request.step.inputs),
            request_id=str(
                pending.task_metadata.get("workflow_parent_request_id")
                or (
                    pending.task_metadata.get("omniagent_route", {}).get("request_id")
                    if isinstance(pending.task_metadata.get("omniagent_route"), dict)
                    else ""
                )
                or pending.request_id
            ),
        )
        state.action_ledger.update(
            record.action_id,
            self._pending_status(pending.status),
            external_task_id=pending.task_id,
            external_request_id=pending.request_id,
            task_metadata=dict(pending.task_metadata),
        )
        pending.action_id = record.action_id
        pending.idempotency_key = record.idempotency_key
        self._save_action_ledger(state)
        return record

    @staticmethod
    def _pending_status(status: str) -> ActionStatus:
        normalized = normalize_task_status(status)
        if normalized == ActionStatus.QUEUED.value:
            return ActionStatus.QUEUED
        if normalized == ActionStatus.RUNNING.value:
            return ActionStatus.RUNNING
        if normalized == ActionStatus.RETRY_WAIT.value:
            return ActionStatus.RETRY_WAIT
        return ActionStatus.SUBMITTED

    @staticmethod
    def _pending_from_action(record: ActionRecord) -> PendingExecution:
        metadata = dict(record.task_metadata)
        try:
            next_poll_at = float(metadata.get("next_poll_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            next_poll_at = 0.0
        try:
            deadline_at = float(metadata.get("deadline_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            deadline_at = 0.0
        return PendingExecution(
            step_id=record.step_id,
            iteration=record.iteration,
            task_id=record.external_task_id,
            request_id=str(
                record.external_request_id
                or metadata.get("request_id")
                or record.request_id
            ),
            gateway_tool=str(metadata.get("gateway_tool", "")),
            backend=record.backend,
            status=record.status,
            next_poll_at=next_poll_at,
            deadline_at=deadline_at,
            task_metadata=metadata,
            action_id=record.action_id,
            idempotency_key=record.idempotency_key,
            remote_status=str(metadata.get("status", "")),
            rpc_status="recovered",
        )

    def _mark_action_unknown(
        self,
        state: ResearchState,
        record: ActionRecord,
        reason: str,
    ) -> None:
        state.action_ledger.update(
            record.action_id,
            ActionStatus.UNKNOWN,
            error=reason,
        )
        self._save_action_ledger(state)

    def _update_pending_action(
        self,
        state: ResearchState,
        pending: PendingExecution,
        status: ActionStatus,
        reason: str = "",
        *,
        result: A1TaskResult | None = None,
    ) -> None:
        record = self._find_action_for_pending(state, pending)
        if record is None:
            return
        if result is not None and isinstance(result.task_metadata, dict):
            pending.task_metadata.update(result.task_metadata)
        pending.task_metadata.update(
            {
                "task_id": pending.task_id,
                "request_id": pending.request_id,
                "gateway_tool": pending.gateway_tool,
                "next_poll_at": pending.next_poll_at,
                "deadline_at": pending.deadline_at,
                "remote_status": pending.remote_status,
                "rpc_status": pending.rpc_status,
            }
        )
        state.action_ledger.update(
            record.action_id,
            status,
            external_task_id=pending.task_id,
            external_request_id=pending.request_id,
            result_status=(result.result_status if result is not None else None),
            error=reason or pending.last_poll_error or None,
            task_metadata=dict(pending.task_metadata),
        )
        self._save_action_ledger(state)

    def _mark_action_result(
        self,
        state: ResearchState,
        context: _IterationContext,
        result: A1TaskResult,
        *,
        result_key: str = "",
        terminal: bool = True,
    ) -> None:
        record = context.action
        if record is None:
            return
        if (
            getattr(context, "action_disposition", "") == "blocked"
            and record.is_terminal
        ):
            return
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        task_id = str(metadata.get("task_id") or record.external_task_id).strip()
        request_id = str(metadata.get("request_id") or record.request_id).strip()
        status = self._action_status_for_result(result, terminal=terminal)
        ledger_metadata = (
            external_task_summary(metadata)
            if terminal and result.result_status != "task_pending"
            else dict(metadata)
        )
        updated = state.action_ledger.update(
            record.action_id,
            status,
            external_task_id=task_id or None,
            external_request_id=request_id or None,
            result_key=result_key or None,
            result_status=result.result_status,
            error=(result.errors[-1] if result.errors else None),
            retryable=(
                execution_failure_retryable(result)
                if status is ActionStatus.FAILED
                else None
            ),
            task_metadata=ledger_metadata,
        )
        if state.pending_execution is not None:
            state.pending_execution.action_id = record.action_id
            state.pending_execution.idempotency_key = record.idempotency_key
        self._save_action_ledger(state)
        self._emit(
            "action_state_changed",
            {
                "action_id": record.action_id,
                "status": status.value,
                "task_id": task_id or None,
                "request_id": request_id or None,
                "result_status": result.result_status,
                "terminal": terminal,
                "retryable": updated.retryable,
            },
        )

    @classmethod
    def _action_status_for_result(
        cls,
        result: A1TaskResult,
        *,
        terminal: bool,
    ) -> ActionStatus:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        remote_status = str(metadata.get("status", "")).strip().lower()
        if remote_status in {"timed_out", "timeout", "timedout"}:
            return ActionStatus.TIMED_OUT
        if remote_status in {"failed", "cancelled", "canceled"}:
            return ActionStatus.FAILED
        if remote_status == "manual_review":
            return ActionStatus.MANUAL_REVIEW
        if remote_status == "dead_letter":
            return ActionStatus.DEAD_LETTER
        if remote_status == "unknown":
            return ActionStatus.UNKNOWN
        if metadata.get("submission_unknown") or metadata.get("rpc_status") == "timeout":
            return ActionStatus.UNKNOWN
        if metadata.get("idempotency_conflict"):
            return ActionStatus.FAILED
        if result.result_status in {"task_wait_timed_out", "task_unknown", "poll_transient_error"}:
            return ActionStatus.UNKNOWN
        if result.result_status in {"task_timed_out", "remote_task_timed_out"}:
            return ActionStatus.TIMED_OUT
        if result.result_status in {
            "idempotency_conflict",
            "action_reconciliation_required",
        }:
            return ActionStatus.UNKNOWN
        if result.result_status == "task_pending":
            return cls._pending_status(remote_status)
        if not terminal:
            if result.success:
                return ActionStatus.RUNNING
            if result.result_status in {"task_resume_failed", "execution_transport_error", "error"}:
                return ActionStatus.UNKNOWN
            return ActionStatus.FAILED
        if result.success:
            return ActionStatus.SUCCEEDED
        if result.result_status in {"task_resume_failed", "execution_transport_error"}:
            return ActionStatus.UNKNOWN
        return ActionStatus.FAILED

    def _restore_pending_from_action_ledger(self, state: ResearchState) -> None:
        if state.pending_execution is not None:
            return
        candidates = [
            item
            for item in state.action_ledger.unresolved()
            if item.external_task_id
        ]
        if len(candidates) != 1:
            return
        state.pending_execution = self._pending_from_action(candidates[0])

    def _restore_action_for_request(
        self,
        context: _IterationContext,
    ) -> bool:
        """Restore a checkpointed binding before asking a selector to rebind."""
        if context.action is not None or context.request is None:
            return context.action is not None
        state = context.state
        request = context.request
        matches = [
            item
            for item in state.action_ledger.records.values()
            if item.iteration == request.iteration
            and item.step_id == request.step.step_id
            and item.status
            in {
                ActionStatus.ADMITTED.value,
                ActionStatus.SUBMITTED.value,
                ActionStatus.QUEUED.value,
                ActionStatus.RUNNING.value,
                ActionStatus.UNKNOWN.value,
                ActionStatus.SUCCEEDED.value,
            }
        ]
        if not matches:
            return False
        record = max(matches, key=lambda item: (item.updated_at, item.attempt))
        context.action = record
        request.request_id = record.request_id
        context.binding = self._binding_from_snapshot(record)

        if record.status == ActionStatus.SUCCEEDED.value:
            loaded = (
                self.persistence.load_execution_result(record.result_key)
                if self.persistence is not None and record.result_key
                else None
            )
            if loaded is None:
                self._mark_action_unknown(
                    state, record, "successful action result is missing during resume"
                )
                context.result = A1TaskResult(
                    success=False,
                    result_status="action_reconciliation_required",
                    errors=[
                        "ACTION_RECONCILIATION_REQUIRED: successful action result is missing"
                    ],
                )
            else:
                context.result, context.artifact_path = loaded
                context.skip_dispatch = True
            context.action_disposition = "reuse"
            return True

        if record.external_task_id:
            state.pending_execution = self._pending_from_action(record)
            context.pending = True
            context.action_disposition = "attach"
            return True
        if record.status in {
            ActionStatus.ADMITTED.value,
            ActionStatus.SUBMITTED.value,
        } or (
            record.status == ActionStatus.UNKNOWN.value
            and record.task_metadata.get("submission_unknown")
        ):
            context.action_disposition = "replay"
            return True
        context.action_disposition = "blocked"
        context.result = A1TaskResult(
            success=False,
            result_status="action_reconciliation_required",
            errors=[
                "ACTION_RECONCILIATION_REQUIRED: external task has no recoverable identity"
            ],
        )
        context.skip_dispatch = True
        return True

    @staticmethod
    def _binding_from_snapshot(record: ActionRecord) -> RouteDecision | None:
        snapshot = record.task_metadata.get("binding_snapshot")
        if not isinstance(snapshot, dict):
            return None
        try:
            backend = ExecutionBackend(str(snapshot.get("backend", "a1")))
        except ValueError:
            return None
        raw_intent = snapshot.get("semantic_intent")
        intent = (
            SemanticCapabilityIntent.from_inputs({"semantic_intent": raw_intent})
            if isinstance(raw_intent, dict)
            else None
        )
        candidates: list[ResourceCandidate] = []
        raw_candidates = snapshot.get("candidates", [])
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if not isinstance(item, dict) or not item.get("qualified_name"):
                    continue
                candidates.append(
                    ResourceCandidate(
                        qualified_name=str(item["qualified_name"]),
                        description=str(item.get("description", "")),
                        score=(
                            float(item["score"])
                            if isinstance(item.get("score"), int | float)
                            else None
                        ),
                        input_schema=(
                            dict(item.get("input_schema", {}))
                            if isinstance(item.get("input_schema", {}), dict)
                            else {}
                        ),
                        output_schema=(
                            dict(item.get("output_schema", {}))
                            if isinstance(item.get("output_schema", {}), dict)
                            else {}
                        ),
                        capability_version=str(item.get("capability_version") or ""),
                        effect_contract=(
                            dict(item.get("effect_contract", {}))
                            if isinstance(item.get("effect_contract", {}), dict)
                            else {}
                        ),
                        result_adapter=str(item.get("result_adapter") or "generic"),
                        execution_mode=str(item.get("execution_mode") or "sync"),
                        lifecycle=(
                            dict(item.get("lifecycle", {}))
                            if isinstance(item.get("lifecycle", {}), dict)
                            else {}
                        ),
                        retry_policy=(
                            dict(item.get("retry_policy", {}))
                            if isinstance(item.get("retry_policy", {}), dict)
                            else {}
                        ),
                        timeout_policy=(
                            dict(item.get("timeout_policy", {}))
                            if isinstance(item.get("timeout_policy", {}), dict)
                            else {}
                        ),
                        idempotency_policy=(
                            dict(item.get("idempotency_policy", {}))
                            if isinstance(item.get("idempotency_policy", {}), dict)
                            else {}
                        ),
                        provenance_policy=(
                            dict(item.get("provenance_policy", {}))
                            if isinstance(item.get("provenance_policy", {}), dict)
                            else {}
                        ),
                    )
                )
        coverage_raw = snapshot.get("condition_coverage", {})
        coverage = (
            ConditionCoverage(
                required_conditions=tuple(coverage_raw.get("required_conditions", [])),
                binding_covered_conditions=tuple(
                    coverage_raw.get("binding_covered_conditions", [])
                ),
                verification_required_conditions=tuple(
                    coverage_raw.get("verification_required_conditions", [])
                ),
                uncovered_conditions=tuple(coverage_raw.get("uncovered_conditions", [])),
            )
            if isinstance(coverage_raw, dict)
            else ConditionCoverage()
        )
        return RouteDecision(
            backend=backend,
            reason_code=str(snapshot.get("reason_code", "resumed_binding")),
            rationale=str(snapshot.get("rationale", "resumed checkpoint binding")),
            query=str(snapshot.get("query", "")),
            candidates=candidates,
            requested_backend=str(snapshot.get("requested_backend", "")),
            admitted_capability=str(snapshot.get("admitted_capability", "")),
            selected_capability=str(snapshot.get("selected_capability", "")),
            route_signature=str(snapshot.get("route_signature", record.route_signature)),
            execution_signature=str(
                snapshot.get("execution_signature", "")
            ),
            execution_payload=(
                dict(snapshot.get("execution_payload", {}))
                if isinstance(snapshot.get("execution_payload", {}), dict)
                else {}
            ),
            idempotency_key=record.idempotency_key,
            request_id=record.request_id,
            binding_id=str(snapshot.get("binding_id", "")),
            catalog_revision=str(snapshot.get("catalog_revision", "")),
            condition_coverage=coverage,
            output_contract=(
                dict(snapshot.get("output_contract", {}))
                if isinstance(snapshot.get("output_contract", {}), dict)
                else {}
            ),
            evidence_purpose=str(
                snapshot.get("evidence_purpose", "claim_evidence")
            ),
            policy_metadata=(
                dict(snapshot.get("policy_metadata", {}))
                if isinstance(snapshot.get("policy_metadata", {}), dict)
                else {}
            ),
            semantic_intent=intent,
            bound_call=ClosedLoopRuntime._bound_call_from_dict(
                snapshot.get("bound_call")
            ),
            bound_workflow=ClosedLoopRuntime._bound_workflow_from_dict(
                snapshot.get("bound_workflow")
            ),
            domain_workflow=(
                dict(snapshot.get("domain_workflow", {}))
                if isinstance(snapshot.get("domain_workflow", {}), dict)
                else {}
            ),
        )

    @staticmethod
    def _effect_contract_from_dict(value: Any) -> EffectContract:
        if not isinstance(value, dict):
            return EffectContract()
        requirements: list[PathValueRequirement] = []
        for item in value.get("required_value_matches", []):
            if not isinstance(item, dict):
                continue
            requirements.append(
                PathValueRequirement(
                    path=str(item.get("path", "")),
                    expected_values=tuple(str(v) for v in item.get("expected_values", [])),
                    case_sensitive=bool(item.get("case_sensitive", False)),
                )
            )
        return EffectContract(
            required_paths=tuple(str(v) for v in value.get("required_paths", [])),
            any_of_paths=tuple(str(v) for v in value.get("any_of_paths", [])),
            required_value_matches=tuple(requirements),
            required_artifacts=tuple(str(v) for v in value.get("required_artifacts", [])),
            description=str(value.get("description", "")),
        )

    @classmethod
    def _bound_call_from_dict(cls, value: Any) -> BoundCapabilityCall | None:
        if not isinstance(value, dict) or not str(value.get("tool_name", "")).strip():
            return None
        return BoundCapabilityCall(
            tool_name=str(value["tool_name"]),
            arguments=(
                dict(value.get("arguments", {}))
                if isinstance(value.get("arguments", {}), dict)
                else {}
            ),
            input_schema=(
                dict(value.get("input_schema", {}))
                if isinstance(value.get("input_schema", {}), dict)
                else {}
            ),
            output_schema=(
                dict(value.get("output_schema", {}))
                if isinstance(value.get("output_schema", {}), dict)
                else {}
            ),
            effects=cls._effect_contract_from_dict(value.get("effects")),
            binding_reason=str(value.get("binding_reason", "resumed_binding")),
            capability_version=str(value.get("capability_version") or ""),
            result_adapter=str(value.get("result_adapter") or "generic"),
            execution_mode=str(value.get("execution_mode") or "sync"),
            lifecycle=(
                dict(value.get("lifecycle", {}))
                if isinstance(value.get("lifecycle", {}), dict)
                else {}
            ),
            retry_policy=(
                dict(value.get("retry_policy", {}))
                if isinstance(value.get("retry_policy", {}), dict)
                else {}
            ),
            timeout_policy=(
                dict(value.get("timeout_policy", {}))
                if isinstance(value.get("timeout_policy", {}), dict)
                else {}
            ),
            idempotency_policy=(
                dict(value.get("idempotency_policy", {}))
                if isinstance(value.get("idempotency_policy", {}), dict)
                else {}
            ),
            provenance_policy=(
                dict(value.get("provenance_policy", {}))
                if isinstance(value.get("provenance_policy", {}), dict)
                else {}
            ),
        )

    @classmethod
    def _bound_workflow_from_dict(cls, value: Any) -> BoundCapabilityWorkflow | None:
        if not isinstance(value, dict) or not str(value.get("workflow_id", "")).strip():
            return None
        steps: list[WorkflowCallTemplate] = []
        for item in value.get("steps", []):
            if not isinstance(item, dict) or not str(item.get("tool_name", "")).strip():
                continue
            steps.append(
                WorkflowCallTemplate(
                    step_id=str(item.get("step_id", "")),
                    tool_name=str(item["tool_name"]),
                    arguments=(
                        dict(item.get("arguments", {}))
                        if isinstance(item.get("arguments", {}), dict)
                        else {}
                    ),
                    input_schema=(
                        dict(item.get("input_schema", {}))
                        if isinstance(item.get("input_schema", {}), dict)
                        else {}
                    ),
                    output_schema=(
                        dict(item.get("output_schema", {}))
                        if isinstance(item.get("output_schema", {}), dict)
                        else {}
                    ),
                    effects=cls._effect_contract_from_dict(item.get("effects")),
                )
            )
        return BoundCapabilityWorkflow(
            workflow_id=str(value["workflow_id"]),
            inputs=(
                dict(value.get("inputs", {}))
                if isinstance(value.get("inputs", {}), dict)
                else {}
            ),
            steps=steps,
            effects=cls._effect_contract_from_dict(value.get("effects")),
            max_steps=int(value.get("max_steps", 5)),
            binding_reason=str(value.get("binding_reason", "resumed_binding")),
            evidence_purpose=str(value.get("evidence_purpose", "claim_evidence")),
        )

    async def _graph_dispatch(self, context: _IterationContext) -> None:
        self._graph_enter(context, "dispatch", LoopPhase.DISPATCH)
        state = context.state
        if context.request is None:
            raise ValueError("dispatch node requires a prepared request")
        if context.result is None:
            if context.action is None:
                self._admit_action(context)
            if context.action is None:
                raise ValueError("dispatch node has no admitted action")
            state.action_ledger.update(
                context.action.action_id,
                ActionStatus.SUBMITTED,
                request_id=context.action.request_id,
                result_status="submitted",
            )
            self._save_action_ledger(state)
            self._emit(
                "action_state_changed",
                {
                    "action_id": context.action.action_id,
                    "status": ActionStatus.SUBMITTED.value,
                    "request_id": context.action.request_id,
                    "terminal": False,
                },
            )
            self._save_checkpoint(state, "action_submitted")
            self._emit(
                "execution_dispatch_started",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id if context.step else None,
                    "bound": context.binding is not None,
                },
            )
            try:
                if context.binding is not None and hasattr(self.a1_tool, "dispatch"):
                    result = await self.a1_tool.dispatch(context.request, context.binding)
                else:
                    result = await self.a1_tool.run(context.request)
            except Exception as exc:
                result = A1TaskResult(
                    success=False,
                    result_status="error",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            context.result = result
            if self._execution_used_a1(result):
                state.a1_call_count += 1
            self._mark_action_result(
                state,
                context,
                result,
                terminal=False,
            )
            self._save_checkpoint(state, "dispatch_result_received")

        assert context.result is not None
        initial_remote_status = self._external_task_status(context.result)
        if initial_remote_status in {"manual_review", "dead_letter"}:
            state.status = RunStatus.NEEDS_REVIEW
            state.phase = LoopPhase.REVIEW
            state.finish_reason = "external_task_requires_review"
            self._mark_action_result(
                state,
                context,
                context.result,
                terminal=True,
            )
            context.terminal = True
            self._emit(
                "external_task_requires_review",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id if context.step else None,
                    "task_id": context.result.task_metadata.get("task_id")
                    if isinstance(context.result.task_metadata, dict)
                    else None,
                    "request_id": context.result.task_metadata.get("request_id")
                    if isinstance(context.result.task_metadata, dict)
                    else None,
                    "remote_status": initial_remote_status,
                },
            )
            self._save_checkpoint(state, "external_task_requires_review")
            self._graph_exit(context, "dispatch")
            return
        if context.result.result_status == "task_pending":
            pending = self._create_pending_execution(context.request, context.result)
            if context.action is not None:
                pending.action_id = context.action.action_id
                pending.idempotency_key = context.action.idempotency_key
            state.pending_execution = pending
            context.result.task_metadata = dict(context.result.task_metadata) | {
                "next_poll_at": pending.next_poll_at,
                "deadline_at": pending.deadline_at,
                "action_id": pending.action_id,
                "idempotency_key": pending.idempotency_key,
            }
            self._mark_action_result(
                state,
                context,
                context.result,
                terminal=False,
            )
            context.pending = True
            state.phase = LoopPhase.WAIT_EXTERNAL
            self._emit(
                "external_task_submitted",
                {
                    "iteration": pending.iteration,
                    "step_id": pending.step_id,
                    "task_id": pending.task_id,
                    "request_id": pending.request_id,
                    "gateway_tool": pending.gateway_tool,
                    "backend": pending.backend,
                },
            )
            self._save_checkpoint(state, "external_task_submitted")
            self._save_pending_snapshot(state)
        self._graph_exit(context, "dispatch")

    async def _graph_wait_external(self, context: _IterationContext) -> None:
        self._graph_enter(context, "wait_external", LoopPhase.WAIT_EXTERNAL)
        state = context.state
        if context.request is None:
            self._graph_prepare_request(context)
        assert context.request is not None
        pending = state.pending_execution
        if pending is not None:
            if (
                context.step is None
                or pending.step_id != context.step.step_id
                or pending.iteration != len(state.iterations)
            ):
                raise ValueError("pending task does not match the state graph request")
            self._ensure_pending_action(state, pending, context.request)
            context.action = self._find_action_for_pending(state, pending)
            context.result = await self._wait_for_external_task(
                state, context.request, pending
            )
        context.pending = state.pending_execution is not None
        if state.status is not RunStatus.RUNNING:
            context.terminal = True
        self._graph_exit(context, "wait_external")

    def _graph_verify(self, context: _IterationContext) -> None:
        self._graph_enter(context, "verify", LoopPhase.VERIFY)
        if context.request is None or context.result is None:
            raise ValueError("verify node requires request and result")
        state = context.state
        result = context.result
        self._mark_external_failure_retry_state(result)
        self._normalize_execution_result(state, result)
        if result.success:
            try:
                self._validate_output_contract(state, result)
            except Exception as exc:
                result.success = False
                result.result_status = "output_contract_failed"
                result.errors.append(
                    f"OUTPUT_CONTRACT_FAILED: {type(exc).__name__}: {exc}"
                )
        context.verification = context.result_verifier.verify(
            state.scientific_state, context.request, result
        )
        self._graph_exit(context, "verify")

    def _graph_reduce(self, context: _IterationContext) -> None:
        self._graph_enter(context, "reduce", LoopPhase.REDUCE)
        if context.verification is None or context.result is None or context.step is None:
            raise ValueError("reduce node requires verification, result, and step")
        state = context.state
        verification = context.verification
        self.state_reducer.apply(state.scientific_state, verification)
        metadata = (
            dict(context.result.task_metadata)
            if isinstance(context.result.task_metadata, dict)
            else {}
        )
        metadata["omniagent_verification"] = {
            "accepted": verification.accepted,
            "reason": verification.reason,
            "verifier_id": verification.verifier_id,
            "transition_id": verification.transition_id,
            "evidence_ids": [item.evidence_id for item in verification.evidence],
        }
        context.result.task_metadata = metadata
        self._emit(
            "execution_result_verified",
            {
                "iteration": len(state.iterations),
                "step_id": context.step.step_id,
                "accepted": verification.accepted,
                "reason": verification.reason,
                "verifier_id": verification.verifier_id,
                "transition_id": verification.transition_id,
                "state_version": state.scientific_state.state_version,
                "evidence_ids": [item.evidence_id for item in verification.evidence],
            },
        )
        self._emit(
            "scientific_state_reduced",
            {
                "iteration": len(state.iterations),
                "step_id": context.step.step_id,
                "accepted": verification.accepted,
                "state_version": state.scientific_state.state_version,
            },
        )
        if not verification.accepted:
            context.result.success = False
            context.result.result_status = "verification_rejected"
            if verification.reason not in context.result.errors:
                context.result.errors.append(verification.reason)
        self._graph_exit(context, "reduce")

    def _graph_materialize(self, context: _IterationContext) -> None:
        self._graph_enter(context, "materialize", LoopPhase.MATERIALIZE)
        if context.result is None or context.verification is None or context.step is None:
            raise ValueError("materialize node requires result, verification, and step")
        state = context.state
        result = context.result
        if result.output is not None and isinstance(
            state.task_manifest.get("output_config"), dict
        ):
            self._emit(
                "execution_output_deferred",
                {
                    "iteration": len(state.iterations),
                    "step_id": context.step.step_id,
                    "reason": "final output is materialized only from verified Harness synthesis",
                },
            )
        self._record_route_retry_state(state, result)
        if self.persistence:
            context.artifact_path = self.persistence.save_execution_result(
                context.attempt_key, result
            )
            if context.artifact_path not in result.artifacts:
                result.artifacts.append(context.artifact_path)
        self._mark_action_result(
            state,
            context,
            result,
            result_key=context.attempt_key,
            terminal=True,
        )
        context.executions.append(result)
        self._emit(
            "execution_completed",
            {
                "iteration": len(state.iterations),
                "step_id": context.step.step_id,
                "success": result.success,
                "result_status": result.result_status,
                "answer_chars": len(result.answer),
                "metrics": result.metrics,
                "errors": [self._event_text(item) for item in result.errors[:8]],
                "biomni_task": self._task_event_summary(result.task_metadata) or None,
            },
        )
        state.active_executions.append(result)
        context.cursor += 1
        self._save_checkpoint(state, "execution_completed")
        self._graph_exit(context, "materialize")

    async def _graph_finalize(self, context: _IterationContext) -> None:
        self._graph_enter(context, "finalize", LoopPhase.FINALIZE)
        state = context.state
        if context.plan is None:
            raise ValueError("finalize node has no active plan")
        context.executions = list(state.active_executions)
        if not context.executions:
            self._finish_at_limit(state, "a1_call_budget_exhausted")
            context.terminal = True
            self._graph_exit(context, "finalize")
            return

        plan = context.plan
        executions = context.executions
        self._emit(
            "observations_recorded",
            {
                "iteration": len(state.iterations),
                "executions": [
                    self._execution_event_summary(item) for item in executions
                ],
            },
        )
        context.analysis = await self.analyzer.analyze(state, plan, executions)
        self._emit(
            "analysis_completed",
            {"iteration": len(state.iterations), "analysis": context.analysis},
        )
        self._finalize_analysis_output(
            state,
            plan,
            context.analysis,
            context.result_verifier,
            context.visible_paths,
        )
        state.phase = LoopPhase.FINALIZE
        context.evaluation = context.evaluator.evaluate(
            state, plan, executions, context.analysis
        )
        if not 0.0 <= context.evaluation.score <= 1.0:
            raise ValueError("Deterministic evaluator score must lie in [0, 1]")
        self._emit(
            "evaluation_completed",
            {
                "iteration": len(state.iterations),
                "evaluation": context.evaluation,
                "harness_score": context.evaluation.score,
                "score_semantics": "execution_contract_quality",
            },
        )
        context.critique = await self.verifier.verify(
            state, plan, context.analysis, context.evaluation
        )
        if context.critique.source_evaluation_id != context.evaluation.evaluation_id:
            raise ValueError("Feedback decision is not bound to the current evaluation")
        if context.critique.score != context.evaluation.score:
            raise ValueError("Feedback decision attempted to replace deterministic score")
        context.finalization = self.finalization_gate.evaluate(
            state, context.evaluation, context.critique
        )
        self._emit(
            "finalization_evaluated",
            {
                "iteration": len(state.iterations),
                "evaluation_id": context.evaluation.evaluation_id,
                **context.finalization.to_dict(),
            },
        )
        self._apply_finalization_decision(context)
        self._graph_exit(context, "finalize")

    def _apply_finalization_decision(self, context: _IterationContext) -> None:
        state = context.state
        plan = context.plan
        executions = context.executions
        evaluation = context.evaluation
        critique = context.critique
        finalization = context.finalization
        assert plan is not None
        assert evaluation is not None
        assert critique is not None
        assert finalization is not None
        terminal_contract = evaluation.metrics.get("terminal_contract", 0.0) >= 1.0
        insufficient_evidence = finalization.mode == "insufficient_evidence"
        execution_insufficient = any(
            isinstance(item.output, dict)
            and item.output.get("status") == "INSUFFICIENT_EVIDENCE"
            for item in executions
        )
        if terminal_contract and finalization.eligible and finalization.objective_satisfied:
            original_decision = critique.decision.value
            critique.decision = Decision.STOP
            critique.retryable = False
            self._emit(
                "terminal_contract_accepted",
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "harness_score": evaluation.score,
                    "original_decision": original_decision,
                    "reason": "valid_terminal_output_contract",
                },
            )
        elif terminal_contract and not insufficient_evidence:
            self._emit(
                "terminal_contract_rejected",
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "blockers": finalization.blockers,
                },
            )
        completed_iterations = len(state.iterations) + 1
        if critique.decision is Decision.STOP and (
            (evaluation.score < self.policy.target_score and not terminal_contract)
            or not finalization.eligible
            or not finalization.objective_satisfied
        ):
            self._emit(
                "stop_rejected",
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "harness_score": evaluation.score,
                    "target_harness_score": self.policy.target_score,
                    "completed_iterations": completed_iterations,
                    "reason": (
                        "finalization_gate_blocked"
                        if not finalization.eligible
                        else "target_score_not_reached"
                    ),
                    "finalization_blockers": finalization.blockers,
                },
            )
            critique.decision = Decision.REPLAN
            critique.required_changes = list(
                dict.fromkeys(
                    critique.required_changes
                    + list(evaluation.failed_criteria)
                    + list(finalization.blockers)
                )
            ) or [
                "Run a confirmatory analysis that tests the current result against an independent criterion."
            ]
            if not critique.next_experiment.strip():
                critique.next_experiment = critique.required_changes[0]
        if (
            critique.decision is Decision.FAIL
            and evaluation.retryable
            and evaluation.score < self.policy.target_score
            and completed_iterations < self.policy.max_iterations
            and state.a1_call_count < self.policy.max_a1_calls
        ):
            self._emit(
                "retryable_failure_replanned",
                {"evaluation_id": evaluation.evaluation_id, "errors": evaluation.errors},
            )
            critique.decision = Decision.REPLAN
            critique.required_changes = (
                critique.required_changes
                or list(evaluation.failed_criteria)
                or ["Retry with a narrower executable step and address prior errors."]
            )
            if not critique.next_experiment.strip():
                critique.next_experiment = critique.required_changes[0]

        previous_best = state.best_score
        improvement = critique.score - previous_best
        progress_key = self._progress_signature(plan, executions, context.analysis)
        previous_progress = {
            self._progress_signature(record.plan, record.executions, record.analysis)
            for record in state.iterations
        }
        progress_detected = (
            progress_key not in previous_progress
            and self._has_verified_execution_progress(executions)
        )
        if critique.score >= previous_best + self.policy.min_improvement or progress_detected:
            state.best_score = critique.score
            state.stalled_iterations = 0
        else:
            state.best_score = max(previous_best, critique.score)
            state.stalled_iterations += 1

        record = IterationRecord(
            iteration=len(state.iterations),
            plan=plan,
            executions=executions,
            analysis=context.analysis,
            evaluation=evaluation,
            critique=critique,
            score_improvement=improvement,
        )
        state.iterations.append(record)
        state.active_plan = None
        state.active_executions.clear()
        self._emit(
            "iteration_verified",
            {
                "record": {
                    "iteration": record.iteration,
                    "executions": [
                        self._execution_event_summary(item)
                        for item in record.executions
                    ],
                    "evaluation_id": evaluation.evaluation_id,
                    "score": evaluation.score,
                    "feedback_id": critique.feedback_id,
                    "decision": critique.decision.value,
                },
                "best_score": state.best_score,
                "best_harness_score": state.best_score,
                "progress_detected": progress_detected,
                "progress_key": progress_key,
                "stalled_iterations": state.stalled_iterations,
            },
        )
        self._emit(
            "feedback_decided",
            {
                "iteration": record.iteration,
                "evaluation_id": evaluation.evaluation_id,
                "feedback_id": critique.feedback_id,
                "decision": critique.decision,
                "harness_score": evaluation.score,
                "progress_detected": progress_detected,
                "stalled_iterations": state.stalled_iterations,
                "metrics": evaluation.metrics,
                "required_changes": critique.required_changes,
                "next_experiment": critique.next_experiment,
            },
        )
        self._save_checkpoint(state, "iteration_verified")

        if execution_insufficient:
            self._needs_review(state, "insufficient_evidence")
        elif insufficient_evidence and finalization.eligible:
            self._needs_review(state, "insufficient_evidence")
        elif critique.decision is Decision.FAIL:
            self._fail(state, "verifier_declared_failure")
        elif critique.decision is Decision.STOP and finalization.eligible:
            self._complete(state, "verifier_stop")
        elif state.best_score >= self.policy.target_score and finalization.eligible:
            self._complete(state, "harness_target_reached")
        elif state.stalled_iterations >= self.policy.max_stalled_iterations:
            self._finish_at_limit(state, "improvement_plateau", finalization=finalization)
        elif len(state.iterations) >= self.policy.max_iterations:
            self._finish_at_limit(state, "max_iterations_reached", finalization=finalization)
        elif state.a1_call_count >= self.policy.max_a1_calls:
            self._finish_at_limit(state, "a1_call_budget_exhausted", finalization=finalization)
        elif not any(result.success for result in executions) and not critique.retryable:
            self._fail(state, "non_retryable_execution_failure")
        else:
            state.phase = LoopPhase.REPLAN
            self._emit(
                "replan_required",
                {
                    "feedback_id": critique.feedback_id,
                    "evaluation_id": evaluation.evaluation_id,
                    "harness_score": evaluation.score,
                    "required_changes": critique.required_changes,
                    "next_experiment": critique.next_experiment,
                },
            )

    async def _wait_for_external_task(
        self,
        state: ResearchState,
        request: A1TaskRequest,
        pending: PendingExecution,
    ) -> A1TaskResult:
        """Poll one submitted task without dropping its identity on uncertainty."""
        self._ensure_pending_action(state, pending, request)
        while True:
            previous_status = pending.status
            previous_error = pending.last_poll_error
            stage_transition: dict[str, Any] | None = None
            now = time()
            if pending.deadline_at and now >= pending.deadline_at:
                reason = (
                    "TASK_WAIT_TIMEOUT: local wait deadline expired before Biomni "
                    f"reported a terminal state for task {pending.task_id}"
                )
                pending.status = ActionStatus.UNKNOWN.value
                pending.rpc_status = "deadline_exceeded"
                pending.unknown_reason = reason
                pending.last_poll_error = reason
                pending.next_poll_at = now + self._task_poll_interval_seconds()
                pending.task_metadata.update(
                    {
                        "status": ActionStatus.UNKNOWN.value,
                        "last_poll_error": reason,
                        "next_poll_at": pending.next_poll_at,
                        "deadline_at": pending.deadline_at,
                    }
                )
                state.pending_execution = pending
                self._update_pending_action(state, pending, ActionStatus.UNKNOWN, reason)
                self._save_pending_snapshot(state)
                state.status = RunStatus.NEEDS_REVIEW
                state.phase = LoopPhase.REVIEW
                state.finish_reason = "external_task_wait_timeout"
                self._emit(
                    "external_task_wait_timeout",
                    {
                        "iteration": pending.iteration,
                        "step_id": pending.step_id,
                        "task_id": pending.task_id,
                        "request_id": pending.request_id,
                        "deadline_at": pending.deadline_at,
                        "reason": reason,
                    },
                )
                self._save_checkpoint(state, "external_task_wait_timeout")
                return A1TaskResult(
                    success=False,
                    result_status="task_wait_timed_out",
                    errors=[reason],
                    task_metadata=dict(pending.task_metadata),
                )

            delay = max(0.0, pending.next_poll_at - now)
            if delay:
                await asyncio.sleep(delay)
            state.phase = LoopPhase.WAIT_EXTERNAL
            try:
                poll = getattr(self.a1_tool, "poll", None)
                if poll is None:
                    raise RuntimeError("execution backend does not support polling")
                result = await self._poll_external_task_once(
                    poll,
                    state=state,
                    request=replace(request, request_id=pending.request_id),
                    pending=pending,
                )
            except Exception as exc:
                pending.consecutive_poll_errors += 1
                pending.last_poll_error = f"{type(exc).__name__}: {exc}"
                pending.rpc_status = "error"
                result = A1TaskResult(
                    success=False,
                    result_status="task_pending",
                    task_metadata=dict(pending.task_metadata),
                )
            else:
                previous_metadata = dict(pending.task_metadata)
                incoming_metadata = (
                    dict(result.task_metadata)
                    if isinstance(result.task_metadata, dict)
                    else {}
                )
                # Do not carry a stale queued/running status into a terminal
                # workflow result that intentionally has no provider status.
                if result.result_status != "task_pending" and "status" not in incoming_metadata:
                    incoming_metadata["status"] = (
                        "succeeded" if result.success else "failed"
                    )
                previous_stage = previous_metadata.get("domain_workflow_stage")
                incoming_stage = incoming_metadata.get("domain_workflow_stage")
                previous_phase = str(
                    previous_stage.get("phase", "")
                    if isinstance(previous_stage, dict)
                    else ""
                ).strip()
                incoming_phase = str(
                    incoming_stage.get("phase", "")
                    if isinstance(incoming_stage, dict)
                    else ""
                ).strip()
                is_stage_transition = (
                    result.result_status == "task_pending"
                    and bool(previous_phase)
                    and bool(incoming_phase)
                    and previous_phase != incoming_phase
                )
                if is_stage_transition:
                    previous_external_task = {
                        "phase": previous_phase,
                        "task_id": pending.task_id,
                        "request_id": pending.request_id,
                        "gateway_tool": pending.gateway_tool,
                        "status": "succeeded",
                    }
                    metadata = dict(incoming_metadata)
                    route = metadata.get("omniagent_route")
                    previous_route = previous_metadata.get("omniagent_route")
                    if not isinstance(route, dict) and isinstance(previous_route, dict):
                        metadata["omniagent_route"] = previous_route
                    metadata.pop("workflow_resume", None)
                    metadata["previous_external_task"] = previous_external_task

                    next_task_id = str(metadata.get("task_id") or "").strip()
                    if not next_task_id:
                        reason = (
                            "TASK_RESUME_FAILED: pending external stage transition "
                            f"{previous_phase} -> {incoming_phase} is missing task_id"
                        )
                        result.success = False
                        result.result_status = "task_resume_failed"
                        result.errors = [*result.errors, reason]
                        metadata["status"] = "failed"
                    else:
                        next_request_id = str(
                            metadata.get("request_id")
                            or (
                                incoming_stage.get("request_id")
                                if isinstance(incoming_stage, dict)
                                else ""
                            )
                            or pending.request_id
                        ).strip()
                        next_gateway_tool = str(
                            metadata.get("gateway_tool")
                            or (
                                "call_biomni"
                                if incoming_phase == "a1"
                                else "biomni_invoke_tool"
                            )
                        ).strip()
                        next_status = normalize_task_status(
                            metadata.get("status")
                        ) or "submitted"
                        transition_time = time()
                        pending.task_id = next_task_id
                        pending.request_id = next_request_id
                        pending.gateway_tool = next_gateway_tool
                        pending.status = next_status
                        pending.remote_status = next_status
                        pending.rpc_status = "stage_changed"
                        pending.consecutive_poll_errors = 0
                        pending.last_poll_error = ""
                        pending.unknown_reason = ""
                        pending.next_poll_at = (
                            transition_time + self._task_poll_interval_seconds()
                        )
                        pending.deadline_at = (
                            transition_time + self._task_timeout_seconds()
                        )
                        metadata.update(
                            {
                                "task_id": pending.task_id,
                                "request_id": pending.request_id,
                                "gateway_tool": pending.gateway_tool,
                                "status": pending.status,
                                "next_poll_at": pending.next_poll_at,
                                "deadline_at": pending.deadline_at,
                            }
                        )
                        stage_transition = {
                            "iteration": pending.iteration,
                            "step_id": pending.step_id,
                            "from_phase": previous_phase,
                            "to_phase": incoming_phase,
                            "previous_task_id": previous_external_task["task_id"],
                            "previous_request_id": previous_external_task["request_id"],
                            "task_id": pending.task_id,
                            "request_id": pending.request_id,
                            "gateway_tool": pending.gateway_tool,
                            "remote_status": pending.status,
                        }
                else:
                    metadata = previous_metadata
                    metadata.update(incoming_metadata)
                if result.result_status != "task_pending":
                    metadata.pop("workflow_resume", None)
                result.task_metadata = metadata
                pending.task_metadata = metadata
                if result.result_status == "task_pending" and stage_transition is None:
                    workflow_resume = metadata.get("workflow_resume")
                    child_tasks = (
                        workflow_resume.get("child_tasks", [])
                        if isinstance(workflow_resume, dict)
                        else []
                    )
                    if isinstance(child_tasks, list):
                        waiting_children = [
                            item
                            for item in child_tasks
                            if isinstance(item, dict)
                            and str(item.get("task_id") or "").strip()
                            and normalize_task_status(item.get("status"))
                            in {"submitted", "queued", "running", "retry_wait"}
                        ]
                        if waiting_children:
                            child = waiting_children[-1]
                            child_task_id = str(child.get("task_id") or "").strip()
                            if child_task_id and child_task_id != pending.task_id:
                                pending.task_metadata["parent_task_id"] = pending.task_id
                                pending.task_id = child_task_id
                                pending.request_id = str(
                                    child.get("request_id") or pending.request_id
                                ).strip()
                                pending.gateway_tool = str(
                                    child.get("gateway_tool") or pending.gateway_tool
                                ).strip()
                                pending.status = normalize_task_status(
                                    child.get("status")
                                ) or "submitted"
                                pending.remote_status = pending.status
                                pending.rpc_status = "workflow_child_submitted"
                                pending.task_metadata.update(
                                    {
                                        "task_id": pending.task_id,
                                        "request_id": pending.request_id,
                                        "gateway_tool": pending.gateway_tool,
                                        "status": pending.status,
                                    }
                                )
                poll_error = str(metadata.get("last_poll_error", "")).strip()
                if poll_error:
                    pending.consecutive_poll_errors += 1
                    pending.last_poll_error = poll_error
                    pending.rpc_status = "error"
                else:
                    pending.consecutive_poll_errors = 0
                    pending.last_poll_error = ""
                    pending.rpc_status = "response_received"

            remote_status = self._external_task_status(result)
            # A completed outer task may still have a pending child in a bounded
            # MCP workflow. The normalized result status is authoritative for
            # the workflow, not just the outer Biomni task status.
            workflow_pending = (
                result.result_status == "task_pending"
                and isinstance(result.task_metadata, dict)
                and isinstance(result.task_metadata.get("workflow_resume"), dict)
            )
            if workflow_pending and remote_status not in self._NON_TERMINAL_EXTERNAL_STATUSES:
                remote_status = "running"
            is_waiting = (
                result.result_status == "task_pending"
                or remote_status in self._NON_TERMINAL_EXTERNAL_STATUSES
            )
            if not is_waiting:
                pending.remote_status = remote_status
                pending.rpc_status = "terminal"
                self._update_pending_action(
                    state,
                    pending,
                    self._action_status_for_result(result, terminal=True),
                    result.errors[-1] if result.errors else "",
                    result=result,
                )
                state.pending_execution = None
                self._save_pending_snapshot(state)
                if remote_status in {"manual_review", "dead_letter"}:
                    state.status = RunStatus.NEEDS_REVIEW
                    state.phase = LoopPhase.REVIEW
                    state.finish_reason = "external_task_requires_review"
                    self._emit(
                        "external_task_requires_review",
                        {
                            "iteration": pending.iteration,
                            "step_id": pending.step_id,
                            "task_id": pending.task_id,
                            "request_id": pending.request_id,
                            "remote_status": remote_status,
                        },
                    )
                    self._save_checkpoint(state, "external_task_requires_review")
                    return result
                self._emit(
                    "external_task_completed",
                    {
                        "iteration": pending.iteration,
                        "step_id": pending.step_id,
                        "task_id": pending.task_id,
                        "request_id": pending.request_id,
                        "remote_status": remote_status or None,
                        "result_status": result.result_status,
                        "success": result.success,
                    },
                )
                self._save_checkpoint(state, "external_task_completed")
                return result

            if pending.consecutive_poll_errors >= 3:
                reason = (
                    "TASK_UNKNOWN: Biomni task status polling failed after "
                    f"{pending.consecutive_poll_errors} attempts; task identity is retained: "
                    f"{pending.task_id}; last error: {pending.last_poll_error}"
                )
                pending.status = ActionStatus.UNKNOWN.value
                pending.unknown_reason = reason
                pending.rpc_status = "unknown"
                pending.next_poll_at = time() + self._task_poll_interval_seconds()
                pending.task_metadata.update(
                    {
                        "status": ActionStatus.UNKNOWN.value,
                        "last_poll_error": pending.last_poll_error,
                        "next_poll_at": pending.next_poll_at,
                        "deadline_at": pending.deadline_at,
                    }
                )
                state.pending_execution = pending
                self._update_pending_action(state, pending, ActionStatus.UNKNOWN, reason)
                self._save_pending_snapshot(state)
                state.status = RunStatus.NEEDS_REVIEW
                state.phase = LoopPhase.REVIEW
                state.finish_reason = "external_task_unknown"
                self._emit(
                    "external_task_unknown",
                    {
                        "iteration": pending.iteration,
                        "step_id": pending.step_id,
                        "task_id": pending.task_id,
                        "request_id": pending.request_id,
                        "consecutive_poll_errors": pending.consecutive_poll_errors,
                        "last_poll_error": pending.last_poll_error,
                    },
                )
                self._save_checkpoint(state, "external_task_unknown")
                return A1TaskResult(
                    success=False,
                    result_status="poll_transient_error",
                    errors=[reason],
                    task_metadata=dict(pending.task_metadata)
                    | {"lifecycle_status": "unknown"},
                )

            remote_status = str(
                result.task_metadata.get("status", "")
                if isinstance(result.task_metadata, dict)
                else ""
            ).strip().lower()
            pending.remote_status = remote_status
            pending.status = remote_status or pending.status or "submitted"
            pending.next_poll_at = time() + self._next_poll_delay_seconds(
                pending, remote_status
            )
            pending.task_metadata.update(
                {
                    "next_poll_at": pending.next_poll_at,
                    "deadline_at": pending.deadline_at,
                    "remote_status": pending.remote_status,
                    "rpc_status": pending.rpc_status,
                }
            )
            state.pending_execution = pending
            self._update_pending_action(
                state,
                pending,
                self._pending_status(pending.status),
                pending.last_poll_error,
            )
            self._save_pending_snapshot(state)
            state_changed = pending.status != previous_status
            new_error = bool(pending.last_poll_error) and (
                pending.last_poll_error != previous_error
            )
            if stage_transition is not None:
                self._emit("external_task_stage_changed", stage_transition)
                self._save_checkpoint(state, "external_task_stage_changed")
            elif state_changed or new_error:
                self._emit(
                    "external_task_poll_error" if new_error else "external_task_waiting",
                    {
                        "iteration": pending.iteration,
                        "step_id": pending.step_id,
                        "task_id": pending.task_id,
                        "request_id": pending.request_id,
                        "remote_status": pending.status,
                        "consecutive_poll_errors": pending.consecutive_poll_errors,
                        "last_poll_error": pending.last_poll_error or None,
                    },
                )
                self._save_checkpoint(
                    state,
                    "external_task_poll_error" if new_error else "external_task_waiting",
                )

    @classmethod
    def _external_task_status(cls, result: A1TaskResult) -> str:
        """Return the normalized remote lifecycle status from one poll result.

        Biomni's task status is authoritative for async work.  In particular,
        ``queued`` and ``running`` must keep the action in WAIT_EXTERNAL even if
        an adapter reports a non-pending ``result_status``.
        """
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        return normalize_task_status(metadata.get("status"))

    async def _poll_external_task_once(
        self,
        poll: Any,
        *,
        state: ResearchState,
        request: A1TaskRequest,
        pending: PendingExecution,
    ) -> A1TaskResult:
        """Run one status poll under a runtime-owned deadline.

        A backend is responsible for cancelling its own transport task, but the
        Harness still needs an outer bound so a misbehaving client cannot leave
        a durable execution permanently in WAIT_EXTERNAL.
        """
        timeout_seconds = self._poll_request_timeout_seconds()
        event_payload = {
            "iteration": pending.iteration,
            "step_id": pending.step_id,
            "task_id": pending.task_id,
            "request_id": pending.request_id,
            "timeout_seconds": timeout_seconds,
        }
        self._emit("external_task_poll_started", event_payload)
        poll_task = asyncio.ensure_future(poll(request, dict(pending.task_metadata)))
        try:
            completed, _ = await asyncio.wait(
                {poll_task}, timeout=timeout_seconds
            )
        except BaseException:
            if not poll_task.done():
                poll_task.cancel()
            self._emit(
                "external_task_poll_finished",
                event_payload | {"outcome": "cancelled"},
            )
            raise
        if not completed:
            poll_task.cancel()
            await asyncio.sleep(0)
            self._emit(
                "external_task_poll_finished",
                event_payload | {"outcome": "timed_out"},
            )
            raise TimeoutError(
                "OmniAgent poll deadline exceeded for Biomni task "
                f"{pending.task_id} after {timeout_seconds:g} seconds"
            )
        try:
            result = poll_task.result()
        except BaseException as exc:
            self._emit(
                "external_task_poll_finished",
                event_payload
                | {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        self._emit(
            "external_task_poll_finished",
            event_payload
            | {
                "outcome": "completed",
                "result_status": result.result_status,
                "success": result.success,
                "remote_status": str(
                    result.task_metadata.get("status", "")
                    if isinstance(result.task_metadata, dict)
                    else ""
                ).strip().lower()
                or None,
            },
        )
        return result

    def _create_pending_execution(
        self,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> PendingExecution:
        metadata = dict(result.task_metadata) if isinstance(result.task_metadata, dict) else {}
        task_id = str(metadata.get("task_id") or "").strip()
        request_id = str(metadata.get("request_id") or "").strip()
        gateway_tool = str(metadata.get("gateway_tool") or "").strip()
        workflow_resume = metadata.get("workflow_resume")
        if result.result_status == "task_pending" and isinstance(workflow_resume, dict):
            # The outer task can be succeeded while the workflow has already
            # submitted its next child task. Resume must poll that child, not
            # repeatedly poll the completed outer task.
            child_tasks = workflow_resume.get("child_tasks", [])
            if isinstance(child_tasks, list):
                waiting = [
                    item
                    for item in child_tasks
                    if isinstance(item, dict)
                    and str(item.get("task_id") or "").strip()
                    and normalize_task_status(item.get("status"))
                    in {"submitted", "queued", "running", "retry_wait"}
                ]
                if waiting:
                    child = waiting[-1]
                    metadata.setdefault("parent_task_id", task_id)
                    task_id = str(child.get("task_id") or task_id).strip()
                    request_id = str(child.get("request_id") or request_id).strip()
                    gateway_tool = str(child.get("gateway_tool") or gateway_tool).strip()
                    metadata["task_id"] = task_id
                    metadata["request_id"] = request_id
                    if gateway_tool:
                        metadata["gateway_tool"] = gateway_tool
        if not task_id:
            raise ValueError("TASK_PENDING response is missing Biomni task_id")
        route = metadata.get("omniagent_route", {})
        route = route if isinstance(route, dict) else {}
        now = time()
        timeout = self._task_timeout_seconds()
        pending_status = normalize_task_status(metadata.get("status")) or "submitted"
        if result.result_status == "task_pending" and pending_status not in {
            "submitted",
            "queued",
            "running",
            "retry_wait",
        }:
            pending_status = "submitted"
            metadata["status"] = pending_status
        return PendingExecution(
            step_id=request.step.step_id,
            iteration=request.iteration,
            task_id=task_id,
            request_id=str(
                request_id
                or request.request_id
                or f"{request.run_id}:{request.step.step_id}"
            ),
            gateway_tool=str(
                gateway_tool
                or ("call_biomni" if route.get("backend") == "a1" else "biomni_invoke_tool")
            ),
            backend=str(route.get("backend") or "unknown"),
            status=pending_status,
            next_poll_at=now + self._task_poll_interval_seconds(),
            deadline_at=now + timeout if timeout > 0 else 0.0,
            task_metadata=metadata,
            remote_status=pending_status,
            rpc_status="submitted",
            wait_started_at=now,
        )

    def _task_poll_interval_seconds(self) -> float:
        return max(0.0, float(getattr(self.a1_tool, "task_poll_interval_seconds", 2.0)))

    def _next_poll_delay_seconds(
        self,
        pending: PendingExecution,
        remote_status: str,
    ) -> float:
        """Back off non-terminal task polls while preserving lifecycle responsiveness."""
        base_delay = self._task_poll_interval_seconds()
        if remote_status not in {"queued", "running"} or base_delay <= 0:
            return base_delay
        poll_state = pending.task_metadata.get("omniagent_poll", {})
        poll_state = poll_state if isinstance(poll_state, dict) else {}
        previous_status = str(poll_state.get("last_status", "")).strip().lower()
        try:
            status_polls = int(poll_state.get("status_polls", 0))
        except (TypeError, ValueError):
            status_polls = 0
        status_polls = status_polls + 1 if previous_status == remote_status else 1
        minimum = 5.0 if remote_status == "running" else base_delay
        delay = min(
            15.0,
            max(minimum, base_delay * (2 ** min(status_polls - 1, 3))),
        )
        pending.task_metadata["omniagent_poll"] = {
            "last_status": remote_status,
            "status_polls": status_polls,
            "next_poll_delay_seconds": delay,
        }
        return delay

    def _poll_request_timeout_seconds(self) -> float:
        configured = getattr(self.a1_tool, "poll_request_timeout_seconds", None)
        if configured is None:
            client = getattr(self.a1_tool, "client", None)
            configured = getattr(client, "poll_request_timeout_seconds", None)
        try:
            timeout = float(configured)
        except (TypeError, ValueError):
            timeout = 30.0
        # A small grace period lets the Gateway return its own normalized
        # timeout, while still containing a stalled adapter call.
        return max(0.1, timeout + 2.0)

    def _task_timeout_seconds(self) -> float:
        return max(0.0, float(getattr(self.a1_tool, "task_timeout_seconds", 900.0)))

    def _save_pending_snapshot(self, state: ResearchState) -> None:
        if self.persistence is not None:
            self.persistence.save_pending_snapshot(state)

    def _load_reusable_mcp_result(
        self,
        binding: Any,
    ) -> tuple[A1TaskResult, str, str] | None:
        if self.persistence is None or binding is None:
            return None
        backend = getattr(getattr(binding, "backend", None), "value", "")
        if backend != "mcp":
            return None
        return self.persistence.load_reusable_mcp_result(
            str(getattr(binding, "execution_signature", "")),
            str(getattr(binding, "catalog_revision", "")),
        )

    def _normalize_execution_result(
        self,
        state: ResearchState,
        result: A1TaskResult,
    ) -> None:
        """Keep external result data separate from the Harness-owned final artifact."""
        output_config = state.task_manifest.get("output_config", {})
        output_name = str(
            output_config.get("file_path", "")
            if isinstance(output_config, dict)
            else ""
        ).strip()
        if not output_name:
            return
        workspace = Path(state.workspace).resolve()
        final_path = (workspace / output_name).resolve()
        retained: list[str] = []
        ignored: list[str] = []
        for artifact in result.artifacts:
            try:
                artifact_path = Path(artifact).resolve()
            except (OSError, ValueError):
                retained.append(artifact)
                continue
            if artifact_path == final_path:
                ignored.append(artifact)
            else:
                retained.append(artifact)
        if not ignored:
            return
        result.artifacts = retained
        metadata = (
            dict(result.task_metadata)
            if isinstance(result.task_metadata, dict)
            else {}
        )
        manifest = metadata.get("artifact_manifest", [])
        if isinstance(manifest, list):
            metadata["artifact_manifest"] = [
                item
                for item in manifest
                if not (
                    isinstance(item, dict)
                    and str(item.get("path", ""))
                    and Path(str(item["path"])).resolve() == final_path
                )
            ]
        metadata["external_final_artifacts_ignored"] = ignored
        result.task_metadata = metadata
        self._emit(
            "external_final_artifact_ignored",
            {
                "path": str(final_path),
                "artifact_count": len(ignored),
                "reason": "Harness owns final output materialization",
            },
        )

    async def _plan_with_contract_retry(self, state: ResearchState) -> ExperimentPlan:
        """Retry a Planner-only contract violation without dispatching an external tool."""
        retries = self.policy.max_plan_contract_retries
        for attempt in range(retries + 1):
            try:
                plan = await self.planner.plan(state)
            except (PlannerContractError, FeedbackNotConsumedError) as exc:
                self._emit(
                    "plan_contract_rejected",
                    {
                        "iteration": len(state.iterations),
                        "retry_index": attempt,
                        "max_retries": retries,
                        "reason": str(exc),
                    },
                )
                if attempt >= retries:
                    raise
                feedback = "Planner contract feedback: " + str(exc)
                if feedback not in state.constraints:
                    state.constraints.append(feedback)
                self._save_checkpoint(state, "plan_contract_rejected")
                continue
            try:
                self._validate_plan(state, plan)
            except (
                FinalArtifactPlanError,
                PlannerContractError,
                FeedbackNotConsumedError,
            ) as exc:
                self._emit(
                    "plan_contract_rejected",
                    {
                        "iteration": len(state.iterations),
                        "retry_index": attempt,
                        "max_retries": retries,
                        "reason": str(exc),
                    },
                )
                if attempt >= retries:
                    raise
                feedback = (
                    "Planner contract feedback: "
                    + str(exc)
                    + " Plan only the missing scientific evidence; the Harness owns "
                    "capability binding and final materialization."
                )
                if feedback not in state.constraints:
                    state.constraints.append(feedback)
                self._save_checkpoint(state, "plan_contract_rejected")
                continue
            if len(plan.steps) > self.policy.max_steps_per_iteration:
                error = PlannerContractError(
                    f"Plan {plan.plan_id} returned {len(plan.steps)} steps, but the "
                    f"per-iteration limit is {self.policy.max_steps_per_iteration}"
                )
                self._emit(
                    "plan_contract_rejected",
                    {
                        "iteration": len(state.iterations),
                        "retry_index": attempt,
                        "max_retries": retries,
                        "reason": str(error),
                    },
                )
                if attempt >= retries:
                    raise error
                state.constraints.append("Planner contract feedback: " + str(error))
                self._save_checkpoint(state, "plan_contract_rejected")
                continue
            return plan
        raise RuntimeError("unreachable planner contract retry state")

    def _finalize_analysis_output(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        analysis: Any,
        result_verifier: TaskResultVerifier,
        allowed_paths: list[str],
    ) -> None:
        """Materialize an Analyzer synthesis only after binding it to admitted evidence."""
        output_config = state.task_manifest.get("output_config", {})
        output_name = (
            str(output_config.get("file_path", "")).strip()
            if isinstance(output_config, dict)
            else ""
        )
        payload = getattr(analysis, "final_output", None)
        if not output_name or payload is None:
            return

        supporting_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in getattr(analysis, "supporting_evidence_ids", [])
                if str(item).strip()
            )
        )
        envelope_answer = self._matching_envelope_answer(
            state,
            payload,
            supporting_ids=supporting_ids,
        )
        merged_payload = merge_envelope_answer(payload, envelope_answer)
        if merged_payload != payload:
            payload = merged_payload
            analysis.final_output = payload
            self._emit(
                "final_output_envelope_merged",
                {
                    "iteration": len(state.iterations),
                    "source": "admitted_execution_envelope",
                    "field": "answer",
                },
            )

        missing_ids = [
            item for item in supporting_ids if item not in state.scientific_state.evidence
        ]
        if not isinstance(payload, dict) or not supporting_ids or missing_ids:
            reasons: list[str] = []
            if not isinstance(payload, dict):
                reasons.append("Analyzer final_output must be a JSON object.")
            if not supporting_ids:
                reasons.append(
                    "Analyzer final_output requires at least one supporting evidence ID."
                )
            if missing_ids:
                reasons.append(
                    "Analyzer final_output references evidence not admitted to scientific "
                    f"state: {', '.join(missing_ids)}"
                )
            self._emit(
                "analysis_final_output_rejected",
                {
                    "iteration": len(state.iterations),
                    "reason": " ".join(reasons),
                },
            )
            return

        semantic_contract = infer_answer_semantic_contract(state.goal)
        if semantic_contract is not None:
            semantic_check = validate_final_answer_semantics(
                semantic_contract,
                payload,
                state.scientific_state.claims.values(),
                state.scientific_state.evidence,
                supporting_ids,
            )
            if semantic_check.get("passed") is not True:
                synthesized = synthesize_grounded_final_output(
                    semantic_contract,
                    payload,
                    state.goal,
                    supporting_ids,
                    state.scientific_state.evidence,
                    analysis,
                )
                if synthesized is not None:
                    payload = synthesized
                    analysis.final_output = synthesized
                    semantic_check = validate_final_answer_semantics(
                        semantic_contract,
                        payload,
                        state.scientific_state.claims.values(),
                        state.scientific_state.evidence,
                        supporting_ids,
                    )
                    if semantic_check.get("passed") is True:
                        self._emit(
                            "analysis_final_output_synthesized",
                            {
                                "iteration": len(state.iterations),
                                "reason": "deterministic semantic evidence derived from admitted structured record",
                                "supporting_evidence_ids": semantic_check.get(
                                    "evidence_ids", []
                                ),
                            },
                        )
            if semantic_check.get("passed") is not True:
                self._emit(
                    "analysis_final_output_rejected",
                    {
                        "iteration": len(state.iterations),
                        "reason": "; ".join(
                            str(item) for item in semantic_check.get("blockers", [])
                        )
                        or "final output failed semantic evidence validation",
                    },
                )
                self._save_checkpoint(state, "analysis_final_output_rejected")
                return

        request = A1TaskRequest(
            run_id=state.run_id,
            iteration=len(state.iterations),
            research_goal=state.goal,
            hypothesis=plan.hypothesis,
            step=self._harness_finalization_step(),
            global_constraints=state.constraints,
            prior_observations=[],
            prior_evaluations=[],
            allowed_paths=list(allowed_paths),
            state_version=state.scientific_state.state_version,
            route_retry_state=state.route_retry_state,
        )
        result = A1TaskResult(
            success=True,
            output=payload,
            task_metadata={
                "harness_finalization": True,
                "supporting_evidence_ids": supporting_ids,
                "omniagent_route": {
                    "backend": "harness",
                    "execution_invoked": False,
                    "reason": "analysis_grounded_final_output",
                },
            },
        )
        try:
            self._validate_output_contract(state, result)
        except Exception as exc:
            result.success = False
            result.result_status = "output_contract_failed"
            result.errors.append(
                f"OUTPUT_CONTRACT_FAILED: {type(exc).__name__}: {exc}"
            )

        state.phase = LoopPhase.VERIFY
        verification = result_verifier.verify(state.scientific_state, request, result)
        state.phase = LoopPhase.REDUCE
        self.state_reducer.apply(state.scientific_state, verification)
        self._emit(
            "analysis_final_output_verified",
            {
                "iteration": len(state.iterations),
                "accepted": verification.accepted,
                "reason": verification.reason,
                "verifier_id": verification.verifier_id,
                "transition_id": verification.transition_id,
                "supporting_evidence_ids": supporting_ids,
            },
        )
        if not verification.accepted:
            self._emit(
                "analysis_final_output_rejected",
                {
                    "iteration": len(state.iterations),
                    "reason": verification.reason,
                },
            )
            self._save_checkpoint(state, "analysis_final_output_rejected")
            return

        state.phase = LoopPhase.MATERIALIZE
        try:
            self._materialize_output(state, result)
        except Exception as exc:
            self._emit(
                "analysis_final_output_rejected",
                {
                    "iteration": len(state.iterations),
                    "reason": f"OUTPUT_CONTRACT_FAILED: {type(exc).__name__}: {exc}",
                },
            )
            self._save_checkpoint(state, "analysis_final_output_rejected")
            return
        self._emit(
            "analysis_final_output_materialized",
            {
                "iteration": len(state.iterations),
                "path": str((Path(state.workspace).resolve() / output_name).resolve()),
                "supporting_evidence_ids": supporting_ids,
            },
        )
        self._save_checkpoint(state, "analysis_final_output_materialized")

    @staticmethod
    def _output_matches_final_payload(final_payload: Any, output: Any) -> bool:
        if not isinstance(final_payload, dict) or not isinstance(output, dict):
            return False
        return bool(set(final_payload).intersection(output))

    @classmethod
    def _matching_envelope_answer(
        cls,
        state: ResearchState,
        final_payload: Any,
        *,
        supporting_ids: list[str] | tuple[str, ...] = (),
    ) -> str:
        """Find an admitted answer belonging to the result being finalized."""
        evidence = getattr(state.scientific_state, "evidence", {})
        for evidence_id in supporting_ids:
            record = evidence.get(evidence_id)
            payload = getattr(record, "payload", None)
            if isinstance(payload, dict) and payload.get("answer"):
                return str(payload["answer"])
        for result in reversed(getattr(state, "active_executions", [])):
            if (
                result.answer.strip()
                and cls._output_matches_final_payload(final_payload, result.output)
            ):
                return result.answer
        for record in reversed(list(evidence.values())):
            payload = getattr(record, "payload", None)
            if not isinstance(payload, dict):
                continue
            answer = payload.get("answer")
            output = payload.get("output")
            if (
                answer
                and cls._output_matches_final_payload(final_payload, output)
            ):
                return str(answer)
        return ""

    @staticmethod
    def _harness_finalization_step() -> ExperimentStep:
        return ExperimentStep(
            step_id="harness_finalization",
            objective="Materialize a grounded final output from admitted evidence.",
            inputs={
                "workflow_phase": "synthesis",
                "execution_backend": "harness",
            },
        )

    def _validate_output_contract(self, state: ResearchState, result: Any) -> None:
        """Check a final-output contract before scientific evidence is admitted."""
        output_config = state.task_manifest.get("output_config", {})
        output_name = str(output_config.get("file_path", "")).strip()
        if output_config.get("format", "json") != "json" or not output_name:
            return
        payload = getattr(result, "output", None)
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise ValueError("JSON output contract requires an object payload")
        schema = output_config.get("schema")
        if isinstance(schema, dict):
            errors = validate_schema_instance(payload, schema, strict_objects=True)
            if errors:
                raise ValueError("; ".join(errors[:3]))
        workspace = Path(state.workspace).resolve()
        output = (workspace / output_name).resolve()
        if not output.is_relative_to(workspace):
            raise ValueError("Configured output path escapes the isolated workspace")

    def _materialize_output(self, state: ResearchState, result: Any) -> None:
        """Atomically persist a structured final output owned by the Harness."""
        output_config = state.task_manifest.get("output_config", {})
        output_name = str(output_config.get("file_path", "")).strip()
        if output_config.get("format", "json") != "json" or not output_name:
            return
        payload = getattr(result, "output", None)
        if payload is None:
            return
        self._validate_output_contract(state, result)
        schema = output_config.get("schema")
        workspace = Path(state.workspace).resolve()
        output = (workspace / output_name).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        if output.exists() and (output.is_symlink() or not output.is_file()):
            raise ValueError("final output path is not a regular file")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(output)
        artifact_declaration: dict[str, Any] = {
            "path": str(output),
            "media_type": "application/json",
        }
        if isinstance(schema, dict):
            artifact_declaration["content_schema"] = schema
        verified = verify_artifacts([artifact_declaration], [str(workspace)])
        state.final_output_materialized = True
        if str(output) not in result.artifacts:
            result.artifacts.append(str(output))
        metadata = (
            dict(result.task_metadata)
            if isinstance(result.task_metadata, dict)
            else {}
        )
        manifest = metadata.get("artifact_manifest", [])
        manifest = manifest if isinstance(manifest, list) else []
        manifest_by_path = {
            str(item.get("path")): item for item in manifest if isinstance(item, dict)
        }
        for item in verified:
            manifest_by_path[item.path] = item.to_dict()
        metadata["artifact_manifest"] = list(manifest_by_path.values())
        result.task_metadata = metadata
        route = (
            result.task_metadata.get("omniagent_route", {})
            if isinstance(result.task_metadata, dict)
            else {}
        )
        self._emit(
            "output_materialized",
            {
                "path": str(output),
                "format": "json",
                "source": f"{route.get('backend', 'execution')}_structured_output",
            },
        )

    @staticmethod
    def _record_route_retry_state(state: ResearchState, result: A1TaskResult) -> None:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        route = metadata.get("omniagent_route", {})
        retry = route.get("retry_state", {}) if isinstance(route, dict) else {}
        if not isinstance(retry, dict):
            return
        signature = str(
            retry.get("retry_key")
            or retry.get("execution_signature")
            or retry.get("route_signature", "")
        ).strip()
        if not signature:
            return
        try:
            failure_count = int(retry.get("failure_count", 0))
        except (TypeError, ValueError):
            failure_count = 0
        if failure_count > 0:
            state.route_retry_state[signature] = dict(retry)
        else:
            state.route_retry_state.pop(signature, None)

    @staticmethod
    def _mark_external_failure_retry_state(result: A1TaskResult) -> None:
        """Prevent automatic resubmission after a durable task wait has failed."""
        if result.result_status not in {
            "task_timed_out",
            "task_wait_timed_out",
            "task_unknown",
            "poll_transient_error",
        }:
            return
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        route = metadata.get("omniagent_route", {})
        if not isinstance(route, dict):
            return
        retry = route.get("retry_state", {})
        if not isinstance(retry, dict):
            return
        signature = str(
            retry.get("retry_key")
            or retry.get("execution_signature")
            or retry.get("route_signature", "")
        ).strip()
        if not signature:
            return
        retry = dict(retry)
        try:
            prior_failures = int(retry.get("failure_count", 0))
        except (TypeError, ValueError):
            prior_failures = 0
        retry["failure_count"] = max(1, prior_failures)
        retry["failure_limit"] = 1
        retry["retry_key"] = signature
        retry["result_status"] = result.result_status
        retry["last_error"] = result.errors[-1] if result.errors else result.result_status
        route = dict(route)
        route["retry_state"] = retry
        metadata = dict(metadata)
        metadata["omniagent_route"] = route
        result.task_metadata = metadata

    @staticmethod
    def _execution_used_a1(result: A1TaskResult) -> bool:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        route = metadata.get("omniagent_route")
        if not isinstance(route, dict):
            return True
        return (
            route.get("backend") == "a1"
            and route.get("execution_invoked", True) is not False
        )

    @staticmethod
    def _has_verified_execution_progress(executions: list[A1TaskResult]) -> bool:
        """Only admitted verifier transitions can establish scientific progress."""
        for result in executions:
            metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
            verification = metadata.get("omniagent_verification")
            if isinstance(verification, dict) and verification.get("accepted") is True:
                return True
        return False

    @staticmethod
    def _validate_plan(state: ResearchState, plan: ExperimentPlan) -> None:
        if not plan.steps:
            raise PlannerContractError("Planner returned no executable steps")
        step_ids = [step.step_id for step in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlannerContractError("Planner returned duplicate step IDs")
        output_config = state.task_manifest.get("output_config", {})
        output_name = (
            str(output_config.get("file_path", "")).strip()
            if isinstance(output_config, dict)
            else ""
        )
        for step in plan.steps:
            execution_owned = [
                key
                for key in ("execution_backend", "tool_name", "arguments")
                if key in step.inputs
            ]
            if execution_owned:
                raise PlannerContractError(
                    "Planner cannot provide execution-owned field(s): "
                    + ", ".join(execution_owned)
                )
            if plan.planner_contract_version and not isinstance(
                step.inputs.get("semantic_intent"), dict
            ):
                raise PlannerContractError(
                    "Planner contract v2 requires semantic_intent on every step"
                )
            if plan.planner_contract_version and not str(
                step.inputs.get("tool_query", "")
            ).strip():
                raise PlannerContractError(
                    "Planner contract v2 requires tool_query on every step"
                )
            if "semantic_intent" in step.inputs and SemanticCapabilityIntent.from_inputs(
                step.inputs
            ) is None:
                raise PlannerContractError(
                    "Planner semantic_intent is malformed or contains an unsupported enum"
                )
            if output_name and ClosedLoopRuntime._step_references_final_output(
                state, step, output_name
            ):
                raise FinalArtifactPlanError(
                    "Planner step references the configured final artifact. The Harness "
                    "owns final output materialization from grounded analysis."
                )
        critique = state.last_critique
        if critique and critique.requires_consumption:
            if critique.feedback_id not in plan.feedback_ids_consumed:
                raise FeedbackNotConsumedError(
                    f"Plan {plan.plan_id} did not consume required feedback "
                    f"{critique.feedback_id}"
                )
            evaluation = state.last_evaluation
            if (
                evaluation is None
                or evaluation.evaluation_id not in plan.evidence_refs_consumed
            ):
                raise FeedbackNotConsumedError(
                    f"Plan {plan.plan_id} did not cite the required evaluation "
                    f"{evaluation.evaluation_id if evaluation else '<missing>'}"
                )
            if not plan.adaptation_summary.strip():
                raise FeedbackNotConsumedError(
                    f"Plan {plan.plan_id} did not explain how feedback changed the workflow"
                )

    @staticmethod
    def _step_references_final_output(
        state: ResearchState,
        step: Any,
        output_name: str,
    ) -> bool:
        """Reject final-artifact references regardless of planner operation or side effect."""
        workspace_output = str(
            (Path(state.workspace).resolve() / output_name).resolve()
        )
        targets = {
            output_name.replace("\\", "/").casefold(),
            workspace_output.replace("\\", "/").casefold(),
            Path(output_name).name.casefold(),
        }

        def references(value: Any) -> bool:
            if isinstance(value, dict):
                return any(references(item) for item in value.values())
            if isinstance(value, list | tuple | set):
                return any(references(item) for item in value)
            if not isinstance(value, str):
                return False
            text = value.replace("\\", "/").casefold()
            return any(target and target in text for target in targets)

        return references(
            {
                "objective": step.objective,
                "expected_outputs": step.expected_outputs,
                "inputs": step.inputs,
            }
        )

    @staticmethod
    def _progress_signature(
        plan: ExperimentPlan,
        executions: list[Any],
        analysis: Any,
    ) -> str:
        """Hash material execution progress, excluding Planner rephrasing."""
        _ = (plan, analysis)
        execution_payload = []
        for result in executions:
            metadata = external_task_summary(result.task_metadata)
            route = metadata.get("omniagent_route")
            route = route if isinstance(route, dict) else {}
            tool_names = sorted(
                {
                    str(event.get("tool_name") or event.get("gateway_tool"))
                    for event in result.tool_trace
                    if isinstance(event, dict)
                    and (event.get("tool_name") or event.get("gateway_tool"))
                }
            )
            error_classes = sorted(
                {
                    str(error).split(":", 1)[0].strip().casefold()
                    for error in result.errors
                    if str(error).strip()
                }
            )
            execution_payload.append(
                {
                    "action_id": metadata.get("action_id"),
                    "idempotency_key": metadata.get("idempotency_key"),
                    "execution_signature": route.get("execution_signature"),
                    "capability": (
                        route.get("selected_capability")
                        or route.get("admitted_capability")
                    ),
                    "success": result.success,
                    "result_status": result.result_status,
                    "remote_status": metadata.get("status"),
                    "error_code": metadata.get("error_code"),
                    "error_classes": error_classes,
                    "tool_names": tool_names,
                    "metrics": result.metrics,
                    "material_output": material_result_leaves(
                        result.output, limit=24
                    ),
                }
            )
        encoded = json.dumps(
            execution_payload, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _complete(self, state: ResearchState, reason: str) -> None:
        blockers = self._completion_blockers(state)
        if blockers:
            self._emit(
                "completion_blocked",
                {"reason": reason, "blockers": blockers},
            )
            self._needs_review(state, "completion_blocked")
            return
        state.status = RunStatus.COMPLETED
        state.phase = LoopPhase.COMPLETE
        state.finish_reason = reason

    def _needs_review(self, state: ResearchState, reason: str) -> None:
        state.status = RunStatus.NEEDS_REVIEW
        state.phase = LoopPhase.REVIEW
        state.finish_reason = reason

    def _fail(self, state: ResearchState, reason: str) -> None:
        state.status = RunStatus.FAILED
        state.phase = LoopPhase.FAILED
        state.finish_reason = reason

    def _finish_at_limit(
        self,
        state: ResearchState,
        reason: str,
        *,
        finalization: FinalizationDecision | None = None,
    ) -> None:
        decision = finalization or self.finalization_gate.evaluate(
            state, state.last_evaluation
        )
        if decision.mode == "insufficient_evidence" and decision.eligible:
            self._needs_review(state, "insufficient_evidence")
            return
        lower_reason = str(reason).lower()
        if any(
            marker in lower_reason
            for marker in (
                "max_iterations",
                "budget",
                "timeout",
                "timed_out",
                "unknown",
                "plateau",
            )
        ):
            self._emit(
                "completion_blocked",
                {
                    "reason": reason,
                    "finalization": decision.to_dict(),
                    "status": RunStatus.NEEDS_REVIEW.value,
                },
            )
            self._needs_review(state, reason)
            return
        if decision.eligible and decision.objective_satisfied:
            self._complete(state, reason)
            return
        self._emit(
            "completion_blocked",
            {
                "reason": reason,
                "finalization": decision.to_dict(),
            },
        )
        self._needs_review(state, f"{reason}:finalization_blocked")

    @staticmethod
    def _completion_blockers(state: ResearchState) -> list[str]:
        blockers: list[str] = []
        if state.pending_execution is not None:
            blockers.append(
                "pending external task: " + state.pending_execution.task_id
            )
        ledger = getattr(state, "action_ledger", None)
        if ledger is not None:
            for action in ledger.unresolved():
                blockers.append(
                    f"unresolved action {action.action_id} ({action.status})"
                )
            for action in ledger.records.values():
                if action.status == ActionStatus.TIMED_OUT.value:
                    blockers.append(f"timed-out action {action.action_id}")
        return list(dict.fromkeys(blockers))

    @staticmethod
    def _event_text(value: Any, limit: int = 2000) -> str:
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    @classmethod
    def _task_event_summary(cls, metadata: Any) -> dict[str, Any]:
        return external_task_summary(metadata)

    @classmethod
    def _run_finished_summary(cls, state: ResearchState) -> dict[str, Any]:
        pending = state.pending_execution
        pending_summary: dict[str, Any] | None = None
        if pending is not None:
            pending_summary = cls._task_event_summary(
                dict(pending.task_metadata)
                | {
                    "task_id": pending.task_id,
                    "request_id": pending.request_id,
                    "status": pending.status,
                    "remote_status": pending.remote_status,
                    "rpc_status": pending.rpc_status,
                    "gateway_tool": pending.gateway_tool,
                }
            )
        scientific = state.scientific_state
        return {
            "run_id": state.run_id,
            "status": state.status.value,
            "phase": state.phase.value,
            "finish_reason": state.finish_reason,
            "score": state.best_score,
            "harness_status": state.status.value,
            "harness_score": state.best_score,
            "status_semantics": "harness_execution_only",
            "iteration_count": len(state.iterations),
            "a1_call_count": state.a1_call_count,
            "action_count": len(state.action_ledger.records),
            "evidence_count": len(scientific.evidence),
            "claim_count": len(scientific.claims),
            "attempt_count": len(scientific.attempts),
            "state_version": scientific.state_version,
            "final_output_materialized": state.final_output_materialized,
            "pending_external_task": pending_summary,
        }

    @classmethod
    def _execution_event_summary(cls, result: A1TaskResult) -> dict[str, Any]:
        return {
            "success": result.success,
            "result_status": result.result_status,
            "answer_chars": len(result.answer),
            "metrics": result.metrics,
            "errors": [cls._event_text(item) for item in result.errors[:8]],
            "artifacts": list(result.artifacts[:20]),
            "task": cls._task_event_summary(result.task_metadata),
        }

    def _save_checkpoint(self, state: ResearchState, safe_point: str) -> None:
        if self.persistence is None:
            return
        self.persistence.save_checkpoint(state, safe_point=safe_point)

    @staticmethod
    def _attempt_key(state: ResearchState, step_id: str) -> str:
        return ":".join(
            (
                state.run_id,
                str(len(state.iterations)),
                step_id,
                str(state.scientific_state.state_version),
            )
        )

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.persistence is not None:
            self.persistence.append_event(event, payload)
        self.event_sink(event, payload)
