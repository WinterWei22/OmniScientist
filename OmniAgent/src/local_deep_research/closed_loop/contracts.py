from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from .action_ledger import ActionLedger
from .scientific_state import CanonicalEntityRecord, ScientificState


class PlannerContractError(ValueError):
    """Raised when a planner response contains execution-owned fields."""


class Decision(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    RETRY = "retry"
    STOP = "stop"
    FAIL = "fail"


class LoopPhase(str, Enum):
    INITIALIZE = "initialize"
    PLAN = "plan"
    BIND = "bind"
    DISPATCH = "dispatch"
    WAIT_EXTERNAL = "wait_external"
    VERIFY = "verify"
    REDUCE = "reduce"
    MATERIALIZE = "materialize"
    FINALIZE = "finalize"
    REVIEW = "review"
    # Kept only to resume P0 checkpoints created before the constrained phases.
    EXECUTE = "execute"
    OBSERVE = "observe"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    DECIDE = "decide"
    REPLAN = "replan"
    COMPLETE = "complete"
    FAILED = "failed"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass(slots=True)
class ExperimentStep:
    step_id: str
    objective: str
    inputs: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentPlan:
    hypothesis: str
    rationale: str
    steps: list[ExperimentStep]
    plan_id: str = field(default_factory=lambda: f"plan-{uuid4().hex[:10]}")
    feedback_ids_consumed: list[str] = field(default_factory=list)
    evidence_refs_consumed: list[str] = field(default_factory=list)
    adaptation_summary: str = ""
    planner_contract_version: str = ""
    planner_contract_violations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class A1TaskRequest:
    run_id: str
    iteration: int
    research_goal: str
    hypothesis: str
    step: ExperimentStep
    global_constraints: list[str]
    prior_observations: list[dict[str, Any]]
    prior_evaluations: list[dict[str, Any]]
    allowed_paths: list[str]
    state_version: int = 0
    route_retry_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_id: str = ""
    canonical_entities: list[CanonicalEntityRecord] = field(default_factory=list)
    entity_corrections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "research_goal": self.research_goal,
            "working_assumption": self.hypothesis,
            "experiment_step": asdict(self.step),
            "global_constraints": self.global_constraints,
            "prior_observations": self.prior_observations,
            "prior_evaluations": self.prior_evaluations,
            "allowed_paths": self.allowed_paths,
            "state_version": self.state_version,
            "request_id": self.request_id,
            "canonical_entities": [asdict(item) for item in self.canonical_entities],
            "entity_corrections": dict(self.entity_corrections),
            "response_contract": {
                "answer": "concise result",
                "output": "JSON object to persist at output_config.file_path when a file output is required",
                "tool_trace": "internal tools actually used",
                "observations": "structured experimental observations",
                "metrics": "numeric metrics only",
                "artifacts": "paths created under allowed_paths only",
                "method_provenance": (
                    "list of method records linked to the actual tool or artifact; "
                    "required for method-sensitive scientific evidence"
                ),
                "errors": "execution errors",
            },
        }

    def to_prompt(self) -> str:
        return (
            "Execute exactly one bounded scientific experiment step. Do not redesign "
            "the overall research plan. Access only the allowed paths and return the "
            "requested structured evidence. The experiment_step.inputs contains the "
            "planner's workflow_phase and semantic_intent; OmniAgent has already resolved "
            "that intent to this A1 execution. Carry out the requested open-ended "
            "experiment, code/data work, multi-tool analysis, or artifact creation. "
            "When canonical_entities are present, they were resolved by the Harness "
            "from an authoritative database and override conflicting names in the "
            "planner text. When they are absent, resolve entities within Biomni from "
            "the supplied entity context. Never use an identifier listed in "
            "entity_corrections; use its replacement instead. "
            "Keep the final answer concise. Do not paste "
            "full source code or full datasets into the final answer; create artifacts "
            "inside allowed paths when code or data output is needed. If the task "
            "requires a JSON output file, return its complete JSON body in the "
            "structured 'output' field so the runtime can persist it under the task "
            "output contract.\n\n"
            + json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        )


@dataclass(slots=True)
class A1TaskResult:
    success: bool
    result_status: str = "success"
    answer: str = ""
    output: Any = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    verification_payload: Any = field(
        default=None,
        repr=False,
        metadata={"runtime_only": True},
    )


@dataclass(slots=True)
class PendingExecution:
    """Durable identity and poll state for a submitted Biomni task."""

    step_id: str
    iteration: int
    task_id: str
    request_id: str
    gateway_tool: str
    backend: str
    status: str = "submitted"
    next_poll_at: float = 0.0
    deadline_at: float = 0.0
    consecutive_poll_errors: int = 0
    last_poll_error: str = ""
    task_metadata: dict[str, Any] = field(default_factory=dict)
    remote_status: str = ""
    rpc_status: str = "not_started"
    unknown_reason: str = ""
    wait_started_at: float = 0.0
    action_id: str = ""
    idempotency_key: str = ""


@dataclass(slots=True)
class AnalysisResult:
    summary: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    final_output: dict[str, Any] | None = None
    supporting_evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkflowEvaluation:
    evaluator_id: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    satisfied_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retryable: bool = True
    evaluation_id: str = field(
        default_factory=lambda: f"evaluation-{uuid4().hex[:10]}"
    )


@dataclass(slots=True)
class Critique:
    decision: Decision
    score: float
    summary: str
    satisfied_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    required_changes: list[str] = field(default_factory=list)
    next_experiment: str = ""
    retryable: bool = True
    source_evaluation_id: str = ""
    feedback_id: str = field(default_factory=lambda: f"feedback-{uuid4().hex[:10]}")

    @property
    def requires_consumption(self) -> bool:
        return self.decision in {Decision.CONTINUE, Decision.REPLAN, Decision.RETRY}


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    plan: ExperimentPlan
    executions: list[A1TaskResult]
    analysis: AnalysisResult
    evaluation: WorkflowEvaluation
    critique: Critique
    score_improvement: float = 0.0


@dataclass(slots=True)
class LoopPolicy:
    max_iterations: int = 6
    max_a1_calls: int = 12
    target_score: float = 0.8
    min_improvement: float = 0.01
    max_stalled_iterations: int = 2
    max_steps_per_iteration: int = 1
    max_plan_contract_retries: int = 1
    working_memory_token_budget: int = 2400

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_a1_calls < 1:
            raise ValueError("max_a1_calls must be positive")
        if not 0.0 <= self.target_score <= 1.0:
            raise ValueError("target_score must be between 0 and 1")
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative")
        if self.max_stalled_iterations < 1:
            raise ValueError("max_stalled_iterations must be positive")
        if self.max_steps_per_iteration < 1:
            raise ValueError("max_steps_per_iteration must be positive")
        if self.max_plan_contract_retries < 0:
            raise ValueError("max_plan_contract_retries cannot be negative")
        if self.working_memory_token_budget < 256:
            raise ValueError("working_memory_token_budget must be at least 256")


@dataclass(slots=True)
class ResearchState:
    goal: str
    constraints: list[str]
    workspace: str
    task_manifest: dict[str, Any]
    run_id: str = field(default_factory=lambda: f"run-{uuid4().hex[:12]}")
    phase: LoopPhase = LoopPhase.INITIALIZE
    status: RunStatus = RunStatus.RUNNING
    iterations: list[IterationRecord] = field(default_factory=list)
    best_score: float = 0.0
    stalled_iterations: int = 0
    a1_call_count: int = 0
    finish_reason: str = ""
    working_memory_token_budget: int = 2400
    active_plan: ExperimentPlan | None = None
    active_executions: list[A1TaskResult] = field(default_factory=list)
    pending_execution: PendingExecution | None = None
    route_retry_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_ledger: ActionLedger = field(default_factory=ActionLedger)
    final_output_materialized: bool = False
    scientific_state: ScientificState = field(init=False)

    def __post_init__(self) -> None:
        task_id = str(self.task_manifest.get("id", "")).strip() or self.run_id
        self.scientific_state = ScientificState(task_id=task_id, goal=self.goal)

    @property
    def last_critique(self) -> Critique | None:
        return self.iterations[-1].critique if self.iterations else None

    @property
    def last_evaluation(self) -> WorkflowEvaluation | None:
        return self.iterations[-1].evaluation if self.iterations else None

    def prior_observations(self) -> list[dict[str, Any]]:
        return list(
            self.working_memory(purpose="execute").get("evidence", [])
        )

    def prior_evaluations(self) -> list[dict[str, Any]]:
        return list(
            self.working_memory(purpose="execute").get("prior_evaluations", [])
        )

    def working_memory(self, *, purpose: str, focus: str = "") -> dict[str, Any]:
        from .working_memory import WorkingMemoryProjector

        return WorkingMemoryProjector(self.working_memory_token_budget).project(
            self,
            purpose=purpose,
            focus=focus,
        )

    def planner_context(self) -> dict[str, Any]:
        return self.working_memory(purpose="plan")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ActionLedger):
        return _jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
            if not item.metadata.get("runtime_only")
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
