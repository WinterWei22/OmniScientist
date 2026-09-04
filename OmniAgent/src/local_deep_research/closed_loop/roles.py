from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Protocol

from .contracts import (
    A1TaskResult,
    AnalysisResult,
    Critique,
    Decision,
    ExperimentPlan,
    ExperimentStep,
    PlannerContractError,
    ResearchState,
    WorkflowEvaluation,
)
from .execution_models import (
    ExecutionBackend,
    ExecutionShape,
    SemanticCapabilityIntent,
    SemanticOperation,
)


class Planner(Protocol):
    async def plan(self, state: ResearchState) -> ExperimentPlan: ...


class Analyzer(Protocol):
    async def analyze(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        results: list[A1TaskResult],
    ) -> AnalysisResult: ...


class Verifier(Protocol):
    async def verify(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        analysis: AnalysisResult,
        evaluation: WorkflowEvaluation,
    ) -> Critique: ...


class LLMPlanner:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def plan(self, state: ResearchState) -> ExperimentPlan:
        previous_critique = state.last_critique
        previous_evaluation = state.last_evaluation
        required_feedback_id = (
            previous_critique.feedback_id
            if previous_critique and previous_critique.requires_consumption
            else ""
        )
        required_evaluation_id = (
            previous_evaluation.evaluation_id
            if required_feedback_id and previous_evaluation
            else ""
        )
        replan_contract = ""
        if required_feedback_id:
            replan_contract = (
                "\n\nMANDATORY REPLAN CONTRACT:\n"
                "This is a replan after deterministic evaluation feedback. Copy these "
                "exact IDs into the output arrays; do not omit them or return empty arrays:\n"
                f"- feedback_ids_consumed must include {required_feedback_id!r}\n"
                f"- evidence_refs_consumed must include {required_evaluation_id!r}\n"
                "- adaptation_summary must state what changed in this next step and why "
                "it addresses the required changes below.\n"
                "The arrays are provenance metadata, not new scientific evidence.\n"
            )
        prompt = (
            "You are OmniAgent's scientific Workflow Planner. Produce exactly one bounded "
            "next step. You propose scientific intent; the Harness, not the model, chooses "
            "MCP versus A1, resolves a canonical capability, compiles arguments from its "
            "schema, and owns all workspace writes. Never output execution_backend, "
            "tool_name, or arguments. An optional capability_hint may describe a capability "
            "in semantic terms, but it is not an executable instruction.\n"
            "Each step must contain step_id, objective, inputs, constraints, "
            "expected_outputs, and success_criteria. The inputs object MUST contain "
            "workflow_phase (research|execution|validation|synthesis), tool_query, and "
            "semantic_intent. Put these three keys inside inputs, not beside inputs. "
            "semantic_intent must contain operation "
            "(retrieve|validate|analyze|experiment|generate_artifact|synthesize), "
            "capability_query, execution_shape (single_capability|multi_capability|adaptive), "
            "schema_bound (boolean), side_effect (read_only|workspace_write), "
            "required_output_fields, expected_artifacts, entity_context, and rationale. "
            "Use protocol-neutral entity keys such as gene_symbol, protein_name, organism, "
            "and experimental_method; do not use MCP argument names or raw request bodies. "
            "Only declare raw fields that this immediate step can return. Do not request the "
            "configured final output file: the Result Analyst returns grounded JSON and the "
            "Harness materializes it atomically. Database retrieval results do not need a "
            "planner-requested workspace file.\n"
            "On later iterations, explicitly adapt to deterministic metrics, failed criteria, "
            "execution errors, and required changes. Consume the previous feedback_id and "
            "evaluation_id and explain the adaptation.\n"
            "Return JSON with keys hypothesis, rationale, adaptation_summary, "
            "feedback_ids_consumed, evidence_refs_consumed, and steps. hypothesis is an "
            "optional working assumption. steps must contain exactly one item with step_id, "
            "objective, inputs, constraints, expected_outputs, and success_criteria. "
            "For clarity, the required shape is: inputs={workflow_phase, tool_query, "
            "semantic_intent={operation, capability_query, execution_shape, schema_bound, "
            "side_effect, required_output_fields, expected_artifacts, entity_context, "
            "rationale}, plus protocol-neutral entity values. The literal string "
            "final_submission.json must never appear anywhere in your response. Do not "
            "plan creating, writing, reading, or updating the final submission file."
            + replan_contract
            + "\nSTATE:\n"
            + json.dumps(_planner_context(state), ensure_ascii=False, indent=2)
        )
        data = await _invoke_json(self.model, prompt)
        data = _sanitize_harness_owned_output_references(
            data, _harness_output_name(state)
        )
        top_level_forbidden = [
            key
            for key in ("execution_backend", "tool_name", "arguments")
            if key in data
        ]
        if top_level_forbidden:
            raise PlannerContractError(
                "Planner cannot provide execution-owned top-level field(s): "
                + ", ".join(top_level_forbidden)
            )
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != 1:
            raise PlannerContractError("Planner must return exactly one step")
        steps = []
        violations: list[str] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                raise PlannerContractError("Planner step must be a JSON object")
            inputs = _mapping(item.get("inputs", {}))
            forbidden = [
                key
                for key in ("execution_backend", "tool_name", "arguments")
                if key in item or key in inputs
            ]
            if forbidden:
                violations.append(
                    "Planner cannot choose execution-owned field(s): "
                    + ", ".join(forbidden)
                )
            for key in ("workflow_phase", "tool_query", "semantic_intent"):
                top_level_value = item.get(key)
                nested_value = inputs.get(key)
                if key in item and key in inputs and top_level_value != nested_value:
                    violations.append(
                        f"Planner step has conflicting {key} values beside and inside inputs"
                    )
                elif key in item and key not in inputs:
                    # Accept the earlier documented shape, but canonicalize it before
                    # the request reaches binding or execution.
                    inputs[key] = top_level_value
            if not str(inputs.get("tool_query", "")).strip():
                violations.append("Planner step is missing tool_query")
            if not isinstance(inputs.get("semantic_intent"), dict):
                violations.append("Planner step is missing semantic_intent")
            else:
                raw_intent = inputs["semantic_intent"]
                try:
                    raw_intent["operation"] = SemanticOperation.normalize(
                        raw_intent.get("operation")
                    ).value
                except ValueError:
                    pass
                missing_intent_keys = [
                    key
                    for key in (
                        "operation",
                        "capability_query",
                        "execution_shape",
                        "schema_bound",
                        "side_effect",
                    )
                    if key not in raw_intent
                ]
                if missing_intent_keys:
                    violations.append(
                        "Planner semantic_intent is missing: "
                        + ", ".join(missing_intent_keys)
                    )
                elif SemanticCapabilityIntent.from_inputs(inputs) is None:
                    violations.append(
                        "Planner semantic_intent does not satisfy its enum/schema contract"
                    )
            if not str(item.get("step_id", "")).strip():
                violations.append("Planner step_id is required")
            if not str(item.get("objective", "")).strip():
                violations.append("Planner objective is required")
            steps.append(
                ExperimentStep(
                    step_id=str(item.get("step_id", "")),
                    objective=str(item.get("objective", "")),
                    inputs=inputs,
                    constraints=_strings(item.get("constraints", [])),
                    expected_outputs=_strings(item.get("expected_outputs", [])),
                    success_criteria=_strings(item.get("success_criteria", [])),
                )
            )
        if violations:
            raise PlannerContractError("; ".join(dict.fromkeys(violations)))
        feedback_ids = _strings(data.get("feedback_ids_consumed", []))
        evidence_refs = _strings(data.get("evidence_refs_consumed", []))
        adaptation_summary = str(data.get("adaptation_summary", "")).strip()
        if required_feedback_id:
            normalized_fields: list[str] = []
            if required_feedback_id not in feedback_ids:
                feedback_ids.append(required_feedback_id)
                normalized_fields.append("feedback_ids_consumed")
            if required_evaluation_id and required_evaluation_id not in evidence_refs:
                evidence_refs.append(required_evaluation_id)
                normalized_fields.append("evidence_refs_consumed")
            if not adaptation_summary:
                changes = (
                    previous_critique.next_experiment.strip()
                    if previous_critique and previous_critique.next_experiment.strip()
                    else "; ".join(previous_critique.required_changes)
                    if previous_critique
                    else "the previous evaluation's failed criteria"
                )
                adaptation_summary = (
                    "Contract normalization recorded the required prior feedback and "
                    f"evaluation references; the next step addresses: {changes}."
                )
                normalized_fields.append("adaptation_summary")
            if normalized_fields:
                adaptation_summary = (
                    adaptation_summary
                    + " [planner contract normalized: "
                    + ", ".join(normalized_fields)
                    + "]"
                )
        return ExperimentPlan(
            hypothesis=str(data.get("hypothesis", "")),
            rationale=str(data.get("rationale", "")),
            steps=steps,
            feedback_ids_consumed=feedback_ids,
            evidence_refs_consumed=evidence_refs,
            adaptation_summary=adaptation_summary,
            planner_contract_version="omniagent.planner.v2",
            planner_contract_violations=[],
        )


class LLMAnalyzer:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def analyze(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        results: list[A1TaskResult],
    ) -> AnalysisResult:
        focus = " ".join(step.objective for step in plan.steps)
        prompt = (
            "You are OmniAgent's Result Analyst. Convert experimental outputs into "
            "structured observations. Preserve numeric metrics and identify anomalies "
            "and evidence gaps. Do not propose the next research plan or create files. "
            "When task_output_contract is present and the verifier-admitted evidence is "
            "sufficient, return final_output as the complete JSON object required by that "
            "contract and supporting_evidence_ids as the exact existing evidence IDs that "
            "support it. Otherwise return final_output: null and explain the gap. Do not "
            "invent evidence IDs, semantic methods, or evidence fields. Return JSON with "
            "summary, observations, metrics, anomalies, evidence_gaps, final_output, "
            "supporting_evidence_ids.\n\n"
            + json.dumps(
                {
                    "working_memory": state.working_memory(
                        purpose="analyze", focus=focus
                    ),
                    "plan": plan,
                    "results": _compact_results(results),
                    "task_output_contract": state.task_manifest.get("output_config", {}),
                },
                default=_json_default,
                ensure_ascii=False,
                indent=2,
            )
        )
        data = await _invoke_json(self.model, prompt)
        metrics = _numeric_metrics(data.get("metrics", {}))
        return AnalysisResult(
            summary=str(data.get("summary", "")),
            observations=_records(data.get("observations", [])),
            metrics=metrics,
            anomalies=_strings(data.get("anomalies", [])),
            evidence_gaps=_strings(data.get("evidence_gaps", [])),
            final_output=(
                _mapping(data.get("final_output"))
                if isinstance(data.get("final_output"), dict)
                else None
            ),
            supporting_evidence_ids=_strings(data.get("supporting_evidence_ids", [])),
        )


class LLMVerifier:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def verify(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        analysis: AnalysisResult,
        evaluation: WorkflowEvaluation,
    ) -> Critique:
        focus = " ".join(step.objective for step in plan.steps)
        prompt = (
            "You are OmniAgent's Feedback Decider. The deterministic evaluator is the only "
            "authority for score and criteria; do not invent or revise its metrics. Decide "
            "whether the workflow should continue, replan, retry, stop, or fail, and state "
            "the concrete change required for the next round. The configured final output "
            "is owned by the Harness; never instruct the Planner to create or write it. "
            "Return JSON with decision "
            "(continue|replan|retry|stop|fail), summary, evidence_gaps, "
            "required_changes, next_experiment, retryable.\n\n"
            + json.dumps(
                {
                    "working_memory": _model_working_memory(
                        state, purpose="verify", focus=focus
                    ),
                    "plan": plan,
                    "analysis": analysis,
                    "deterministic_evaluation": evaluation,
                },
                default=_json_default,
                ensure_ascii=False,
                indent=2,
            )
        )
        data = await _invoke_json(self.model, prompt)
        try:
            decision = Decision(str(data.get("decision", Decision.REPLAN.value)).lower())
        except ValueError:
            decision = Decision.REPLAN
        return Critique(
            decision=decision,
            score=evaluation.score,
            summary=str(data.get("summary", "")),
            satisfied_criteria=list(evaluation.satisfied_criteria),
            failed_criteria=list(evaluation.failed_criteria),
            evidence_gaps=_strings(data.get("evidence_gaps", [])),
            required_changes=(
                _strings(data.get("required_changes", []))
                or list(evaluation.failed_criteria)
            ),
            next_experiment=str(data.get("next_experiment", "")),
            retryable=evaluation.retryable and bool(data.get("retryable", True)),
            source_evaluation_id=evaluation.evaluation_id,
        )


async def _invoke_json(model: Any, prompt: str) -> dict[str, Any]:
    attempts = 3
    base_delay_seconds = 30.0
    for attempt in range(attempts):
        try:
            response = await model.ainvoke(prompt)
            return _parse_json_response(response)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= attempts - 1:
                raise
            await asyncio.sleep(base_delay_seconds * (attempt + 1))
    raise RuntimeError("unreachable Qwen retry state")


def _parse_json_response(response: Any) -> dict[str, Any]:
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ModuleNotFoundError as import_error:
            raise RuntimeError(
                "Malformed LLM JSON requires the declared 'json-repair' dependency"
            ) from import_error
        value = json.loads(repair_json(text[start : end + 1]))
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value


def _is_rate_limit_error(exc: Exception) -> bool:
    marker = f"{type(exc).__name__}: {exc}".lower()
    return any(value in marker for value in ("ratelimit", "rate limit", "tpm limit", "429"))


def _harness_output_name(state: ResearchState) -> str:
    output_config = state.task_manifest.get("output_config", {})
    if not isinstance(output_config, dict):
        return ""
    return str(output_config.get("file_path", "")).strip()


def _planner_context(state: ResearchState) -> dict[str, Any]:
    context = _model_working_memory(state, purpose="plan")
    task_contract = context.get("task_contract")
    if not isinstance(task_contract, dict):
        return context
    output_config = task_contract.get("output_config")
    if not isinstance(output_config, dict):
        return context
    task_contract["output_config"] = {
        key: value
        for key, value in output_config.items()
        if key != "file_path"
    }
    task_contract["output_config"]["managed_by_harness"] = True
    return context


def _model_working_memory(
    state: ResearchState,
    *,
    purpose: str,
    focus: str = "",
) -> dict[str, Any]:
    """Return bounded working memory while preserving the full research goal.

    WorkingMemoryProjector intentionally compacts its dynamic projection to a token
    budget.  The global goal is the task contract, however, and truncating it can
    hide later workflow stages and final deliverables from the Planner and Verifier.
    Keep the rest of the projection bounded, then restore the verbatim goal for the
    two model roles that make planning and termination decisions.
    """
    context = state.working_memory(purpose=purpose, focus=focus)
    objective = context.get("objective")
    if not isinstance(objective, dict):
        objective = {}
        context["objective"] = objective
    objective["goal"] = state.goal
    metadata = context.get("memory_metadata")
    if isinstance(metadata, dict):
        metadata["global_goal_preserved_verbatim"] = True
        metadata["global_goal_outside_token_budget"] = True
        metadata["global_goal_char_count"] = len(state.goal)
    return context


def _sanitize_harness_owned_output_references(
    value: Any, output_name: str
) -> Any:
    """Remove final-artifact paths from planner text before plan validation."""
    if not output_name:
        return value
    normalized_name = output_name.replace("\\", "/")
    basename = normalized_name.rsplit("/", 1)[-1]
    targets = tuple(
        target.casefold()
        for target in (normalized_name, basename)
        if target
    )
    if isinstance(value, dict):
        return {
            key: _sanitize_harness_owned_output_references(item, output_name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_harness_owned_output_references(item, output_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_harness_owned_output_references(item, output_name)
            for item in value
        )
    if not isinstance(value, str):
        return value
    lowered = value.casefold()
    for target in targets:
        index = lowered.find(target)
        if index >= 0:
            return (
                value[:index]
                + "Harness-owned final result"
                + value[index + len(target) :]
            )
    return value


ANALYZER_MAX_ANSWER_CHARS = 6000
ANALYZER_MAX_TRACE_EVENTS = 12
ANALYZER_MAX_TRACE_CHARS = 1600
ANALYZER_MAX_OBSERVATIONS = 24
ANALYZER_MAX_OBSERVATION_CHARS = 1600
ANALYZER_MAX_ARTIFACTS = 64
ANALYZER_MAX_ERRORS = 16


def _compact_results(results: list[A1TaskResult]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for result in results:
        compact.append(
            {
                "success": result.success,
                "answer": _compact_text(result.answer, ANALYZER_MAX_ANSWER_CHARS),
                "tool_trace": _compact_trace(
                    result.tool_trace,
                    max_events=ANALYZER_MAX_TRACE_EVENTS,
                    text_limit=ANALYZER_MAX_TRACE_CHARS,
                ),
                "observations": _compact_records(
                    result.observations,
                    max_records=ANALYZER_MAX_OBSERVATIONS,
                    text_limit=ANALYZER_MAX_OBSERVATION_CHARS,
                ),
                "metrics": result.metrics,
                "artifacts": result.artifacts[:ANALYZER_MAX_ARTIFACTS],
                "errors": result.errors[:ANALYZER_MAX_ERRORS],
            }
        )
    return compact


def _compact_trace(
    trace: list[dict[str, Any]], *, max_events: int, text_limit: int
) -> list[dict[str, Any]]:
    selected = trace[-max_events:] if len(trace) > max_events else trace
    compact = [_compact_mapping(item, text_limit) for item in selected]
    omitted = len(trace) - len(selected)
    if omitted > 0:
        compact.insert(0, {"event": "trace_truncated", "omitted_events": omitted})
    return compact


def _compact_records(
    value: Any, *, max_records: int, text_limit: int
) -> list[dict[str, Any]]:
    records = _records(value)
    selected = records[:max_records]
    compact = [_compact_mapping(item, text_limit) for item in selected]
    omitted = len(records) - len(selected)
    if omitted > 0:
        compact.append({"type": "records_truncated", "omitted_records": omitted})
    return compact


def _compact_mapping(value: dict[str, Any], text_limit: int) -> dict[str, Any]:
    return {str(key): _compact_value(item, text_limit) for key, item in value.items()}


def _compact_value(value: Any, text_limit: int) -> Any:
    if isinstance(value, dict):
        return _compact_mapping(value, text_limit)
    if isinstance(value, list):
        return [_compact_value(item, text_limit) for item in value[:20]]
    if isinstance(value, str):
        return _compact_text(value, text_limit)
    return value


def _compact_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated " + str(len(text) - limit) + " chars]"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value in (None, ""):
        return {}
    return {"value": value}


def _normalize_execution_backend(inputs: dict[str, Any]) -> str:
    requested = str(inputs.get("execution_backend", "")).strip().lower()
    if requested in {item.value for item in ExecutionBackend}:
        return requested
    if requested:
        raise PlannerContractError(
            f"unsupported execution_backend {requested!r}; Harness binding owns backend selection"
        )
    intent = SemanticCapabilityIntent.from_inputs(inputs)
    if (
        intent is not None
        and intent.operation in {SemanticOperation.RETRIEVE, SemanticOperation.VALIDATE}
        and intent.execution_shape
        in {ExecutionShape.SINGLE_CAPABILITY, ExecutionShape.MULTI_CAPABILITY}
        and intent.schema_bound
    ):
        return ExecutionBackend.MCP.value
    if isinstance(inputs.get("arguments"), dict) and inputs.get("tool_name"):
        return ExecutionBackend.MCP.value
    return ""


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    records: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            records.append({str(key): value for key, value in item.items()})
        else:
            records.append({"value": str(item)})
    return records


def _numeric_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)
