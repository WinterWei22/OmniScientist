from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .contracts import A1TaskResult
from .execution_models import (
    BoundCapabilityWorkflow,
    EffectContract,
    PathValueRequirement,
    ResourceCandidate,
    SemanticCapabilityIntent,
    WorkflowCallTemplate,
    stage_request_id,
)
from .execution_validation import (
    select_path,
    schema_declares_path,
    validate_schema_instance,
    verify_effects,
)
from .task_metadata import upsert_external_task_snapshot

if TYPE_CHECKING:
    from .contracts import A1TaskRequest


class WorkflowBindingError(ValueError):
    pass


def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_resolve_value(item, context) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$ref"}:
        selected = select_path(context, str(value["$ref"]))
        if not selected:
            raise WorkflowBindingError(f"workflow reference produced no value: {value['$ref']}")
        return selected[0] if len(selected) == 1 else selected
    if set(value) == {"$select"}:
        spec = value["$select"]
        if not isinstance(spec, dict):
            raise WorkflowBindingError("$select requires an object")
        source = _resolve_value(spec.get("source"), context)
        return select_path(source, str(spec.get("path", "")))
    if set(value) == {"$map"}:
        spec = value["$map"]
        if not isinstance(spec, dict):
            raise WorkflowBindingError("$map requires an object")
        source = _resolve_value(spec.get("source"), context)
        values = source if isinstance(source, list) else [source]
        variables = spec.get("variables", {})
        resolved_variables = {
            str(key): _resolve_value(item, context)
            for key, item in variables.items()
        } if isinstance(variables, dict) else {}
        template = str(spec.get("template", "{value}"))
        return [
            template.format(value=item, **resolved_variables)
            for item in values
        ]
    if set(value) == {"$unique"}:
        resolved = _resolve_value(value["$unique"], context)
        values = resolved if isinstance(resolved, list) else [resolved]
        return list(dict.fromkeys(values))
    return {str(key): _resolve_value(item, context) for key, item in value.items()}


def _schema_for_reference(
    reference: str,
    *,
    prior_output_schemas: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a workflow reference and return its source schema when known."""
    parts = [part for part in str(reference).split(".") if part]
    if not parts:
        raise WorkflowBindingError("workflow reference must not be empty")
    if parts[0] == "inputs":
        if len(parts) < 2 or not select_path(inputs, ".".join(parts[1:])):
            raise WorkflowBindingError(f"workflow input reference is unavailable: {reference}")
        return None
    if parts[0] != "steps" or len(parts) < 3 or parts[2] != "result":
        raise WorkflowBindingError(
            "workflow references must target inputs.<field> or steps.<id>.result"
        )
    step_id = parts[1]
    schema = prior_output_schemas.get(step_id)
    if not isinstance(schema, dict) or not schema:
        raise WorkflowBindingError(
            f"workflow reference must target a prior step with an output schema: {reference}"
        )
    relative_path = ".".join(parts[3:])
    if relative_path and not schema_declares_path(schema, relative_path):
        raise WorkflowBindingError(
            f"workflow reference path is not declared by step {step_id}: {relative_path}"
        )
    return schema


def _validate_argument_template(
    value: Any,
    *,
    prior_output_schemas: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
) -> bool:
    """Validate references without trying to materialize results from future steps."""
    if isinstance(value, list):
        return any(
            _validate_argument_template(
                item,
                prior_output_schemas=prior_output_schemas,
                inputs=inputs,
            )
            for item in value
        )
    if not isinstance(value, dict):
        return False
    if set(value) == {"$ref"}:
        _schema_for_reference(
            str(value["$ref"]),
            prior_output_schemas=prior_output_schemas,
            inputs=inputs,
        )
        return True
    if set(value) == {"$select"}:
        spec = value["$select"]
        if not isinstance(spec, dict) or "source" not in spec:
            raise WorkflowBindingError("$select requires a source")
        source = spec["source"]
        _validate_argument_template(
            source,
            prior_output_schemas=prior_output_schemas,
            inputs=inputs,
        )
        if isinstance(source, dict) and set(source) == {"$ref"}:
            source_schema = _schema_for_reference(
                str(source["$ref"]),
                prior_output_schemas=prior_output_schemas,
                inputs=inputs,
            )
            select_path_value = str(spec.get("path", ""))
            if source_schema is not None and not schema_declares_path(
                source_schema, select_path_value
            ):
                raise WorkflowBindingError(
                    "workflow selection path is not declared by its source output schema: "
                    + select_path_value
                )
        return True
    if set(value) == {"$unique"}:
        return _validate_argument_template(
            value["$unique"],
            prior_output_schemas=prior_output_schemas,
            inputs=inputs,
        )
    if set(value) == {"$map"}:
        spec = value["$map"]
        if not isinstance(spec, dict) or not isinstance(spec.get("template", ""), str):
            raise WorkflowBindingError("$map requires a string template")
        deferred = _validate_argument_template(
            spec.get("source"),
            prior_output_schemas=prior_output_schemas,
            inputs=inputs,
        )
        variables = spec.get("variables", {})
        if not isinstance(variables, dict):
            raise WorkflowBindingError("$map variables must be an object")
        return deferred or any(
            _validate_argument_template(
                item,
                prior_output_schemas=prior_output_schemas,
                inputs=inputs,
            )
            for item in variables.values()
        )
    return any(
        _validate_argument_template(
            item,
            prior_output_schemas=prior_output_schemas,
            inputs=inputs,
        )
        for item in value.values()
    )


def workflow_from_record(record: dict[str, Any]) -> BoundCapabilityWorkflow:
    def effect(raw: Any) -> EffectContract:
        if not isinstance(raw, dict):
            return EffectContract()
        requirements = raw.get("required_value_matches", [])
        return EffectContract(
            required_paths=tuple(str(item) for item in raw.get("required_paths", [])),
            any_of_paths=tuple(str(item) for item in raw.get("any_of_paths", [])),
            required_value_matches=tuple(
                PathValueRequirement(
                    path=str(item.get("path", "")),
                    expected_values=tuple(
                        str(value) for value in item.get("expected_values", [])
                    ),
                    case_sensitive=bool(item.get("case_sensitive", False)),
                )
                for item in requirements
                if isinstance(item, dict) and str(item.get("path", "")).strip()
            ),
            required_artifacts=tuple(
                str(item) for item in raw.get("required_artifacts", [])
            ),
            description=str(raw.get("description", "")),
        )

    steps_raw = record.get("steps", [])
    if not isinstance(steps_raw, list):
        raise WorkflowBindingError("persisted workflow steps are not a list")
    steps = [
        WorkflowCallTemplate(
            step_id=str(item.get("step_id", "")),
            tool_name=str(item.get("tool_name", "")),
            arguments=dict(item.get("arguments", {})),
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
            effects=effect(item.get("effects")),
        )
        for item in steps_raw
        if isinstance(item, dict)
    ]
    if not steps or any(not step.step_id or not step.tool_name for step in steps):
        raise WorkflowBindingError("persisted workflow has an invalid step")
    return BoundCapabilityWorkflow(
        workflow_id=str(record.get("workflow_id", "")),
        inputs=(
            dict(record.get("inputs", {}))
            if isinstance(record.get("inputs"), dict)
            else {}
        ),
        steps=steps,
        effects=effect(record.get("effects")),
        max_steps=int(record.get("max_steps", 5)),
        binding_reason=str(record.get("binding_reason", "")),
        evidence_purpose=str(record.get("evidence_purpose", "planning_evidence")),
    )


def _result_body(raw: Any) -> Any:
    if isinstance(raw, dict) and "result" in raw and (
        "ok" in raw or "task_id" in raw or "task_type" in raw
    ):
        return raw["result"]
    return raw


def _verification_body(result: A1TaskResult) -> Any:
    payload = result.verification_payload
    return payload if payload is not None else _result_body(result.raw)


def _result_preview(value: Any, limit: int = 1600) -> str:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return encoded if len(encoded) <= limit else encoded[:limit] + "...[truncated]"


class BoundedMCPWorkflowExecutor:
    """Execute a declarative, bounded chain of exact MCP tool calls."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    async def run(
        self,
        request: A1TaskRequest,
        workflow: BoundCapabilityWorkflow,
        *,
        context: dict[str, Any] | None = None,
        start_index: int = 0,
        trace: list[dict[str, Any]] | None = None,
        child_tasks: list[dict[str, Any]] | None = None,
        artifacts: list[str] | None = None,
        output_schema_errors: list[str] | None = None,
        parent_request_id: str = "",
    ) -> A1TaskResult:
        if not 1 <= len(workflow.steps) <= min(workflow.max_steps, 5):
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow",
                errors=["bounded MCP workflows must contain between 1 and 5 steps"],
            )
        if not 0 <= start_index <= len(workflow.steps):
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow_checkpoint",
                errors=["workflow checkpoint has an invalid next-step index"],
            )
        parent_request_id = str(
            parent_request_id
            or request.request_id
            or f"omniagent:{request.run_id}:{request.step.step_id}"
        ).strip()
        context = context or {"inputs": dict(workflow.inputs), "steps": {}}
        if not isinstance(context.get("inputs"), dict) or not isinstance(
            context.get("steps"), dict
        ):
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow_checkpoint",
                errors=["workflow checkpoint has an invalid context"],
            )
        trace = list(trace or [])
        compact_child_tasks: list[dict[str, Any]] = []
        for child in child_tasks or []:
            compact_child_tasks = upsert_external_task_snapshot(
                compact_child_tasks, child
            )
        child_tasks = compact_child_tasks
        artifacts = list(artifacts or [])
        output_schema_errors = list(output_schema_errors or [])
        for index, template in enumerate(workflow.steps[start_index:], start=start_index):
            try:
                arguments = _resolve_value(template.arguments, context)
            except WorkflowBindingError as exc:
                return self._failure(
                    workflow,
                    trace,
                    child_tasks,
                    artifacts,
                    "workflow_binding_failed",
                    str(exc),
                    output_schema_errors=output_schema_errors,
                )
            if not isinstance(arguments, dict):
                return self._failure(
                    workflow,
                    trace,
                    child_tasks,
                    artifacts,
                    "workflow_binding_failed",
                    f"workflow step {template.step_id} arguments are not an object",
                    output_schema_errors=output_schema_errors,
                )
            argument_errors = validate_schema_instance(
                arguments,
                template.input_schema,
                strict_objects=True,
            )
            if argument_errors:
                trace.append(
                    {
                        "event": "mcp_workflow_arguments_rejected",
                        "workflow_id": workflow.workflow_id,
                        "workflow_step_id": template.step_id,
                        "tool_name": template.tool_name,
                        "errors": argument_errors[:3],
                    }
                )
                return self._failure(
                    workflow,
                    trace,
                    child_tasks,
                    artifacts,
                    "invalid_workflow_arguments",
                    "; ".join(argument_errors[:3]),
                    output_schema_errors=output_schema_errors,
                )
            step_inputs = {
                "tool_query": template.tool_name,
                "tool_name": template.tool_name,
                "arguments": arguments,
                "execution_backend": "mcp",
            }
            child_step = replace(
                request.step,
                step_id=f"{request.step.step_id}:{template.step_id}",
                objective=f"Execute bounded workflow step {template.step_id}.",
                inputs=step_inputs,
                expected_outputs=list(template.effects.required_paths),
            )
            trace.append(
                {
                    "event": "mcp_workflow_step_started",
                    "workflow_id": workflow.workflow_id,
                    "workflow_step_id": template.step_id,
                    "workflow_step_index": index,
                    "tool_name": template.tool_name,
                    "arguments": arguments,
                }
            )
            child_request = replace(
                request,
                step=child_step,
                request_id=stage_request_id(
                    parent_request_id,
                    stage="mcp",
                    identity=f"{workflow.workflow_id}:{template.step_id}",
                ),
            )
            direct_bound_call = getattr(self.tool, "invoke_bound_call", None)
            if direct_bound_call is not None:
                result = await direct_bound_call(
                    child_request,
                    tool_name=template.tool_name,
                    arguments=arguments,
                    input_schema=template.input_schema,
                    output_schema=template.output_schema,
                    wait_for_terminal=getattr(
                        self.tool, "supports_blocking_workflow_tasks", False
                    ),
                )
            elif getattr(self.tool, "supports_blocking_workflow_tasks", False):
                result = await self.tool.run(child_request, wait_for_terminal=True)
            else:
                result = await self.tool.run(child_request)
            trace.extend(result.tool_trace)
            artifacts.extend(item for item in result.artifacts if item not in artifacts)
            if isinstance(result.task_metadata, dict) and result.task_metadata:
                child_tasks = upsert_external_task_snapshot(
                    child_tasks, result.task_metadata
                )
            if result.result_status == "task_pending":
                metadata = dict(result.task_metadata)
                metadata["workflow_id"] = workflow.workflow_id
                metadata["workflow_parent_request_id"] = parent_request_id
                metadata["workflow_resume"] = {
                    "parent_request_id": parent_request_id,
                    "waiting_workflow_step": template.step_id,
                    "completed_workflow_steps": list(context["steps"]),
                    "deferred_workflow_steps": [
                        item.step_id for item in workflow.steps[index + 1 :]
                    ],
                    "context": context,
                    "trace": trace,
                    "child_tasks": child_tasks,
                    "artifacts": artifacts,
                    "output_schema_errors": output_schema_errors,
                }
                result.tool_trace = trace
                result.artifacts = artifacts
                result.task_metadata = metadata
                return result
            body = _verification_body(result)
            step_output_errors = (
                validate_schema_instance(body, template.output_schema)
                if result.success
                else []
            )
            output_schema_errors.extend(
                f"{template.step_id}: {error}" for error in step_output_errors
            )
            verification = verify_effects(
                body,
                template.effects,
                artifacts=result.artifacts,
                allowed_paths=request.allowed_paths,
            )
            trace.append(
                {
                    "event": "mcp_workflow_step_completed",
                    "workflow_id": workflow.workflow_id,
                    "workflow_step_id": template.step_id,
                    "tool_name": template.tool_name,
                    "success": result.success,
                    "result_status": result.result_status,
                    "output_schema_errors": step_output_errors,
                    "effect_verification": verification.to_dict(),
                    "result_preview": _result_preview(body),
                }
            )
            if not result.success or step_output_errors or not verification.passed:
                reason = "; ".join(result.errors) or (
                    "; ".join(step_output_errors[:3])
                    or f"workflow step {template.step_id} did not satisfy its effect contract"
                )
                return self._failure(
                    workflow,
                    trace,
                    child_tasks,
                    artifacts,
                    (
                        "workflow_output_schema_failed"
                        if step_output_errors
                        else "workflow_step_failed"
                    ),
                    reason,
                    task_metadata=(
                        dict(result.task_metadata)
                        if isinstance(result.task_metadata, dict)
                        else {}
                    ),
                    raw=body,
                    output_schema_errors=output_schema_errors,
                )
            context["steps"][template.step_id] = {
                "tool_name": template.tool_name,
                "arguments": arguments,
                "result": body,
                "effect_verification": verification.to_dict(),
            }

        aggregate = {
            "workflow_id": workflow.workflow_id,
            "inputs": workflow.inputs,
            "steps": context["steps"],
            "result": context["steps"][workflow.steps[-1].step_id]["result"],
        }
        verification = verify_effects(
            aggregate,
            workflow.effects,
            artifacts=artifacts,
            allowed_paths=request.allowed_paths,
        )
        trace.append(
            {
                "event": "mcp_workflow_completed",
                "workflow_id": workflow.workflow_id,
                "success": verification.passed,
                "effect_verification": verification.to_dict(),
            }
        )
        if not verification.passed:
            return self._failure(
                workflow,
                trace,
                child_tasks,
                artifacts,
                "workflow_effect_unmet",
                "bounded workflow did not satisfy its final effect contract",
                verification=verification.to_dict(),
                raw=aggregate,
                output_schema_errors=output_schema_errors,
            )
        answer_payload = {
            "workflow_id": workflow.workflow_id,
            "result": aggregate["result"],
            "completed_steps": list(context["steps"]),
        }
        answer = json.dumps(answer_payload, ensure_ascii=False, default=str)
        if len(answer) > 12000:
            answer = answer[:12000] + "\n...[truncated]"
        return A1TaskResult(
            success=True,
            result_status="success",
            answer=answer,
            observations=[
                {
                    "workflow_id": workflow.workflow_id,
                    "completed_steps": list(context["steps"]),
                    "result": aggregate["result"],
                }
            ],
            metrics={"workflow_steps_completed": float(len(workflow.steps))},
            artifacts=artifacts,
            tool_trace=trace,
            task_metadata={
                "workflow_id": workflow.workflow_id,
                "workflow_parent_request_id": parent_request_id,
                "child_tasks": child_tasks,
                "effect_verification": verification.to_dict(),
                "output_schema_errors": output_schema_errors,
            },
            raw=aggregate,
        )

    async def resume(
        self,
        request: A1TaskRequest,
        workflow: BoundCapabilityWorkflow,
        completed_task_result: A1TaskResult,
        workflow_resume: dict[str, Any],
    ) -> A1TaskResult:
        waiting_step_id = str(workflow_resume.get("waiting_workflow_step", ""))
        index = next(
            (
                position
                for position, template in enumerate(workflow.steps)
                if template.step_id == waiting_step_id
            ),
            None,
        )
        if index is None:
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow_checkpoint",
                errors=["workflow checkpoint does not identify a known pending step"],
            )
        template = workflow.steps[index]
        context = workflow_resume.get("context")
        trace = workflow_resume.get("trace")
        child_tasks = workflow_resume.get("child_tasks")
        artifacts = workflow_resume.get("artifacts")
        output_schema_errors = workflow_resume.get("output_schema_errors")
        parent_request_id = str(
            workflow_resume.get("parent_request_id")
            or request.request_id
            or f"omniagent:{request.run_id}:{request.step.step_id}"
        ).strip()
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (context, dict),
                (trace, list),
                (child_tasks, list),
                (artifacts, list),
                (output_schema_errors, list),
            )
        ):
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow_checkpoint",
                errors=["workflow checkpoint is incomplete"],
            )
        context = dict(context)
        steps = context.get("steps")
        if not isinstance(steps, dict):
            return A1TaskResult(
                success=False,
                result_status="invalid_workflow_checkpoint",
                errors=["workflow checkpoint has no completed-step context"],
            )
        trace = [item for item in trace if isinstance(item, dict)]
        trace.extend(completed_task_result.tool_trace)
        artifacts = [str(item) for item in artifacts]
        artifacts.extend(
            item for item in completed_task_result.artifacts if item not in artifacts
        )
        compact_child_tasks: list[dict[str, Any]] = []
        for child in child_tasks:
            compact_child_tasks = upsert_external_task_snapshot(
                compact_child_tasks, child
            )
        child_tasks = compact_child_tasks
        if isinstance(completed_task_result.task_metadata, dict):
            child_tasks = upsert_external_task_snapshot(
                child_tasks, completed_task_result.task_metadata
            )
        body = _verification_body(completed_task_result)
        step_output_errors = (
            validate_schema_instance(body, template.output_schema)
            if completed_task_result.success
            else []
        )
        output_schema_errors = [str(item) for item in output_schema_errors]
        output_schema_errors.extend(
            f"{template.step_id}: {error}" for error in step_output_errors
        )
        verification = verify_effects(
            body,
            template.effects,
            artifacts=completed_task_result.artifacts,
            allowed_paths=request.allowed_paths,
        )
        trace.append(
            {
                "event": "mcp_workflow_step_completed",
                "workflow_id": workflow.workflow_id,
                "workflow_step_id": template.step_id,
                "tool_name": template.tool_name,
                "success": completed_task_result.success,
                "result_status": completed_task_result.result_status,
                "output_schema_errors": step_output_errors,
                "effect_verification": verification.to_dict(),
                "result_preview": _result_preview(body),
            }
        )
        if (
            not completed_task_result.success
            or step_output_errors
            or not verification.passed
        ):
            reason = "; ".join(completed_task_result.errors) or (
                "; ".join(step_output_errors[:3])
                or f"workflow step {template.step_id} did not satisfy its effect contract"
            )
            return self._failure(
                workflow,
                trace,
                child_tasks,
                artifacts,
                (
                    "workflow_output_schema_failed"
                    if step_output_errors
                    else "workflow_step_failed"
                ),
                reason,
                task_metadata=(
                    dict(completed_task_result.task_metadata)
                    if isinstance(completed_task_result.task_metadata, dict)
                    else {}
                ),
                raw=body,
                output_schema_errors=output_schema_errors,
            )
        steps[template.step_id] = {
            "tool_name": template.tool_name,
            "arguments": template.arguments,
            "result": body,
            "effect_verification": verification.to_dict(),
        }
        return await self.run(
            request,
            workflow,
            context=context,
            start_index=index + 1,
            trace=trace,
            child_tasks=child_tasks,
            artifacts=artifacts,
            output_schema_errors=output_schema_errors,
            parent_request_id=parent_request_id,
        )

    @staticmethod
    def _failure(
        workflow: BoundCapabilityWorkflow,
        trace: list[dict[str, Any]],
        child_tasks: list[dict[str, Any]],
        artifacts: list[str],
        status: str,
        reason: str,
        *,
        verification: dict[str, Any] | None = None,
        raw: Any = None,
        task_metadata: dict[str, Any] | None = None,
        output_schema_errors: list[str] | None = None,
    ) -> A1TaskResult:
        metadata = dict(task_metadata or {})
        metadata.update(
            {
                "workflow_id": workflow.workflow_id,
                "child_tasks": child_tasks,
                "effect_verification": verification or {"passed": False},
                "output_schema_errors": list(output_schema_errors or []),
            }
        )
        return A1TaskResult(
            success=False,
            result_status=status,
            answer="Bounded MCP workflow failed.",
            errors=[reason],
            artifacts=artifacts,
            tool_trace=trace,
            task_metadata=metadata,
            verification_payload=raw,
            raw=raw,
        )


class CapabilityWorkflowRegistry:
    """Bind semantic effects to declarative workflows without router special cases."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._declarative_records: list[dict[str, Any]] = []
        for record in records or []:
            self.register(record)

    def register(self, record: dict[str, Any]) -> None:
        """Register a validated workflow supplied by configuration or a catalog."""
        if not isinstance(record, dict):
            raise WorkflowBindingError("workflow declaration must be an object")
        if not str(record.get("workflow_id") or "").strip():
            raise WorkflowBindingError("workflow declaration requires workflow_id")
        workflow_from_record(record)
        self._declarative_records.append(dict(record))

    _PDB_COMPONENT_MARKERS = (
        "chemical component",
        "chemcomp",
        "ccd",
        "formal charge",
        "net charge",
        "ligand residue",
        "canonical smiles",
        "inchi",
    )
    _STATE_TRANSFORMATION_MARKERS = (
        "all-neutral",
        "all neutral",
        "non-ionic",
        "non ionic",
        "neutral tautomer",
        "tautomer",
        "protonation state",
        "ph-dependent",
        "ph dependent",
    )
    _PDB_COORDINATE_MARKERS = (
        "coordinate",
        "mmcif",
        ".cif",
        "structure file",
        "atom record",
        "atom count",
        "download pdb",
        "download structure",
    )
    _STRUCTURAL_ANALYSIS_MARKERS = (
        "active site",
        "coordination",
        "coordinating residue",
        "metal identity",
        "distance cutoff",
        "geometric",
        "atom-level",
        "atomic coordinate",
    )
    _PDB_ENTRY_METADATA_PATHS = {
        "entry.id": "detailed_results[*].data.entry.id",
        "entry_id": "detailed_results[*].data.entry.id",
        "pdb_id": "detailed_results[*].data.entry.id",
        "rcsb_accession_numbers": "detailed_results[*].data.entry.id",
        "rcsb_accession_info.deposit_date": (
            "detailed_results[*].data.rcsb_accession_info.deposit_date"
        ),
        "deposit_date": "detailed_results[*].data.rcsb_accession_info.deposit_date",
        "deposition_date": "detailed_results[*].data.rcsb_accession_info.deposit_date",
        "deposition_year": "detailed_results[*].data.rcsb_accession_info.deposit_date",
        "refine.ls_d_res_high": "detailed_results[*].data.refine[*].ls_d_res_high",
        "resolution": "detailed_results[*].data.refine[*].ls_d_res_high",
        "struct.title": "detailed_results[*].data.struct.title",
        "title": "detailed_results[*].data.struct.title",
        "exptl.method": "detailed_results[*].data.exptl[*].method",
        "experimental_method": "detailed_results[*].data.exptl[*].method",
        "rcsb_entry_info.deposited_atom_count": (
            "detailed_results[*].data.rcsb_entry_info.deposited_atom_count"
        ),
        "rcsb_entry_info.deposited_polymer_entity_count": (
            "detailed_results[*].data.rcsb_entry_info.deposited_polymer_entity_count"
        ),
        "rcsb_entry_info.deposited_nonpolymer_entity_count": (
            "detailed_results[*].data.rcsb_entry_info.deposited_nonpolymer_entity_count"
        ),
        "rcsb_entry_info.polymer_composition": (
            "detailed_results[*].data.rcsb_entry_info.polymer_composition"
        ),
    }

    @classmethod
    def required_conditions(
        cls,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> tuple[str, ...]:
        query = cls._query_text(request, intent)
        conditions: list[str] = []
        reference = cls._pdb_entity_reference(request, intent)
        if reference is not None:
            conditions.append(f"pdb_entity_mapping:{reference[0]}_{reference[1]}")
        if cls._requires_state_transformation(query):
            conditions.append("chemical_state:explicit_neutral_tautomer_method")
        return tuple(conditions)

    def unavailability_reason(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        candidates: list[ResourceCandidate],
    ) -> str | None:
        query = self._query_text(request, intent)
        if not any(marker in query for marker in self._PDB_COMPONENT_MARKERS):
            return None
        if not self._requires_state_transformation(query):
            return None
        return (
            "The requested chemical state requires a reproducible neutralization or "
            "tautomer-standardization method. The current admitted workflow can retrieve "
            "a PDB CCD but cannot prove that its raw descriptor represents the requested "
            "chemical state."
        )

    @classmethod
    def requires_adaptive_execution(
        cls,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> bool:
        step_text = " ".join(
            (
                request.step.objective,
                str(request.step.inputs.get("tool_query", "")),
                intent.capability_query,
            )
        ).casefold()
        return any(marker in step_text for marker in cls._STRUCTURAL_ANALYSIS_MARKERS)

    def bind(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        candidates: list[ResourceCandidate],
    ) -> BoundCapabilityWorkflow | None:
        query = self._query_text(request, intent)
        declared = self._bind_declared(request, intent, candidates, query)
        if declared is not None:
            return declared
        if not any(marker in query for marker in self._PDB_COMPONENT_MARKERS):
            return self._pdb_entry_search_workflow(request, intent, candidates, query)
        identifiers = self._pdb_identifiers(request, intent)
        if not identifiers:
            return None
        candidate = next(
            (
                item
                for item in candidates
                if item.qualified_name.endswith(".query_pdb_identifiers")
            ),
            None,
        )
        if candidate is None or not candidate.input_schema:
            return None
        pdb_id = identifiers[0]
        tool_name = candidate.qualified_name
        entity_reference = self._pdb_entity_reference(request, intent)
        if entity_reference is not None:
            return self._entity_component_workflow(
                candidate,
                entity_reference,
                intent,
                query,
            )
        entry_entities = EffectContract(
            required_paths=(
                "detailed_results[*].data.rcsb_entry_container_identifiers.non_polymer_entity_ids[*]",
            ),
            description="PDB entry exposes non-polymer entity identifiers.",
        )
        component_ids = EffectContract(
            required_paths=("detailed_results[*].data.pdbx_entity_nonpoly.comp_id",),
            description="Non-polymer entities expose chemical component identifiers.",
        )
        component_charge = self._component_effect(intent, query)
        return BoundCapabilityWorkflow(
            workflow_id="pdb_chemical_component_definition.v1",
            inputs={"pdb_id": pdb_id},
            max_steps=3,
            binding_reason=(
                "The requested CCD/formal-charge effect requires a bounded PDB entry -> "
                "non-polymer entity -> chemical component chain."
            ),
            steps=[
                WorkflowCallTemplate(
                    step_id="entry",
                    tool_name=tool_name,
                    arguments={
                        "identifiers": [pdb_id],
                        "return_type": "entry",
                        "download": False,
                    },
                    input_schema=candidate.input_schema,
                    output_schema=candidate.output_schema,
                    effects=entry_entities,
                ),
                WorkflowCallTemplate(
                    step_id="nonpolymer_entities",
                    tool_name=tool_name,
                    arguments={
                        "identifiers": {
                            "$unique": {
                                "$map": {
                                    "source": {
                                        "$select": {
                                            "source": {"$ref": "steps.entry.result"},
                                            "path": "detailed_results[*].data.rcsb_entry_container_identifiers.non_polymer_entity_ids[*]",
                                        }
                                    },
                                    "template": "{pdb_id}_{value}",
                                    "variables": {"pdb_id": {"$ref": "inputs.pdb_id"}},
                                }
                            }
                        },
                        "return_type": "nonpolymer_entity",
                        "download": False,
                    },
                    input_schema=candidate.input_schema,
                    output_schema=candidate.output_schema,
                    effects=component_ids,
                ),
                WorkflowCallTemplate(
                    step_id="components",
                    tool_name=tool_name,
                    arguments={
                        "identifiers": {
                            "$unique": {
                                "$select": {
                                    "source": {"$ref": "steps.nonpolymer_entities.result"},
                                    "path": "detailed_results[*].data.pdbx_entity_nonpoly.comp_id",
                                }
                            }
                        },
                        "return_type": "mol_definition",
                        "download": False,
                    },
                    input_schema=candidate.input_schema,
                    output_schema=candidate.output_schema,
                    effects=component_charge,
                ),
            ],
            effects=EffectContract(
                required_paths=tuple(
                    f"steps.components.result.{path}"
                    for path in component_charge.required_paths
                ),
                any_of_paths=tuple(
                    f"steps.components.result.{path}"
                    for path in component_charge.any_of_paths
                ),
                description=(
                    "The enumerated component workflow must satisfy the requested raw "
                    "chemical descriptor contract."
                ),
            ),
        )

    def _bind_declared(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        candidates: list[ResourceCandidate],
        query: str,
    ) -> BoundCapabilityWorkflow | None:
        """Bind configured workflows after checking their match and tool contracts."""
        for record in self._declarative_records:
            match = record.get("match", {})
            if not isinstance(match, dict):
                continue
            operation = str(match.get("operation") or "").strip().lower()
            if operation and operation != intent.operation.value:
                continue
            required_text = str(match.get("query_contains") or "").casefold().strip()
            if required_text and required_text not in query:
                continue
            required_fields = match.get("required_output_fields", [])
            if isinstance(required_fields, str):
                required_fields = [required_fields]
            if isinstance(required_fields, list) and any(
                str(value) not in intent.required_output_fields for value in required_fields
            ):
                continue
            try:
                workflow = workflow_from_record(record)
            except (TypeError, ValueError, WorkflowBindingError):
                continue
            candidate_by_name = {item.qualified_name: item for item in candidates}
            if any(step.tool_name not in candidate_by_name for step in workflow.steps):
                continue
            valid = True
            context = {"inputs": dict(request.step.inputs), "steps": {}}
            prior_output_schemas: dict[str, dict[str, Any]] = {}
            for step in workflow.steps:
                candidate = candidate_by_name[step.tool_name]
                if not candidate.input_schema:
                    valid = False
                    break
                if step.input_schema and step.input_schema != candidate.input_schema:
                    valid = False
                    break
                if step.output_schema and step.output_schema != candidate.output_schema:
                    valid = False
                    break
                try:
                    has_deferred_values = _validate_argument_template(
                        step.arguments,
                        prior_output_schemas=prior_output_schemas,
                        inputs=context["inputs"],
                    )
                    arguments = (
                        None
                        if has_deferred_values
                        else _resolve_value(step.arguments, context)
                    )
                except WorkflowBindingError:
                    valid = False
                    break
                if (
                    arguments is not None
                    and validate_schema_instance(
                        arguments, candidate.input_schema, strict_objects=True
                    )
                ):
                    valid = False
                    break
                if not isinstance(candidate.output_schema, dict) or not candidate.output_schema:
                    valid = False
                    break
                prior_output_schemas[step.step_id] = candidate.output_schema
            if valid:
                return workflow
        return None

    def _pdb_entry_search_workflow(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        candidates: list[ResourceCandidate],
        query: str,
    ) -> BoundCapabilityWorkflow | None:
        if "pdb" not in query and "protein data bank" not in query:
            return None
        if (
            intent.side_effect.value != "read_only"
            or intent.expected_artifacts
            or any(marker in query for marker in self._PDB_COORDINATE_MARKERS)
        ):
            return None
        metadata_effect, deferred_fields = self._entry_metadata_effect(intent)
        if metadata_effect is None:
            return None
        search_candidate = next(
            (item for item in candidates if item.qualified_name.endswith(".query_pdb")),
            None,
        )
        entry_candidate = next(
            (
                item
                for item in candidates
                if item.qualified_name.endswith(".query_pdb_identifiers")
            ),
            None,
        )
        if (
            search_candidate is None
            or entry_candidate is None
            or not search_candidate.input_schema
            or not entry_candidate.input_schema
        ):
            return None
        try:
            max_results = max(1, min(int(request.step.inputs.get("limit", 20)), 100))
        except (TypeError, ValueError):
            max_results = 20
        search_effect = EffectContract(
            required_paths=("result.result_set[*].identifier",),
            description="PDB Search API returns at least one entry identifier.",
        )
        search_arguments = self._pdb_search_arguments(
            request,
            intent,
            search_candidate,
            max_results,
        )
        return BoundCapabilityWorkflow(
            workflow_id="pdb_entry_metadata_search.v1",
            inputs={
                "max_results": max_results,
                "deferred_output_fields": list(deferred_fields),
            },
            max_steps=2,
            binding_reason=(
                "The task requests a PDB entry set and entry-level metadata, so first "
                "retrieve matching entry identifiers and then hydrate those exact entries."
                + (
                    " The remaining requested fields are deliberately deferred because "
                    "they require entity-level or structure-level evidence: "
                    + ", ".join(deferred_fields)
                    + "."
                    if deferred_fields
                    else ""
                )
            ),
            steps=[
                WorkflowCallTemplate(
                    step_id="search",
                    tool_name=search_candidate.qualified_name,
                    arguments=search_arguments,
                    input_schema=search_candidate.input_schema,
                    output_schema=search_candidate.output_schema,
                    effects=search_effect,
                ),
                WorkflowCallTemplate(
                    step_id="entries",
                    tool_name=entry_candidate.qualified_name,
                    arguments={
                        "identifiers": {
                            "$unique": {
                                "$select": {
                                    "source": {"$ref": "steps.search.result"},
                                    "path": "result.result_set[*].identifier",
                                }
                            }
                        },
                        "return_type": "entry",
                        "download": False,
                    },
                    input_schema=entry_candidate.input_schema,
                    output_schema=entry_candidate.output_schema,
                    effects=metadata_effect,
                ),
            ],
            effects=EffectContract(
                required_paths=tuple(
                    f"steps.entries.result.{path}"
                    for path in metadata_effect.required_paths
                ),
                description=(
                    "Each required entry-level field must be present in the records "
                    "hydrated from the PDB search result set."
                ),
            ),
        )

    @staticmethod
    def _pdb_search_arguments(
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        candidate: ResourceCandidate,
        max_results: int,
    ) -> dict[str, Any]:
        """Compile semantic PDB entities into catalog-advertised search parameters."""
        properties = candidate.input_schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        context = dict(intent.entity_context)
        source_text = " ".join(
            part
            for part in (
                request.step.objective,
                str(request.step.inputs.get("tool_query", "")),
                request.research_goal,
            )
            if part
        )
        gene_symbol = CapabilityWorkflowRegistry._preferred_pdb_query_name(
            context.get("gene_symbol", ""),
            context.get("protein_name", ""),
            source_text,
        )
        organism = CapabilityWorkflowRegistry._canonical_organism(
            context.get("organism")
            or CapabilityWorkflowRegistry._extract_organism(source_text)
        )
        arguments: dict[str, Any] = {}
        if "max_results" in properties:
            arguments["max_results"] = max_results
        if gene_symbol and "gene_symbol" in properties:
            arguments["gene_symbol"] = gene_symbol
        if organism and "organism" in properties:
            arguments["organism"] = organism
        if (
            context.get("experimental_method")
            and "experimental_method" in properties
        ):
            arguments["experimental_method"] = context["experimental_method"]
        if any(key in arguments for key in ("gene_symbol", "organism")):
            return arguments
        if "prompt" in properties:
            arguments["prompt"] = request.step.objective or request.research_goal
        return arguments

    @staticmethod
    def _extract_pdb_gene_symbol(text: str) -> str:
        quoted = re.search(
            r"(?:gene|protein)\s*(?:symbol|name)?\s*(?:for|:)?\s*['\"]([A-Za-z][A-Za-z0-9_-]{1,31})['\"]",
            text,
            flags=re.IGNORECASE,
        )
        if quoted:
            return quoted.group(1)
        parenthesized = re.search(
            r"\b([A-Za-z][A-Za-z0-9_-]{1,31})\s*\([^)]{2,32}\)", text
        )
        if parenthesized:
            return parenthesized.group(1)
        named = re.search(
            r"\bfor\s+([A-Za-z][A-Za-z0-9_-]{1,31})\b", text, flags=re.IGNORECASE
        )
        return named.group(1) if named else ""

    @classmethod
    def _preferred_pdb_query_name(
        cls,
        gene_symbol: str,
        protein_name: str,
        source_text: str,
    ) -> str:
        gene_symbol = gene_symbol.strip()
        protein_name = protein_name.strip()
        if protein_name and (
            not gene_symbol or re.fullmatch(r"[A-Za-z]{1,5}\d{3,}", gene_symbol)
        ):
            return protein_name
        return gene_symbol or cls._extract_pdb_gene_symbol(source_text)

    @staticmethod
    def _extract_organism(text: str) -> str:
        match = re.search(
            r"\b([A-Z][a-z]+\s+[a-z]{2,})(?:\s+(?:strain|subsp\.|PAO\d+|ATCC\s*\d+))?",
            text,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _canonical_organism(value: str) -> str:
        match = re.search(r"\b([A-Z][a-z]+\s+[a-z]{2,})\b", value)
        return match.group(1) if match else value.strip()

    def _entry_metadata_effect(
        self,
        intent: SemanticCapabilityIntent,
    ) -> tuple[EffectContract | None, tuple[str, ...]]:
        requested = tuple(
            str(item).strip().casefold() for item in intent.required_output_fields
        )
        if not requested:
            return None, ()
        paths: list[str] = []
        deferred: list[str] = []
        for field in requested:
            path = self._PDB_ENTRY_METADATA_PATHS.get(field)
            if path is None:
                deferred.append(field)
                continue
            paths.append(path)
        if not paths:
            return None, tuple(dict.fromkeys(deferred))
        return (
            EffectContract(
                required_paths=tuple(dict.fromkeys(paths)),
                description="PDB entry records expose every directly available metadata field.",
            ),
            tuple(dict.fromkeys(deferred)),
        )

    def _entity_component_workflow(
        self,
        candidate: ResourceCandidate,
        reference: tuple[str, str],
        intent: SemanticCapabilityIntent,
        query: str,
    ) -> BoundCapabilityWorkflow:
        pdb_id, entity_id = reference
        entity_effect = EffectContract(
            required_paths=(
                "detailed_results[*].data.pdbx_entity_nonpoly.comp_id",
            ),
            required_value_matches=(
                PathValueRequirement(
                    path="detailed_results[*].data.pdbx_entity_nonpoly.entity_id",
                    expected_values=(entity_id,),
                ),
            ),
            description=(
                "The response must map the requested PDB non-polymer entity to its CCD "
                "rather than enumerate unrelated entry components."
            ),
        )
        descriptor_effect = self._component_effect(intent, query)
        return BoundCapabilityWorkflow(
            workflow_id="pdb_entity_component_definition.v2",
            inputs={"pdb_id": pdb_id, "entity_id": entity_id},
            max_steps=2,
            binding_reason=(
                "The requested entity is explicit, so retrieve that non-polymer entity "
                "first and only then retrieve its mapped chemical component."
            ),
            steps=[
                WorkflowCallTemplate(
                    step_id="entity",
                    tool_name=candidate.qualified_name,
                    arguments={
                        "identifiers": [f"{pdb_id}_{entity_id}"],
                        "return_type": "nonpolymer_entity",
                        "download": False,
                    },
                    input_schema=candidate.input_schema,
                    output_schema=candidate.output_schema,
                    effects=entity_effect,
                ),
                WorkflowCallTemplate(
                    step_id="component",
                    tool_name=candidate.qualified_name,
                    arguments={
                        "identifiers": {
                            "$unique": {
                                "$select": {
                                    "source": {"$ref": "steps.entity.result"},
                                    "path": "detailed_results[*].data.pdbx_entity_nonpoly.comp_id",
                                }
                            }
                        },
                        "return_type": "mol_definition",
                        "download": False,
                    },
                    input_schema=candidate.input_schema,
                    output_schema=candidate.output_schema,
                    effects=descriptor_effect,
                ),
            ],
            effects=EffectContract(
                required_paths=tuple(
                    f"steps.component.result.{path}"
                    for path in descriptor_effect.required_paths
                ),
                any_of_paths=tuple(
                    f"steps.component.result.{path}"
                    for path in descriptor_effect.any_of_paths
                ),
                description=(
                    "The final descriptor must come from the CCD mapped from the requested "
                    f"PDB entity {pdb_id}_{entity_id}."
                ),
            ),
        )

    @staticmethod
    def _component_effect(
        intent: SemanticCapabilityIntent,
        query: str,
    ) -> EffectContract:
        fields = " ".join((*intent.required_output_fields, query)).casefold()
        paths: list[str] = []
        if "formal charge" in fields or "net charge" in fields or "charge" in fields:
            paths.append("detailed_results[*].data.chem_comp.pdbx_formal_charge")
        if "inchikey" in fields or "inchi key" in fields:
            paths.append("detailed_results[*].data.rcsb_chem_comp_descriptor.InChIKey")
        if "inchi" in fields and "inchikey" not in fields and "inchi key" not in fields:
            paths.append("detailed_results[*].data.rcsb_chem_comp_descriptor.InChI")
        if "smiles" in fields:
            paths.append("detailed_results[*].data.rcsb_chem_comp_descriptor.SMILES")
        if "formula" in fields:
            paths.append("detailed_results[*].data.chem_comp.formula")
        if "residue" in fields or "ligand name" in fields or "component name" in fields:
            paths.append("detailed_results[*].data.chem_comp.name")
        if not paths:
            paths.append("detailed_results[*].data.chem_comp.pdbx_formal_charge")
        return EffectContract(
            required_paths=tuple(dict.fromkeys(paths)),
            description="Chemical component definitions expose the requested raw CCD descriptors.",
        )

    @staticmethod
    def _query_text(
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> str:
        return " ".join(
            [
                request.research_goal,
                intent.capability_query,
                request.step.objective,
                *request.step.expected_outputs,
            ]
        ).casefold()

    @classmethod
    def _pdb_entity_reference(
        cls,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> tuple[str, str] | None:
        identifiers = cls._pdb_identifiers(request, intent)
        if not identifiers:
            return None
        raw_entity = (
            request.step.inputs.get("entity_id")
            or request.step.inputs.get("pdb_entity_id")
        )
        entity = str(raw_entity).strip() if raw_entity is not None else ""
        if not entity:
            source = " ".join(
                [request.step.objective, intent.capability_query, request.research_goal]
            )
            match = re.search(r"\bentity(?:\s+id)?\s*[:#]?\s*(\d+)\b", source, re.I)
            entity = match.group(1) if match else ""
        if not re.fullmatch(r"\d+", entity):
            return None
        return identifiers[0], entity

    @classmethod
    def _requires_state_transformation(cls, query: str) -> bool:
        return any(marker in query for marker in cls._STATE_TRANSFORMATION_MARKERS)

    @staticmethod
    def _pdb_identifiers(
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> list[str]:
        values = request.step.inputs.get("pdb_ids")
        if not isinstance(values, list):
            values = request.step.inputs.get("identifiers")
        if isinstance(values, list):
            explicit = [
                str(item).strip().upper()
                for item in values
                if re.fullmatch(r"[0-9][A-Z][A-Z0-9]{2}", str(item).strip().upper())
            ]
            if explicit:
                return list(dict.fromkeys(explicit))
        source = " ".join(
            [request.step.objective, intent.capability_query, request.research_goal]
        ).upper()
        return list(
            dict.fromkeys(
                re.findall(r"(?<![A-Z0-9])[0-9][A-Z][A-Z0-9]{2}(?![A-Z0-9])", source)
            )
        )
