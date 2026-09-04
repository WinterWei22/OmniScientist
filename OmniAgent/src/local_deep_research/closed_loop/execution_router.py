from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .capability_binding import CapabilityBindingRegistry
from .capability_resolver import CapabilityCatalogResolver
from .action_ledger import canonical_action_key
from .answer_validation import infer_answer_semantic_contract
from .contracts import A1TaskResult
from .execution_models import (
    BoundCapabilityCall,
    EffectContract,
    ExecutionBackend,
    PathValueRequirement,
    ResourceCandidate,
    RouteDecision,
    SemanticCapabilityIntent,
    stage_request_id,
)
from .execution_validation import (
    validate_schema_instance,
    verify_effects,
)
from .failure_policy import execution_failure_retryable
from .mcp_workflow import (
    BoundedMCPWorkflowExecutor,
    CapabilityWorkflowRegistry,
    WorkflowBindingError,
    workflow_from_record,
)
from .scientific_workflow import (
    project_workflow_evidence,
    workflow_execution_instruction,
)
from .routing_policy import RoutingMode, RoutingPolicy
from .task_metadata import external_task_summary as _external_task_summary

if TYPE_CHECKING:
    from .contracts import A1TaskRequest


EventSink = Callable[[str, dict[str, Any]], None]

A1_HISTORY_MAX_ITEMS = 8
A1_HISTORY_MAX_RECORD_CHARS = 1200


def _public_domain_workflow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluator-private inputs before a workflow crosses into A1."""
    public = deepcopy(payload)
    contract = public.get("verification_contract", {})
    private_keys = (
        contract.get("private_input_keys", []) if isinstance(contract, dict) else []
    )
    private = {str(item).strip() for item in private_keys if str(item).strip()}
    workflow_inputs = public.get("inputs")
    if isinstance(workflow_inputs, dict):
        for key in private:
            workflow_inputs.pop(key, None)
    return public


def _tool_trace_summary(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    fields = (
        "event",
        "backend",
        "workflow_id",
        "workflow_step_id",
        "tool_name",
        "task_id",
        "request_id",
        "success",
        "result_status",
    )
    return [
        {key: item[key] for key in fields if key in item}
        for item in trace
        if isinstance(item, dict)
    ]


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    """Declarative evidence needed before a bounded MCP route is allowed."""

    reason_code: str
    tool_suffixes: tuple[str, ...]
    keyword_groups: tuple[tuple[str, ...], ...]
    search_terms: str
    rationale: str

    def matches(self, text: str) -> bool:
        return all(
            any(keyword in text for keyword in group)
            for group in self.keyword_groups
        )

    def find_candidate(
        self, candidates: list[ResourceCandidate]
    ) -> ResourceCandidate | None:
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.qualified_name.endswith(self.tool_suffixes)
            ),
            None,
        )


CAPABILITY_RULES = (
    CapabilityRule(
        reason_code="schema_bound_structure_lookup",
        tool_suffixes=(".query_pdb", ".query_pdb_identifiers"),
        keyword_groups=(("pdb", "protein data bank", "solution nmr", "nmr structure"),),
        search_terms="RCSB PDB structure query identifiers experimental method organism",
        rationale="A retrieved PDB tool can perform this constrained structure lookup directly.",
    ),
    CapabilityRule(
        reason_code="schema_bound_disease_target_lookup",
        tool_suffixes=(".query_opentarget_disease_targets",),
        keyword_groups=(
            ("disease",),
            ("target", "gene"),
            ("association", "associated", "rank", "top"),
        ),
        search_terms="OpenTargets Platform disease target association ranked lookup",
        rationale="A retrieved OpenTargets tool can perform this ranked disease-target lookup directly.",
    ),
    CapabilityRule(
        reason_code="schema_bound_interaction_lookup",
        tool_suffixes=(".query_stringdb_top_interactor",),
        keyword_groups=(("interact",), ("score", "confidence")),
        search_terms="STRING protein-protein interaction top interactor confidence score query_stringdb_top_interactor",
        rationale="A retrieved STRING tool can select the highest-scoring interaction directly.",
    ),
)


class A1ExecutionBackend:
    """Run one bounded experiment through Biomni A1 with compact MCP hints."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    async def initialize(self, *, load_catalog: bool = True) -> None:
        """Initialize A1 while supporting old tool adapters without catalog control."""
        initialize = self.tool.initialize
        try:
            parameters = inspect.signature(initialize).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_catalog = any(
            parameter.name == "load_catalog"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_catalog:
            await initialize(load_catalog=load_catalog)
        else:
            await initialize()

    async def run(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> A1TaskResult:
        inputs = dict(request.step.inputs)
        if decision.domain_workflow:
            public_workflow = _public_domain_workflow_payload(decision.domain_workflow)
            contract = decision.domain_workflow.get("verification_contract", {})
            private_keys = (
                contract.get("private_input_keys", [])
                if isinstance(contract, dict)
                else []
            )
            for key in private_keys:
                inputs.pop(str(key), None)
            inputs["domain_workflow"] = public_workflow
            mcp_evidence = inputs.pop("domain_workflow_evidence", None)
            inputs["domain_workflow_instruction"] = workflow_execution_instruction(
                public_workflow,
                mcp_evidence=mcp_evidence,
            )
        # ``result_contract`` is a provider option, not the OmniAgent domain
        # evidence contract.  The latter is verified locally after Biomni
        # returns, so never send its required paths to the remote worker.
        # An explicitly namespaced Biomni contract remains available for a
        # capability that advertises and owns such a contract.
        inputs.pop("result_contract", None)
        inputs["retrieved_biomni_capabilities"] = [
            {
                "qualified_name": item.qualified_name,
                "description": item.description[:240],
                "score": item.score,
            }
            for item in decision.candidates[:3]
        ]
        step = replace(
            request.step,
            objective=self._compact_text(request.step.objective, 1200),
            inputs=inputs,
            constraints=self._compact_list(request.step.constraints, 6, 500),
            expected_outputs=self._compact_list(request.step.expected_outputs, 5, 500),
            success_criteria=self._compact_list(request.step.success_criteria, 5, 500),
        )
        compact_request = replace(
            request,
            step=step,
            research_goal=self._compact_text(request.research_goal, 1600),
            hypothesis=self._compact_text(request.hypothesis, 1000),
            global_constraints=self._compact_list(request.global_constraints, 8, 500),
            prior_observations=self._compact_history(
                request.prior_observations,
                max_items=A1_HISTORY_MAX_ITEMS,
                max_record_chars=A1_HISTORY_MAX_RECORD_CHARS,
            ),
            prior_evaluations=self._compact_history(
                request.prior_evaluations,
                max_items=A1_HISTORY_MAX_ITEMS,
                max_record_chars=A1_HISTORY_MAX_RECORD_CHARS,
            ),
        )
        result = await self.tool.run(compact_request)
        result.tool_trace.insert(
            0,
            {
                "event": "execution_backend_called",
                "backend": ExecutionBackend.A1.value,
                "gateway_tool": "call_biomni",
                "route_reason": decision.reason_code,
                "semantic_intent": (
                    decision.semantic_intent.to_dict()
                    if decision.semantic_intent
                    else None
                ),
                "effective_prompt_chars": len(compact_request.to_prompt()),
                "effective_prompt_estimated_tokens": max(
                    1, len(compact_request.to_prompt()) // 4
                ),
                "domain_workflow": decision.domain_workflow.get("qualified_id")
                if decision.domain_workflow
                else None,
                "biomni_result_contract_forwarded": bool(
                    inputs.get("biomni_result_contract")
                ),
            },
        )
        return result

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        return await self.tool.poll(request, task_metadata)

    def capability_admission_error(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent | None,
    ) -> str | None:
        gateway = getattr(self.tool, "gateway", None)
        if gateway is None:
            return None
        operation = intent.operation.value if intent is not None else "adaptive"
        return gateway.a1_capability_admission_error(
            operation=operation,
            requires_method_provenance=(
                infer_answer_semantic_contract(request.research_goal) is not None
            ),
        )

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    @classmethod
    def _compact_list(
        cls,
        values: list[Any],
        max_items: int,
        item_limit: int,
    ) -> list[str]:
        return [cls._compact_text(item, item_limit) for item in values[:max_items]]

    @staticmethod
    def _compact_history(
        records: list[dict[str, Any]],
        *,
        max_items: int,
        max_record_chars: int,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for record in records[-max_items:]:
            encoded = json.dumps(record, ensure_ascii=False, default=str)
            if len(encoded) > max_record_chars:
                encoded = encoded[:max_record_chars] + "...[truncated]"
            compact.append({"summary": encoded})
        return compact


class LayeredMCPExecutionBackend:
    """Use layered MCP for compact capability discovery and exact invocation."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    async def initialize(self) -> None:
        await self.tool.initialize()

    async def inspect(
        self,
        request: A1TaskRequest,
        *,
        workflow_spec: Any | None = None,
    ) -> tuple[str, list[ResourceCandidate], dict[str, Any]]:
        query = self._expand_query(request, self.tool._build_search_query(request))
        payload = await self.tool.search_capabilities(query)
        matches = self.tool._extract_matches(payload)
        candidates: list[ResourceCandidate] = []
        self._append_records(candidates, matches[: self.tool.max_results])
        if workflow_spec is not None:
            await self._append_workflow_candidates(
                workflow_spec,
                candidates,
                primary_query=query,
            )
        metadata = {
            key: payload.get(key)
            for key in ("catalog_size", "catalog_protocol", "catalog_revision", "retrieval")
            if isinstance(payload, dict) and payload.get(key) is not None
        }
        return query, candidates, metadata

    async def _append_workflow_candidates(
        self,
        workflow_spec: Any,
        candidates: list[ResourceCandidate],
        *,
        primary_query: str,
    ) -> None:
        searched = {primary_query.casefold().strip()}
        gateway = getattr(self.tool, "gateway", None)
        get_descriptor = getattr(gateway, "get_capability_descriptor", None)
        for node in getattr(workflow_spec, "nodes", ()):
            if str(getattr(node, "executor", "")) != "mcp":
                continue
            capability_id = str(getattr(node, "capability_id", "") or "").strip()
            if capability_id and self._has_candidate(candidates, capability_id):
                continue
            if capability_id and callable(get_descriptor):
                descriptor = get_descriptor(capability_id)
                if descriptor is not None:
                    self._append_records(candidates, [descriptor.to_record()])
                    continue
            capability_query = str(
                getattr(node, "capability_query", "") or capability_id
            ).strip()
            normalized_query = capability_query.casefold()
            if not capability_query or normalized_query in searched:
                continue
            searched.add(normalized_query)
            payload = await self.tool.search_capabilities(capability_query)
            matches = self.tool._extract_matches(payload)
            self._append_records(candidates, matches[: self.tool.max_results])

    @classmethod
    def _append_records(
        cls,
        candidates: list[ResourceCandidate],
        records: Any,
    ) -> None:
        for record in records:
            if not isinstance(record, dict):
                continue
            candidate = cls._candidate_from_record(record)
            if candidate.qualified_name and not cls._has_candidate(
                candidates, candidate.qualified_name
            ):
                candidates.append(candidate)

    @staticmethod
    def _has_candidate(candidates: list[ResourceCandidate], qualified_name: str) -> bool:
        return any(item.qualified_name == qualified_name for item in candidates)

    @staticmethod
    def _candidate_from_record(item: dict[str, Any]) -> ResourceCandidate:
        def mapping(key: str) -> dict[str, Any]:
            value = item.get(key, {})
            return value if isinstance(value, dict) else {}

        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, int | float) else None
        return ResourceCandidate(
            qualified_name=str(
                item.get("qualified_name") or item.get("canonical_name") or item.get("name") or ""
            ),
            description=str(item.get("description") or "")[:800],
            score=score,
            input_schema=mapping("input_schema"),
            output_schema=mapping("output_schema"),
            capability_version=str(item.get("capability_version") or ""),
            effect_contract=mapping("effect_contract"),
            result_adapter=str(item.get("result_adapter") or "generic"),
            execution_mode=str(item.get("execution_mode") or "sync"),
            lifecycle=mapping("lifecycle"),
            retry_policy=mapping("retry_policy"),
            timeout_policy=mapping("timeout_policy"),
            idempotency_policy=mapping("idempotency_policy"),
            provenance_policy=mapping("provenance_policy"),
        )

    async def run(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> A1TaskResult:
        if decision.bound_workflow is not None:
            result = await BoundedMCPWorkflowExecutor(self.tool).run(
                request, decision.bound_workflow
            )
            result.tool_trace.insert(
                0,
                {
                    "event": "execution_backend_called",
                    "backend": ExecutionBackend.MCP.value,
                    "gateway_tools": ["biomni_search_tools", "biomni_invoke_tool"],
                    "route_reason": decision.reason_code,
                    "workflow_id": decision.bound_workflow.workflow_id,
                },
            )
            return result
        if decision.bound_call is None:
            return A1TaskResult(
                success=False,
                result_status="capability_binding_missing",
                errors=["MCP execution requires a validated BoundCapabilityCall"],
            )
        inputs = dict(request.step.inputs)
        inputs["tool_query"] = decision.query
        inputs["tool_name"] = decision.bound_call.tool_name
        inputs["arguments"] = dict(decision.bound_call.arguments)
        constraints = list(request.step.constraints)
        if "opentargets" in decision.query.lower():
            constraints.append(
                "For API tools that accept a direct structured query, prefer that "
                "argument over a nested natural-language LLM translation."
            )
        step = replace(request.step, inputs=inputs, constraints=constraints)
        bound_request = replace(request, step=step)
        direct_bound_call = getattr(self.tool, "invoke_bound_call", None)
        if direct_bound_call is not None:
            result = await direct_bound_call(
                bound_request,
                tool_name=decision.bound_call.tool_name,
                arguments=dict(decision.bound_call.arguments),
                input_schema=decision.bound_call.input_schema,
                output_schema=decision.bound_call.output_schema,
                wait_for_terminal=False,
            )
        else:
            result = await self.tool.run(bound_request)
        metadata = dict(result.task_metadata) if isinstance(result.task_metadata, dict) else {}
        metadata["bound_call"] = decision.bound_call.to_dict()
        result.task_metadata = metadata
        result.tool_trace.insert(
            0,
            {
                "event": "execution_backend_called",
                "backend": ExecutionBackend.MCP.value,
                "gateway_tools": ["biomni_search_tools", "biomni_invoke_tool"],
                "route_reason": decision.reason_code,
                "admitted_capability": decision.admitted_capability,
                "selected_capability": decision.selected_capability,
            },
        )
        if result.result_status == "task_pending":
            return result
        payload = (
            result.verification_payload
            if result.verification_payload is not None
            else self._result_payload(result.raw)
        )
        output_errors = validate_schema_instance(
            payload, decision.bound_call.output_schema
        )
        effect_verification = verify_effects(
            payload,
            decision.bound_call.effects,
            artifacts=result.artifacts,
            allowed_paths=request.allowed_paths,
        )
        metadata["output_schema_errors"] = output_errors
        metadata["effect_verification"] = effect_verification.to_dict()
        result.task_metadata = metadata
        if output_errors or not effect_verification.passed:
            result.success = False
            result.result_status = "effect_verification_failed"
            reason = (
                "; ".join(output_errors[:3])
                if output_errors
                else "MCP result did not satisfy the bound effect contract"
            )
            if reason not in result.errors:
                result.errors.append(reason)
        return result

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        result = await self.tool.poll(request, task_metadata)
        route = task_metadata.get("omniagent_route", {})
        bound_call = route.get("bound_call") if isinstance(route, dict) else None
        if result.result_status == "task_pending":
            return result
        bound_workflow = route.get("bound_workflow") if isinstance(route, dict) else None
        workflow_resume = task_metadata.get("workflow_resume")
        if isinstance(bound_workflow, dict) and isinstance(workflow_resume, dict):
            try:
                workflow = workflow_from_record(bound_workflow)
            except WorkflowBindingError as exc:
                return A1TaskResult(
                    success=False,
                    result_status="invalid_workflow_checkpoint",
                    errors=[str(exc)],
                    task_metadata=dict(result.task_metadata),
                    tool_trace=list(result.tool_trace),
                )
            return await BoundedMCPWorkflowExecutor(self.tool).resume(
                request,
                workflow,
                result,
                workflow_resume,
            )
        if not isinstance(bound_call, dict):
            return result
        output_schema = bound_call.get("output_schema", {})
        effects_raw = bound_call.get("effects", {})
        effects = EffectContract(
            required_paths=tuple(effects_raw.get("required_paths", [])),
            any_of_paths=tuple(effects_raw.get("any_of_paths", [])),
            required_value_matches=tuple(
                PathValueRequirement(
                    path=str(item.get("path", "")),
                    expected_values=tuple(
                        str(value)
                        for value in item.get("expected_values", [])
                    ),
                    case_sensitive=bool(item.get("case_sensitive", False)),
                )
                for item in effects_raw.get("required_value_matches", [])
                if isinstance(item, dict)
                and str(item.get("path", "")).strip()
                and isinstance(item.get("expected_values"), list | tuple)
            ),
            required_artifacts=tuple(effects_raw.get("required_artifacts", [])),
            description=str(effects_raw.get("description", "")),
        ) if isinstance(effects_raw, dict) else EffectContract()
        payload = (
            result.verification_payload
            if result.verification_payload is not None
            else self._result_payload(result.raw)
        )
        output_errors = validate_schema_instance(payload, output_schema)
        effect_verification = verify_effects(
            payload,
            effects,
            artifacts=result.artifacts,
            allowed_paths=request.allowed_paths,
        )
        metadata = dict(result.task_metadata) if isinstance(result.task_metadata, dict) else {}
        metadata["bound_call"] = bound_call
        metadata["output_schema_errors"] = output_errors
        metadata["effect_verification"] = effect_verification.to_dict()
        result.task_metadata = metadata
        if output_errors or not effect_verification.passed:
            result.success = False
            result.result_status = "effect_verification_failed"
            result.errors.append(
                "; ".join(output_errors[:3])
                if output_errors
                else "MCP result did not satisfy the bound effect contract"
            )
        return result

    async def bind(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> None:
        if decision.bound_workflow is not None:
            decision.selected_capability = decision.bound_workflow.workflow_id
            return
        if decision.reason_code == "schema_parameters_required":
            records = [
                {
                    "name": item.qualified_name.rsplit(".", 1)[-1],
                    "qualified_name": item.qualified_name,
                    "description": item.description,
                    "input_schema": item.input_schema,
                    "output_schema": item.output_schema,
                    "effect_contract": item.effect_contract,
                    "capability_version": item.capability_version,
                    "result_adapter": item.result_adapter,
                    "execution_mode": item.execution_mode,
                    "score": item.score,
                }
                for item in decision.candidates[:3]
                if item.input_schema
            ]
            parameterization = await self.tool.parameterize_schema_bound_request(
                request, records
            )
            selected_name = str(parameterization["tool_name"])
            candidate = next(
                (
                    item
                    for item in decision.candidates
                    if item.qualified_name == selected_name
                ),
                None,
            )
            if candidate is None:
                raise ValueError("Selected capability is absent from the retrieved catalog")
            arguments = parameterization["arguments"]
            if not isinstance(arguments, dict):
                raise ValueError("Selected capability arguments are not an object")
            intent = decision.semantic_intent
            if intent is None:
                raise ValueError("Schema parameterization requires a semantic intent")
            argument_errors = validate_schema_instance(
                arguments, candidate.input_schema, strict_objects=True
            )
            if argument_errors:
                raise ValueError("; ".join(argument_errors[:3]))
            effects = EffectContract.from_intent(intent)
            decision.reason_code = "schema_parameterized"
            decision.rationale = (
                "Qwen selected a retrieved MCP capability and filled arguments after "
                "its input schema was disclosed."
            )
            decision.admitted_capability = candidate.qualified_name
            decision.selected_capability = candidate.qualified_name
            decision.binding_id = "catalog.schema_parameterized.v1"
            decision.bound_call = BoundCapabilityCall(
                tool_name=candidate.qualified_name,
                arguments=arguments,
                input_schema=candidate.input_schema,
                output_schema=candidate.output_schema,
                effects=effects,
                binding_reason=decision.reason_code,
                capability_version=candidate.capability_version,
                result_adapter=candidate.result_adapter,
                execution_mode=candidate.execution_mode,
                lifecycle=dict(candidate.lifecycle),
                retry_policy=dict(candidate.retry_policy),
                timeout_policy=dict(candidate.timeout_policy),
                idempotency_policy=dict(candidate.idempotency_policy),
                provenance_policy=dict(candidate.provenance_policy),
            )
            decision.parameterization = {
                key: value
                for key, value in parameterization.items()
                if key != "candidate"
            }
            decision.condition_coverage = decision.condition_coverage.__class__(
                required_conditions=decision.condition_coverage.required_conditions,
                binding_covered_conditions=(
                    "catalog_candidate_schema",
                    f"catalog_capability:{candidate.qualified_name}",
                    "input_schema",
                ),
                verification_required_conditions=tuple(
                    f"field:{value}" for value in effects.required_paths
                ) + tuple(f"artifact:{value}" for value in effects.required_artifacts),
            )
            decision.output_contract = {
                **decision.output_contract,
                "output_schema": candidate.output_schema,
                "effects": effects.to_dict(),
            }
            return
        candidate: ResourceCandidate | None = None
        arguments: dict[str, Any] | None = None
        requested = str(
            request.step.inputs.get("tool_name")
            or request.step.inputs.get("tool_query")
            or ""
        ).strip()
        explicit_arguments = request.step.inputs.get("arguments")
        if isinstance(explicit_arguments, dict):
            arguments = dict(explicit_arguments)
        admitted = decision.admitted_capability

        if decision.reason_code == "schema_bound_structure_lookup":
            pdb_ids = HybridExecutionRouter._pdb_identifiers(request)
            candidate = next(
                (
                    item
                    for item in decision.candidates
                    if item.qualified_name.endswith(".query_pdb_identifiers") and pdb_ids
                ),
                next(
                    (
                        item
                        for item in decision.candidates
                        if item.qualified_name.endswith(".query_pdb")
                    ),
                    None,
                ),
            )
            if candidate is not None and arguments is None:
                raw_limit = request.step.inputs.get("limit", 5)
                try:
                    max_results = max(1, min(int(raw_limit), 50))
                except (TypeError, ValueError):
                    max_results = 5
                if candidate.qualified_name.endswith(".query_pdb_identifiers"):
                    attributes = request.step.inputs.get("fields")
                    if not isinstance(attributes, list):
                        attributes = request.step.inputs.get("attributes")
                    arguments = {
                        "identifiers": pdb_ids,
                        "attributes": attributes or [],
                        "return_type": str(
                            request.step.inputs.get("return_type", "entry")
                        ),
                    }
                else:
                    structured = HybridExecutionRouter._pdb_structured_query(
                        request, max_results
                    )
                    arguments = (
                        {"query": structured, "max_results": max_results}
                        if structured is not None
                        else {"prompt": decision.query, "max_results": max_results}
                    )
        elif decision.reason_code == "schema_bound_disease_target_lookup":
            candidate = next(
                (
                    item
                    for item in decision.candidates
                    if item.qualified_name.endswith(
                        ".query_opentarget_disease_targets"
                    )
                ),
                None,
            )
            disease = HybridExecutionRouter._disease_lookup_value(request)
            if candidate is not None and arguments is None and disease is not None:
                raw_limit = request.step.inputs.get("limit", 5)
                try:
                    max_results = max(1, min(int(raw_limit), 100))
                except (TypeError, ValueError):
                    max_results = 5
                arguments = {"disease": disease, "max_results": max_results}
        elif decision.reason_code == "schema_bound_interaction_lookup":
            candidate = next(
                (
                    item
                    for item in decision.candidates
                    if item.qualified_name.endswith(".query_stringdb_top_interactor")
                ),
                None,
            )
            gene = HybridExecutionRouter._interaction_gene_value(request)
            if candidate is not None and arguments is None and gene is not None:
                arguments = {
                    "gene_symbol": gene,
                    "species": 9606,
                    "max_results": 100,
                }
        else:
            target = admitted or requested
            candidate = next(
                (item for item in decision.candidates if item.qualified_name == target),
                None,
            )

        if candidate is None or arguments is None:
            raise ValueError("No exact candidate and complete argument object could be bound")
        argument_errors = validate_schema_instance(
            arguments, candidate.input_schema, strict_objects=True
        )
        if argument_errors:
            raise ValueError("; ".join(argument_errors[:3]))
        decision.selected_capability = candidate.qualified_name
        decision.bound_call = BoundCapabilityCall(
            tool_name=candidate.qualified_name,
            arguments=arguments,
            input_schema=candidate.input_schema,
            output_schema=candidate.output_schema,
            effects=EffectContract.from_intent(decision.semantic_intent),
            binding_reason=decision.reason_code,
            capability_version=candidate.capability_version,
            result_adapter=candidate.result_adapter,
            execution_mode=candidate.execution_mode,
            lifecycle=dict(candidate.lifecycle),
            retry_policy=dict(candidate.retry_policy),
            timeout_policy=dict(candidate.timeout_policy),
            idempotency_policy=dict(candidate.idempotency_policy),
            provenance_policy=dict(candidate.provenance_policy),
        )

    @staticmethod
    def _result_payload(raw: Any) -> Any:
        if isinstance(raw, dict) and "result" in raw and (
            "ok" in raw or "task_id" in raw or "task_type" in raw
        ):
            return raw["result"]
        return raw

    @staticmethod
    def _expand_query(request: A1TaskRequest, query: str) -> str:
        """Add domain capability terms without naming a benchmark or fixing a result."""
        semantic_intent = SemanticCapabilityIntent.from_inputs(request.step.inputs)
        if semantic_intent is not None:
            return semantic_intent.capability_query[:1200]
        text = " ".join(
            [
                request.step.objective,
                query,
                *request.step.expected_outputs,
                *request.step.constraints,
                *request.step.success_criteria,
            ]
        ).lower()
        expansions: list[str] = []
        for rule in CAPABILITY_RULES:
            if rule.matches(text):
                expansions.append(rule.search_terms)
        if "pharmacophore" in text and ("smiles" in text or ".smi" in text):
            expansions.append("RDKit Python read SMILES write Python artifact")
        if not expansions:
            return query[:1200]
        return (query + " " + " ".join(expansions))[:1200]


class HybridExecutionRouter:
    """Deterministically route one OmniAgent step to A1 or layered MCP.

    MCP retrieval always acts as a compact planning-time capability check. The selected
    execution backend is A1 for open-ended/multi-tool work and MCP for exact,
    schema-bound or confirmatory work. Decisions are auditable and require no extra LLM.
    """

    _A1_MARKERS = (
        "analy",
        "investigat",
        "explor",
        "design",
        "screen",
        "prioriti",
        "rank",
        "compare",
        "integrat",
        "multi-step",
        "workflow",
        "interpret",
        "recommend",
    )
    _MCP_MARKERS = (
        "verify",
        "validate",
        "confirm",
        "check",
        "inspect",
        "read file",
        "write file",
        "execute python",
        "run script",
        "convert",
        "calculate",
        "generate json",
    )

    def __init__(
        self,
        *,
        a1_backend: A1ExecutionBackend,
        mcp_backend: LayeredMCPExecutionBackend,
        resolver: CapabilityCatalogResolver | None = None,
        workflow_registry: CapabilityWorkflowRegistry | None = None,
        binding_registry: CapabilityBindingRegistry | None = None,
        domain_workflow_registry: Any | None = None,
        routing_policy: RoutingPolicy | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.a1_backend = a1_backend
        self.mcp_backend = mcp_backend
        self.resolver = resolver or CapabilityCatalogResolver()
        self.workflow_registry = workflow_registry or CapabilityWorkflowRegistry()
        self.routing_policy = routing_policy or RoutingPolicy.from_environment()
        self.binding_registry = binding_registry or CapabilityBindingRegistry(
            resolver=self.resolver,
            workflow_registry=self.workflow_registry,
            domain_workflow_registry=domain_workflow_registry,
            routing_policy=self.routing_policy,
        )
        self.event_sink = event_sink or (lambda _event, _payload: None)
        self.route_history: list[dict[str, Any]] = []
        self._last_backend: ExecutionBackend | None = None
        self._last_success: bool | None = None
        self._failure_counts: dict[str, int] = {}
        self._failure_limits: dict[str, int] = {}

    @property
    def exposed_tool_names(self) -> tuple[str, ...]:
        return ("call_biomni", "biomni_search_tools", "biomni_invoke_tool")

    async def initialize(self) -> None:
        if self.routing_policy.mode is RoutingMode.A1_ONLY:
            await self.a1_backend.initialize(load_catalog=False)
            return
        await self.mcp_backend.initialize()
        await self.a1_backend.initialize(load_catalog=True)

    async def run(self, request: A1TaskRequest) -> A1TaskResult:
        decision = await self.prepare(request)
        return await self.dispatch(request, decision)

    async def bind(self, request: A1TaskRequest) -> RouteDecision:
        """Expose prepare so the Harness can ledger before dispatch."""
        return await self.prepare(request)

    async def prepare(self, request: A1TaskRequest) -> RouteDecision:
        """Resolve and bind a request without dispatching an external task."""
        semantic_intent = SemanticCapabilityIntent.from_inputs(request.step.inputs)
        raw_requested_backend = str(
            request.step.inputs.get("execution_backend", "")
        ).strip().lower()
        query = (
            semantic_intent.capability_query
            if semantic_intent is not None
            else str(request.step.inputs.get("tool_query", ""))
        )[:1200]
        candidates: list[ResourceCandidate] = []
        metadata: dict[str, Any] = {}
        domain_workflow_spec = None
        resolve_domain_workflow = getattr(
            self.binding_registry, "resolve_domain_workflow", None
        )
        if semantic_intent is not None and callable(resolve_domain_workflow):
            domain_workflow_spec = resolve_domain_workflow(
                request, semantic_intent
            )
        if raw_requested_backend and raw_requested_backend not in {
            item.value for item in ExecutionBackend
        }:
            decision = RouteDecision(
                backend=ExecutionBackend.UNAVAILABLE,
                reason_code="planner_invalid_backend",
                rationale=(
                    f"Planner supplied unsupported execution_backend {raw_requested_backend!r}; "
                    "the Harness must bind the backend from capability evidence."
                ),
                query=query,
                candidates=[],
                semantic_intent=semantic_intent,
            )
            decision.requested_backend = raw_requested_backend
            decision.route_signature = self._route_signature(request, decision)
            decision.execution_payload = self._execution_payload(request, decision)
            decision.execution_signature = self._execution_signature(request, decision)
            decision.idempotency_key = self._idempotency_key(request, decision)
            self.event_sink(
                "execution_route_decided",
                {"iteration": request.iteration, "step_id": request.step.step_id, **decision.to_dict()},
            )
            return decision
        # A1 is self-contained; MCP needs capability retrieval before a safe
        # structured invocation can be selected and schema-checked.
        explicit_direct = bool(
            request.step.inputs.get("tool_name")
            and isinstance(request.step.inputs.get("arguments"), dict)
            and self.routing_policy.profile_for(
                str(request.step.inputs.get("tool_name"))
            ) is not None
        )
        if self.routing_policy.mode is RoutingMode.A1_ONLY:
            should_inspect = False
        else:
            should_inspect = self.routing_policy.should_inspect(
                semantic_intent,
                workflow_spec=domain_workflow_spec,
            ) or explicit_direct
        if should_inspect:
            try:
                query, candidates, metadata = await self.mcp_backend.inspect(
                    request,
                    workflow_spec=domain_workflow_spec,
                )
                self.event_sink(
                    "biomni_capabilities_retrieved",
                    {
                        "iteration": request.iteration,
                        "step_id": request.step.step_id,
                        "query": query,
                        "catalog_size": metadata.get("catalog_size"),
                        "candidates": [item.to_dict() for item in candidates],
                    },
                )
            except Exception as exc:
                self.event_sink(
                    "biomni_capability_retrieval_failed",
                    {
                        "iteration": request.iteration,
                        "step_id": request.step.step_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        candidates = self.routing_policy.filter_candidates(candidates)
        if semantic_intent is not None:
            decision = self.binding_registry.bind(
                request,
                semantic_intent,
                candidates=candidates,
                catalog_revision=str(metadata.get("catalog_revision") or ""),
                previous_backend=(self._last_backend.value if self._last_backend else ""),
            )
        else:
            decision = self.decide(request, query=query, candidates=candidates)
            decision.catalog_revision = str(metadata.get("catalog_revision") or "")
        decision.requested_backend = raw_requested_backend or "unspecified"
        if (
            self.routing_policy.mode is RoutingMode.A1_ONLY
            and raw_requested_backend == ExecutionBackend.MCP.value
        ):
            decision.policy_metadata = {
                **decision.policy_metadata,
                "backend_override": {
                    "requested": raw_requested_backend,
                    "effective": decision.backend.value,
                    "reason": "routing_policy_a1_only",
                    "policy_version": self.routing_policy.version,
                },
            }
            self.event_sink(
                "execution_backend_overridden",
                {
                    "iteration": request.iteration,
                    "step_id": request.step.step_id,
                    "requested_backend": raw_requested_backend,
                    "effective_backend": decision.backend.value,
                    "reason": "routing_policy_a1_only",
                    "policy_version": self.routing_policy.version,
                },
            )
        if (
            decision.backend is ExecutionBackend.MCP
            and decision.bound_call is None
            and decision.bound_workflow is None
        ):
            try:
                await self.mcp_backend.bind(request, decision)
            except ValueError as exc:
                decision = self._fallback_decision(
                    decision,
                    reason_code="mcp_binding_fallback_to_a1",
                    rationale=str(exc),
                )
        if decision.backend is ExecutionBackend.MCP:
            policy_errors: list[str] = []
            if decision.bound_call is not None:
                policy_errors = self.routing_policy.authorize_call(decision.bound_call)
            elif decision.bound_workflow is not None:
                policy_errors = self.routing_policy.authorize_workflow(
                    decision.bound_workflow,
                    domain_workflow=bool(domain_workflow_spec),
                )
            if policy_errors:
                decision = self._fallback_decision(
                    decision,
                    reason_code="mcp_policy_fallback_to_a1",
                    rationale="; ".join(policy_errors[:4]),
                )
        if decision.backend is ExecutionBackend.A1:
            admission = getattr(self.a1_backend, "capability_admission_error", None)
            admission_error = (
                admission(request, semantic_intent) if callable(admission) else None
            )
            if admission_error:
                decision.backend = ExecutionBackend.UNAVAILABLE
                decision.reason_code = "a1_capability_unavailable"
                decision.rationale = admission_error
        decision.route_signature = self._route_signature(request, decision)
        decision.execution_payload = self._execution_payload(request, decision)
        decision.execution_signature = self._execution_signature(request, decision)
        decision.idempotency_key = self._idempotency_key(request, decision)
        retry_key = self._retry_key(decision)
        failure_count = max(
            self._prior_failure_count(request, retry_key),
            self._prior_failure_count(request, decision.route_signature),
        )
        failure_limit = self._failure_limits.get(
            retry_key, self._failure_limits.get(decision.route_signature, 2)
        )
        if failure_count >= failure_limit:
            decision.retry_blocked = True
        self.event_sink(
            "execution_route_decided",
            {
                "iteration": request.iteration,
                "step_id": request.step.step_id,
                **decision.to_dict(),
            },
        )
        return decision

    async def dispatch(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> A1TaskResult:
        """Execute a prepared binding and persist its retry decision in the result."""
        retry_key = self._retry_key(decision)
        failure_count = max(
            self._prior_failure_count(request, retry_key),
            self._prior_failure_count(request, decision.route_signature),
        )
        failure_limit = self._failure_limits.get(
            retry_key, self._failure_limits.get(decision.route_signature, 2)
        )
        execution_invoked = (
            not decision.retry_blocked
            and decision.backend is not ExecutionBackend.UNAVAILABLE
        )
        if decision.retry_blocked:
            result = A1TaskResult(
                success=False,
                result_status="duplicate_route_blocked",
                errors=[
                    "Repeated execution fingerprint blocked after prior failures; change "
                    "the capability, validated arguments, expected effect, or backend."
                ],
                tool_trace=[
                    {
                        "event": "execution_route_blocked",
                        "backend": decision.backend.value,
                        "route_signature": decision.route_signature,
                        "prior_failure_count": failure_count,
                    }
                ],
            )
        elif decision.backend is ExecutionBackend.UNAVAILABLE:
            result = A1TaskResult(
                success=False,
                result_status="capability_unavailable",
                errors=[decision.rationale],
                tool_trace=[
                    {
                        "event": "execution_capability_unavailable",
                        "backend": ExecutionBackend.UNAVAILABLE.value,
                        "route_signature": decision.route_signature,
                        "reason_code": decision.reason_code,
                    }
                ],
            )
        else:
            if (
                decision.backend is ExecutionBackend.A1
                and decision.domain_workflow
                and decision.bound_workflow is not None
                and self.routing_policy.mode is not RoutingMode.A1_ONLY
            ):
                result = await self._run_mixed_domain_workflow(request, decision)
                if (
                    not result.success
                    and self._should_fallback_mcp_failure(result)
                ):
                    mcp_decision = replace(
                        decision,
                        backend=ExecutionBackend.MCP,
                        reason_code="domain_workflow_mcp_stage",
                    )
                    result = await self._dispatch_mcp_fallback_to_a1(
                        request, mcp_decision, result
                    )
                    decision = self._fallback_decision(
                        decision,
                        reason_code="mcp_execution_fallback_to_a1",
                        rationale=(
                            "The domain workflow MCP stage reached a terminal failure; "
                            "A1 owns the complete workflow recovery."
                        ),
                    )
            else:
                backend = (
                    self.a1_backend
                    if decision.backend is ExecutionBackend.A1
                    else self.mcp_backend
                )
                result = await backend.run(request, decision)
            if (
                decision.backend is ExecutionBackend.MCP
                and not result.success
                and self._should_fallback_mcp_failure(result)
            ):
                result = await self._dispatch_mcp_fallback_to_a1(
                    request, decision, result
                )
                decision = self._fallback_decision(
                    decision,
                    reason_code="mcp_execution_fallback_to_a1",
                    rationale=(
                        "The MCP attempt reached a terminal failure; A1 receives the "
                        "same logical intent and owns adaptive recovery."
                    ),
                )
        if result.result_status == "task_pending":
            retry_count = failure_count
        elif result.success:
            self._failure_counts.pop(retry_key, None)
            self._failure_limits.pop(retry_key, None)
            retry_count = 0
        else:
            retry_count = failure_count + 1
            self._failure_counts[retry_key] = retry_count
            self._failure_limits[retry_key] = (
                1 if self._non_retryable_route_failure(result) else 2
            )
        task_metadata = (
            dict(result.task_metadata)
            if isinstance(result.task_metadata, dict)
            else {}
        )
        task_metadata["omniagent_route"] = decision.to_dict()
        task_metadata["omniagent_route"]["execution_invoked"] = execution_invoked
        task_metadata["omniagent_route"]["retry_state"] = {
            "route_signature": decision.route_signature,
            "execution_signature": decision.execution_signature,
            "retry_key": retry_key,
            "failure_count": retry_count,
            "failure_limit": self._failure_limits.get(
                retry_key, failure_limit
            ),
            "result_status": result.result_status,
            "last_error": result.errors[-1] if result.errors else "",
            "catalog_revision": decision.catalog_revision,
            "state_version": request.state_version,
        }
        result.task_metadata = task_metadata
        route_record = {
            "iteration": request.iteration,
            "step_id": request.step.step_id,
            "backend": decision.backend.value,
            "requested_backend": decision.requested_backend,
            "reason_code": decision.reason_code,
            "success": result.success,
            "execution_invoked": execution_invoked,
            "route_signature": decision.route_signature,
            "retry_blocked": decision.retry_blocked,
            "candidate_count": len(decision.candidates),
            "admitted_capability": decision.admitted_capability,
            "selected_capability": decision.selected_capability,
            "bound_call": (
                decision.bound_call.to_dict() if decision.bound_call else None
            ),
            "bound_workflow": (
                decision.bound_workflow.to_dict()
                if decision.bound_workflow
                else None
            ),
            "semantic_intent": (
                decision.semantic_intent.to_dict()
                if decision.semantic_intent
                else None
            ),
            "binding_id": decision.binding_id,
            "catalog_revision": decision.catalog_revision,
            "condition_coverage": decision.condition_coverage.to_dict(),
            "output_contract": decision.output_contract,
            "domain_workflow": decision.domain_workflow or None,
            "retry_state": task_metadata["omniagent_route"]["retry_state"],
        }
        self.route_history.append(route_record)
        self.event_sink("execution_route_completed", route_record)
        self._last_backend = decision.backend
        self._last_success = result.success
        return result

    @staticmethod
    def _should_fallback_mcp_failure(result: A1TaskResult) -> bool:
        """Fallback only terminal, known MCP failures; never duplicate a live job."""
        if result.result_status == "task_pending":
            return False
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        remote_status = str(metadata.get("status", "")).strip().casefold()
        if remote_status in {"queued", "running", "retry_wait", "unknown"}:
            return False
        return result.result_status not in {
            "task_unknown",
            "task_wait_timed_out",
            "poll_transient_error",
            "task_timed_out",
        }

    async def _dispatch_mcp_fallback_to_a1(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
        mcp_result: A1TaskResult,
    ) -> A1TaskResult:
        fallback = self._fallback_decision(
            decision,
            reason_code="mcp_execution_fallback_to_a1",
            rationale=(
                "The MCP attempt reached a terminal failure; A1 receives the same "
                "logical intent and owns adaptive recovery."
            ),
        )
        failure_context = {
            "status": mcp_result.result_status,
            "errors": [str(item) for item in mcp_result.errors[:3]],
            "task": _external_task_summary(mcp_result.task_metadata),
        }
        inputs = dict(request.step.inputs)
        inputs["mcp_fallback_context"] = failure_context
        fallback_request = replace(
            request,
            step=replace(request.step, inputs=inputs),
        )
        a1_result = await self.a1_backend.run(fallback_request, fallback)
        metadata = (
            dict(a1_result.task_metadata)
            if isinstance(a1_result.task_metadata, dict)
            else {}
        )
        metadata["mcp_fallback"] = failure_context
        metadata["mcp_fallback_route"] = decision.to_dict()
        a1_result.task_metadata = metadata
        a1_result.tool_trace = [
            {
                "event": "mcp_execution_fallback_to_a1",
                "mcp_result_status": mcp_result.result_status,
                "mcp_errors": [str(item) for item in mcp_result.errors[:3]],
            },
            *mcp_result.tool_trace,
            *a1_result.tool_trace,
        ]
        a1_result.artifacts = list(
            dict.fromkeys([*mcp_result.artifacts, *a1_result.artifacts])
        )
        return a1_result

    @staticmethod
    def _fallback_decision(
        decision: RouteDecision,
        *,
        reason_code: str,
        rationale: str,
    ) -> RouteDecision:
        domain_workflow = decision.domain_workflow
        if isinstance(domain_workflow, dict) and domain_workflow:
            domain_workflow = json.loads(
                json.dumps(domain_workflow, ensure_ascii=False, default=str)
            )
            nodes = domain_workflow.get("nodes", [])
            if isinstance(nodes, list):
                for node in nodes:
                    if not isinstance(node, dict) or node.get("executor") == "harness":
                        continue
                    node["executor"] = "a1"
                    node["capability_id"] = ""
            domain_workflow["mcp_bindings"] = {}
            domain_workflow["execution_ownership"] = "a1_full_workflow"
            domain_workflow["routing_fallback_reason"] = rationale
        metadata = dict(decision.policy_metadata)
        metadata["fallback_from"] = decision.backend.value
        metadata["fallback_reason"] = reason_code
        return replace(
            decision,
            backend=ExecutionBackend.A1,
            reason_code=reason_code,
            rationale=rationale,
            binding_id="adaptive.a1_fallback.v1",
            bound_call=None,
            bound_workflow=None,
            domain_workflow=domain_workflow or {},
            evidence_purpose="claim_evidence",
            policy_metadata=metadata,
        )

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        """Continue an existing task using its persisted route and task identity."""
        route = task_metadata.get("omniagent_route", {})
        if not isinstance(route, dict):
            return A1TaskResult(
                success=False,
                result_status="task_resume_failed",
                errors=["TASK_RESUME_FAILED: pending task lacks OmniAgent route metadata"],
            )
        backend = str(route.get("backend", ""))
        workflow_stage = task_metadata.get("domain_workflow_stage")
        if (
            backend == ExecutionBackend.A1.value
            and isinstance(workflow_stage, dict)
            and workflow_stage.get("phase") == "mcp"
        ):
            result = await self.mcp_backend.poll(request, task_metadata)
            if result.result_status == "task_pending":
                metadata = (
                    dict(result.task_metadata)
                    if isinstance(result.task_metadata, dict)
                    else {}
                )
                metadata["omniagent_route"] = route
                metadata["domain_workflow_stage"] = workflow_stage
                result.task_metadata = metadata
                return result
            if result.success:
                decision = self._decision_from_route(route, request)
                return await self._dispatch_a1_after_mcp(request, decision, result)
        elif backend == ExecutionBackend.A1.value:
            result = await self.a1_backend.poll(request, task_metadata)
        elif backend == ExecutionBackend.MCP.value:
            result = await self.mcp_backend.poll(request, task_metadata)
        else:
            return A1TaskResult(
                success=False,
                result_status="task_resume_failed",
                errors=[
                    "TASK_RESUME_FAILED: persisted route cannot resume this external task"
                ],
                task_metadata=dict(task_metadata),
            )
        metadata = dict(result.task_metadata) if isinstance(result.task_metadata, dict) else {}
        metadata["omniagent_route"] = route
        if isinstance(workflow_stage, dict):
            metadata["domain_workflow_stage"] = workflow_stage
            if workflow_stage.get("phase") == "a1":
                result.tool_trace = [
                    item
                    for item in workflow_stage.get("mcp_trace", [])
                    if isinstance(item, dict)
                ] + result.tool_trace
                result.artifacts = list(
                    dict.fromkeys(
                        [
                            *(
                                str(item)
                                for item in workflow_stage.get("mcp_artifacts", [])
                                if str(item)
                            ),
                            *result.artifacts,
                        ]
                    )
                )
        result.task_metadata = metadata
        if route.get("bound_workflow"):
            workflow_resume = task_metadata.get("workflow_resume")
            if isinstance(workflow_resume, dict):
                result.tool_trace.insert(
                    0,
                    {
                        "event": "execution_workflow_task_polled",
                        "waiting_workflow_step": workflow_resume.get(
                            "waiting_workflow_step"
                        ),
                        "deferred_workflow_steps": workflow_resume.get(
                            "deferred_workflow_steps", []
                        ),
                    },
                )
        result.tool_trace.insert(
            0,
            {
                "event": "execution_backend_polled",
                "backend": backend,
                "task_id": metadata.get("task_id"),
                "request_id": metadata.get("request_id"),
            },
        )
        return result

    async def _run_mixed_domain_workflow(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> A1TaskResult:
        mcp_decision = replace(
            decision,
            backend=ExecutionBackend.MCP,
            reason_code="domain_workflow_mcp_stage",
        )
        result = await self.mcp_backend.run(request, mcp_decision)
        if result.result_status == "task_pending":
            metadata = (
                dict(result.task_metadata)
                if isinstance(result.task_metadata, dict)
                else {}
            )
            metadata["domain_workflow_stage"] = {
                "phase": "mcp",
                "workflow_id": decision.domain_workflow.get("qualified_id"),
                "parent_request_id": decision.request_id or request.request_id,
            }
            result.task_metadata = metadata
            return result
        if not result.success:
            return result
        return await self._dispatch_a1_after_mcp(request, decision, result)

    async def _dispatch_a1_after_mcp(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
        mcp_result: A1TaskResult,
    ) -> A1TaskResult:
        evidence = (
            mcp_result.verification_payload
            if mcp_result.verification_payload is not None
            else mcp_result.raw
        )
        inputs = dict(request.step.inputs)
        projected_mcp_evidence = project_workflow_evidence(
            decision.domain_workflow,
            evidence,
            artifacts=mcp_result.artifacts,
        )
        inputs["domain_workflow_evidence"] = projected_mcp_evidence
        workflow_id = str(
            decision.domain_workflow.get("qualified_id") or "domain_workflow"
        )
        mcp_metadata = (
            mcp_result.task_metadata
            if isinstance(mcp_result.task_metadata, dict)
            else {}
        )
        workflow_stage = mcp_metadata.get("domain_workflow_stage")
        parent_request_id = str(
            decision.request_id
            or (
                workflow_stage.get("parent_request_id")
                if isinstance(workflow_stage, dict)
                else ""
            )
            or mcp_metadata.get("workflow_parent_request_id")
            or request.request_id
            or f"omniagent:{request.run_id}:{request.step.step_id}"
        ).strip()
        a1_request_id = stage_request_id(
            parent_request_id,
            stage="a1",
            identity=workflow_id,
        )
        staged_request = replace(
            request,
            step=replace(request.step, inputs=inputs),
            request_id=a1_request_id,
        )
        a1_result = await self.a1_backend.run(staged_request, decision)
        a1_metadata = (
            dict(a1_result.task_metadata)
            if isinstance(a1_result.task_metadata, dict)
            else {}
        )
        # The A1 task owns the active lifecycle after this boundary. MCP
        # workflow state remains available only as nested provenance below.
        a1_metadata.pop("workflow_resume", None)
        a1_metadata.setdefault("request_id", a1_request_id)
        a1_metadata["omniagent_route"] = decision.to_dict()
        a1_metadata["domain_workflow_stage"] = {
            "phase": "a1",
            "workflow_id": workflow_id,
            "parent_request_id": parent_request_id,
            "request_id": a1_request_id,
            "mcp_trace": _tool_trace_summary(mcp_result.tool_trace),
            "mcp_artifacts": list(mcp_result.artifacts),
            "mcp_task_metadata": _external_task_summary(mcp_result.task_metadata),
            "mcp_evidence": projected_mcp_evidence,
        }
        a1_result.task_metadata = a1_metadata
        a1_result.tool_trace = [
            *mcp_result.tool_trace,
            {
                "event": "domain_workflow_stage_completed",
                "workflow_id": decision.domain_workflow.get("qualified_id"),
                "completed_stage": "mcp",
                "next_stage": "a1",
            },
            *a1_result.tool_trace,
        ]
        a1_result.artifacts = list(
            dict.fromkeys([*mcp_result.artifacts, *a1_result.artifacts])
        )
        return a1_result

    @staticmethod
    def _decision_from_route(
        route: dict[str, Any],
        request: A1TaskRequest,
    ) -> RouteDecision:
        raw_intent = route.get("semantic_intent")
        intent = SemanticCapabilityIntent.from_inputs(
            {"semantic_intent": raw_intent}
        ) if isinstance(raw_intent, dict) else None
        return RouteDecision(
            backend=ExecutionBackend.A1,
            reason_code=str(route.get("reason_code") or "domain_workflow_bound"),
            rationale=str(route.get("rationale") or "Resume mixed domain workflow."),
            query=str(route.get("query") or request.step.inputs.get("tool_query") or ""),
            semantic_intent=intent,
            output_contract=(
                dict(route.get("output_contract"))
                if isinstance(route.get("output_contract"), dict)
                else {}
            ),
            domain_workflow=(
                dict(route.get("domain_workflow"))
                if isinstance(route.get("domain_workflow"), dict)
                else {}
            ),
            request_id=str(route.get("request_id") or request.request_id),
        )

    def decide(
        self,
        request: A1TaskRequest,
        *,
        query: str,
        candidates: list[ResourceCandidate],
    ) -> RouteDecision:
        semantic_intent = SemanticCapabilityIntent.from_inputs(request.step.inputs)
        explicit = str(request.step.inputs.get("execution_backend", "")).lower()
        if explicit and explicit not in {item.value for item in ExecutionBackend}:
            return self._decision(
                ExecutionBackend.UNAVAILABLE,
                "planner_invalid_backend",
                "The planner supplied an unsupported backend; Harness binding must decide it.",
                query,
                candidates,
            )
        if self.routing_policy.mode is RoutingMode.A1_ONLY:
            return self._decision(
                ExecutionBackend.A1,
                "routing_policy_a1_only",
                "The configured policy assigns all execution to A1.",
                query,
                [],
            )
        if semantic_intent is None:
            candidate = self._named_schema_candidate(request, candidates)
            if candidate is not None and self.routing_policy.profile_for(
                candidate.qualified_name
            ) is not None:
                return self._decision(
                    ExecutionBackend.MCP,
                    "legacy_exact_allowlisted_call",
                    "A legacy exact call is allowed only after catalog and direct-profile validation.",
                    query,
                    [candidate],
                )
            return self._decision(
                ExecutionBackend.A1,
                "planner_intent_missing_fallback_to_a1",
                "The plan has no valid semantic intent; A1 owns the adaptive execution.",
                query,
                [],
            )
        if not self.routing_policy.should_inspect(semantic_intent):
            return self._decision(
                ExecutionBackend.A1,
                "semantic_policy_fallback_to_a1",
                "The intent is outside the small direct-MCP planning retrieval boundary.",
                query,
                [],
            )
        filtered = self.routing_policy.filter_candidates(candidates)
        compatible = [
            item
            for item in filtered
            if self.routing_policy.derive_arguments(request, item, semantic_intent)
            is not None
        ]
        if compatible:
            return self._decision(
                ExecutionBackend.MCP,
                "semantic_policy_match",
                "An allowlisted direct MCP capability can be bound from the typed intent.",
                query,
                compatible,
            )
        return self._decision(
            ExecutionBackend.A1,
            "semantic_capability_fallback_to_a1",
            "No allowlisted direct MCP capability can be safely bound from the typed intent.",
            query,
            [],
        )

    @staticmethod
    def _requested_backend(
        request: A1TaskRequest,
        semantic_intent: SemanticCapabilityIntent | None,
    ) -> ExecutionBackend:
        raw = str(request.step.inputs.get("execution_backend", "")).strip().lower()
        if raw in {item.value for item in ExecutionBackend}:
            return ExecutionBackend(raw)
        if raw:
            return ExecutionBackend.UNAVAILABLE
        if semantic_intent is not None and CapabilityCatalogResolver.requires_catalog(
            semantic_intent
        ):
            return ExecutionBackend.MCP
        if request.step.inputs.get("tool_name") and isinstance(
            request.step.inputs.get("arguments"), dict
        ):
            return ExecutionBackend.MCP
        return ExecutionBackend.A1

    @staticmethod
    def _route_signature(request: A1TaskRequest, decision: RouteDecision) -> str:
        intent = decision.semantic_intent
        payload = {
            "backend": decision.backend.value,
            "binding_id": decision.binding_id,
            "catalog_revision": decision.catalog_revision,
            "state_version": request.state_version,
            "semantic_intent": HybridExecutionRouter._normalized_intent(intent),
            "bound_call": (
                decision.bound_call.to_dict() if decision.bound_call else None
            ),
            "bound_workflow": (
                decision.bound_workflow.to_dict() if decision.bound_workflow else None
            ),
        }
        if decision.bound_call is None and decision.bound_workflow is None:
            payload["semantic_contract"] = {
                "requested_tool": request.step.inputs.get("tool_name"),
                "arguments": request.step.inputs.get("arguments"),
            }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _execution_payload(
        request: A1TaskRequest, decision: RouteDecision
    ) -> dict[str, Any]:
        """Describe immutable executable work independently of planner wording."""
        inputs = request.step.inputs
        dependencies = {
            key: inputs.get(key)
            for key in (
                "state_dependencies",
                "evidence_refs",
                "evidence_ids",
                "replicate_id",
            )
            if key in inputs
        }
        payload: dict[str, Any] = {
            "backend": decision.backend.value,
            "binding_id": decision.binding_id,
            "catalog_revision": decision.catalog_revision,
            "dependencies": dependencies,
        }
        if decision.bound_call is not None:
            payload["bound_call"] = {
                "tool_name": decision.bound_call.tool_name,
                "arguments": decision.bound_call.arguments,
                "capability_version": decision.bound_call.capability_version,
                "execution_mode": decision.bound_call.execution_mode,
            }
            return payload
        if decision.bound_workflow is not None:
            payload["bound_workflow"] = {
                "workflow_id": decision.bound_workflow.workflow_id,
                "inputs": decision.bound_workflow.inputs,
                "steps": [
                    {
                        "tool_name": step.tool_name,
                        "arguments": step.arguments,
                    }
                    for step in decision.bound_workflow.steps
                ],
                "max_steps": decision.bound_workflow.max_steps,
            }
            return payload

        payload["semantic_intent"] = HybridExecutionRouter._normalized_intent(
            decision.semantic_intent
        )
        payload["structured_inputs"] = {
            str(key): value
            for key, value in inputs.items()
            if key
            not in {
                "execution_backend",
                "tool_query",
                "semantic_intent",
                "retrieved_biomni_capabilities",
                "request_id",
                "workflow_phase",
                "state_dependencies",
                "evidence_refs",
                "evidence_ids",
                "replicate_id",
            }
        }
        if decision.semantic_intent is None:
            payload["fallback_contract"] = {
                "objective": " ".join(request.step.objective.split()).casefold(),
                "expected_outputs": sorted(
                    " ".join(str(item).split()).casefold()
                    for item in request.step.expected_outputs
                ),
                "success_criteria": sorted(
                    " ".join(str(item).split()).casefold()
                    for item in request.step.success_criteria
                ),
            }
        return payload

    @staticmethod
    def _execution_signature(request: A1TaskRequest, decision: RouteDecision) -> str:
        """Fingerprint immutable structured work for safe cross-iteration reuse."""
        payload = decision.execution_payload or HybridExecutionRouter._execution_payload(
            request, decision
        )
        return canonical_action_key(payload)[:16]

    @staticmethod
    def _idempotency_key(request: A1TaskRequest, decision: RouteDecision) -> str:
        """Identify one logical action without run or mutable state version.

        The ledger is scoped to one run. Omitting ``run_id`` lets a repeated
        action proposed in a later planning iteration attach to or reuse the
        earlier action instead of submitting the same external request again.
        """
        payload = decision.execution_payload or HybridExecutionRouter._execution_payload(
            request, decision
        )
        return canonical_action_key(payload)

    @staticmethod
    def _retry_key(decision: RouteDecision) -> str:
        """Keep equivalent external work deduplicated across failure-only state changes."""
        return decision.execution_signature or decision.route_signature

    def _prior_failure_count(self, request: A1TaskRequest, signature: str) -> int:
        persisted = request.route_retry_state.get(signature, {})
        try:
            persisted_count = (
                int(persisted.get("failure_count", 0))
                if isinstance(persisted, dict)
                else 0
            )
        except (TypeError, ValueError):
            persisted_count = 0
        return max(self._failure_counts.get(signature, 0), persisted_count)

    @staticmethod
    def _normalized_intent(
        intent: SemanticCapabilityIntent | None,
    ) -> dict[str, Any] | None:
        if intent is None:
            return None
        side_effect = intent.side_effect.value
        if intent.operation.value in {"retrieve", "validate"}:
            side_effect = "read_only"
        return {
            "operation": intent.operation.value,
            "capability_query": " ".join(intent.capability_query.split()).casefold(),
            "execution_shape": intent.execution_shape.value,
            "schema_bound": intent.schema_bound,
            "side_effect": side_effect,
            "required_output_fields": sorted(intent.required_output_fields),
            "expected_artifacts": sorted(intent.expected_artifacts),
            "entity_context": {
                key: intent.entity_context[key] for key in sorted(intent.entity_context)
            },
            "capability_hint": intent.capability_hint.strip().casefold(),
        }

    @staticmethod
    def _non_retryable_route_failure(result: A1TaskResult) -> bool:
        if not execution_failure_retryable(result):
            return True
        text = " ".join([result.result_status, *result.errors]).lower()
        markers = (
            "context length",
            "context window",
            "maximum context",
            "token limit",
            "max_react_steps",
            "max react steps",
            "task_timed_out",
            "task_wait_timed_out",
            "task_unknown",
            "poll_transient_error",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _schema_bound_candidate(
        request: A1TaskRequest,
        text: str,
        candidates: list[ResourceCandidate],
    ) -> tuple[CapabilityRule, ResourceCandidate] | None:
        """Return a bounded capability only when its evidence and inputs are present."""
        for rule in CAPABILITY_RULES:
            if not rule.matches(text):
                continue
            candidate = rule.find_candidate(candidates)
            if candidate is None:
                continue
            if rule.reason_code == "schema_bound_disease_target_lookup":
                if HybridExecutionRouter._disease_lookup_value(request) is None:
                    continue
            elif rule.reason_code == "schema_bound_interaction_lookup":
                if HybridExecutionRouter._interaction_gene_value(request) is None:
                    continue
            return rule, candidate
        return None

    @staticmethod
    def _pdb_query_candidate(
        text: str,
        candidates: list[ResourceCandidate],
    ) -> ResourceCandidate | None:
        structure_markers = ("pdb", "protein data bank", "solution nmr", "nmr structure")
        if not any(marker in text for marker in structure_markers):
            return None
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.qualified_name.endswith(
                    (".query_pdb", ".query_pdb_identifiers")
                )
            ),
            None,
        )

    @staticmethod
    def _named_schema_candidate(
        request: A1TaskRequest,
        candidates: list[ResourceCandidate],
    ) -> ResourceCandidate | None:
        """Match an explicitly named tool only when its schema is satisfiable."""
        requested = str(
            request.step.inputs.get("tool_name")
            or request.step.inputs.get("tool_query")
            or ""
        ).strip()
        arguments = request.step.inputs.get("arguments")
        if not requested or not isinstance(arguments, dict):
            return None
        candidate = next(
            (item for item in candidates if item.qualified_name == requested),
            None,
        )
        if candidate is None:
            return None
        required = candidate.input_schema.get("required", [])
        if not isinstance(required, list):
            return None
        if any(str(name) not in arguments for name in required):
            return None
        return candidate

    @staticmethod
    def _pdb_structured_query(
        request: A1TaskRequest,
        max_results: int,
    ) -> dict[str, Any] | None:
        """Build an RCSB query without invoking Biomni's nested LLM parser."""
        inputs = request.step.inputs
        text = " ".join(
            [request.step.objective, str(inputs.get("tool_query", ""))]
        )

        def value(*keys: str) -> str | None:
            for key in keys:
                item = inputs.get(key)
                if isinstance(item, str) and item.strip():
                    return item.strip()
            return None

        gene = value("gene_symbol", "gene", "target_gene")
        if gene is None:
            match = re.search(
                r"\b(?:gene|protein|for)\s+(?P<gene>[A-Z][A-Z0-9-]{1,15})\b",
                text,
            )
            gene = match.group("gene") if match else None
        organism = value("organism", "species")
        if organism is None and re.search(r"\bhuman|homo sapiens\b", text, re.I):
            organism = "Homo sapiens"
        method = value("method", "experimental_method")
        if method is None and re.search(r"solution\s+nmr", text, re.I):
            method = "SOLUTION NMR"
        cutoff = value("date_cutoff", "date_limit", "release_date_before")

        nodes: list[dict[str, Any]] = []

        def add(attribute: str, operator: str, item: str | None) -> None:
            if item:
                nodes.append(
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": attribute,
                            "operator": operator,
                            "value": item,
                        },
                    }
                )

        add("rcsb_entity_source_organism.rcsb_gene_name.value", "exact_match", gene)
        add("rcsb_entity_source_organism.ncbi_scientific_name", "exact_match", organism)
        add("exptl.method", "exact_match", method.upper() if method else None)
        add("rcsb_accession_info.deposit_date", "less_or_equal", cutoff)
        if not nodes:
            return None
        return {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_results}
            },
        }

    @staticmethod
    def _pdb_identifiers(request: A1TaskRequest) -> list[str]:
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
        text = " ".join(
            [
                request.step.objective,
                str(request.step.inputs.get("tool_query", "")),
            ]
        )
        found = re.findall(r"(?<![A-Z0-9])[0-9][A-Z][A-Z0-9]{2}(?![A-Z0-9])", text.upper())
        return list(dict.fromkeys(found))

    @classmethod
    def _opentarget_disease_target_candidate(
        cls,
        request: A1TaskRequest,
        text: str,
        candidates: list[ResourceCandidate],
    ) -> ResourceCandidate | None:
        if not all(marker in text for marker in ("disease", "target")):
            return None
        if not any(marker in text for marker in ("associated", "association", "rank", "top")):
            return None
        if cls._disease_lookup_value(request) is None:
            return None
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.qualified_name.endswith(
                    ".query_opentarget_disease_targets"
                )
            ),
            None,
        )

    @staticmethod
    def _disease_lookup_value(request: A1TaskRequest) -> str | None:
        for key in ("disease", "disease_name", "disease_id"):
            value = request.step.inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        source = " ".join(
            [
                request.step.objective,
                str(request.step.inputs.get("tool_query", "")),
            ]
        )
        patterns = (
            r"\b(?:associated\s+with|targets?\s+for)\s+"
            r"(?P<disease>[A-Za-z][A-Za-z0-9' -]{1,120}?)(?=\s*(?:,|;|\.|\b(?:rank(?:ed|ing)?|sort(?:ed|ing)?|limit|using|from|with|by|according|as)\b|$))",
            r"\b(?P<disease>(?:EFO|MONDO|ORPHA|DOID)_[A-Za-z0-9]+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                disease = match.group("disease").strip()
                if disease:
                    return disease
        return None

    @classmethod
    def _stringdb_interaction_candidate(
        cls,
        request: A1TaskRequest,
        text: str,
        candidates: list[ResourceCandidate],
    ) -> ResourceCandidate | None:
        if "interact" not in text or "score" not in text:
            return None
        if cls._interaction_gene_value(request) is None:
            return None
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.qualified_name.endswith(
                    ".query_stringdb_top_interactor"
                )
            ),
            None,
        )

    @staticmethod
    def _interaction_gene_value(request: A1TaskRequest) -> str | None:
        for key in ("gene_symbol", "gene", "target_gene"):
            value = request.step.inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        source = " ".join(
            [
                request.step.objective,
                str(request.step.inputs.get("tool_query", "")),
            ]
        )
        match = re.search(
            r"\b(?:with|for)\s+(?P<gene>[A-Z][A-Z0-9-]{1,15})\b",
            source,
        )
        return match.group("gene") if match else None

    def _decision(
        self,
        backend: ExecutionBackend,
        reason_code: str,
        rationale: str,
        query: str,
        candidates: list[ResourceCandidate],
    ) -> RouteDecision:
        return RouteDecision(
            backend=backend,
            reason_code=reason_code,
            rationale=rationale,
            query=query,
            candidates=candidates,
            previous_backend=(self._last_backend.value if self._last_backend else ""),
        )
