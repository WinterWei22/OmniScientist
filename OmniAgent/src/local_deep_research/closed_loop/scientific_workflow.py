from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .execution_validation import validate_schema_instance


class ScientificWorkflowError(ValueError):
    """Raised when a domain workflow cannot be safely registered or compiled."""


_EXECUTORS = frozenset({"mcp", "a1", "harness"})
_OPERATORS = frozenset(
    {
        "resolve_entity",
        "retrieve_record",
        "retrieve_artifact",
        "transform",
        "compute",
        "cross_validate",
        "aggregate",
        "verify",
        "materialize",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """The evidence contract for one scientific property or claim."""

    property: str
    context: str
    required_fields: tuple[str, ...] = ()
    accepted_method_classes: tuple[str, ...] = ()
    prohibited_sole_sources: tuple[str, ...] = ()
    required_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.property.strip():
            raise ScientificWorkflowError("evidence requirement requires property")
        if not self.context.strip():
            raise ScientificWorkflowError("evidence requirement requires context")

    def to_dict(self) -> dict[str, Any]:
        return {
            "property": self.property,
            "context": self.context,
            "required_fields": list(self.required_fields),
            "accepted_method_classes": list(self.accepted_method_classes),
            "prohibited_sole_sources": list(self.prohibited_sole_sources),
            "required_provenance": list(self.required_provenance),
        }


@dataclass(frozen=True, slots=True)
class HandoffSelector:
    """Select bounded MCP evidence for the next executor with source provenance."""

    name: str
    paths: tuple[str, ...]
    required: bool = False
    max_matches: int = 8

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ScientificWorkflowError("handoff selector requires name")
        if not self.paths or any(not str(path).strip() for path in self.paths):
            raise ScientificWorkflowError(
                f"handoff selector {self.name!r} requires non-empty paths"
            )
        if self.max_matches < 1:
            raise ScientificWorkflowError(
                f"handoff selector {self.name!r} requires max_matches >= 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paths": list(self.paths),
            "required": self.required,
            "max_matches": self.max_matches,
        }


@dataclass(frozen=True, slots=True)
class WorkflowNodeSpec:
    """A typed domain step; the executor owns its side effects."""

    node_id: str
    executor: str
    operator: str
    depends_on: tuple[str, ...] = ()
    capability_query: str = ""
    capability_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRequirement, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        executor = self.executor.strip().lower()
        operator = self.operator.strip().lower()
        if executor not in _EXECUTORS:
            raise ScientificWorkflowError(
                f"workflow node {self.node_id!r} has unsupported executor {self.executor!r}"
            )
        if operator not in _OPERATORS:
            raise ScientificWorkflowError(
                f"workflow node {self.node_id!r} has unsupported operator {self.operator!r}"
            )
        if not self.node_id.strip():
            raise ScientificWorkflowError("workflow node requires node_id")
        if executor == "mcp" and not (self.capability_id or self.capability_query):
            raise ScientificWorkflowError(
                f"MCP node {self.node_id!r} requires capability_id or capability_query"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        return value


@dataclass(frozen=True, slots=True)
class ScientificWorkflowSpec:
    """Versioned, provider-neutral workflow declaration."""

    workflow_id: str
    version: str
    intent_match: dict[str, Any]
    input_schema: dict[str, Any]
    nodes: tuple[WorkflowNodeSpec, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    handoff_selectors: tuple[HandoffSelector, ...] = ()
    verification_contract: dict[str, Any] = field(default_factory=dict)
    completion_contract: dict[str, Any] = field(default_factory=dict)
    failure_policy: dict[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.workflow_id.strip() or not self.version.strip():
            raise ScientificWorkflowError("workflow_id and version are required")
        if not isinstance(self.intent_match, dict):
            raise ScientificWorkflowError("intent_match must be an object")
        if not isinstance(self.input_schema, dict):
            raise ScientificWorkflowError("workflow input_schema must be an object")
        node_ids = [node.node_id for node in self.nodes]
        if not node_ids:
            raise ScientificWorkflowError("workflow must contain at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ScientificWorkflowError("workflow node IDs must be unique")
        selector_names = [selector.name for selector in self.handoff_selectors]
        if len(set(selector_names)) != len(selector_names):
            raise ScientificWorkflowError("workflow handoff selector names must be unique")
        known = set(node_ids)
        for node in self.nodes:
            missing = set(node.depends_on) - known
            if missing:
                raise ScientificWorkflowError(
                    f"workflow node {node.node_id!r} depends on unknown nodes: {sorted(missing)}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {node.node_id: node for node in self.nodes}

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ScientificWorkflowError("workflow dependencies contain a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent in by_id[node_id].depends_on:
                visit(parent)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            visit(node_id)
        errors = validate_schema_instance(
            {"workflow_inputs": {}},
            self.input_schema,
            strict_objects=True,
        )
        # An empty placeholder is allowed for schemas with required runtime inputs.
        if errors and self.input_schema.get("type") not in {None, "object"}:
            raise ScientificWorkflowError(
                "workflow input_schema must describe an object"
            )

    @property
    def qualified_id(self) -> str:
        return f"{self.workflow_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "qualified_id": self.qualified_id,
            "intent_match": self.intent_match,
            "input_schema": self.input_schema,
            "nodes": [item.to_dict() for item in self.nodes],
            "evidence_requirements": [
                item.to_dict() for item in self.evidence_requirements
            ],
            "handoff_selectors": [
                item.to_dict() for item in self.handoff_selectors
            ],
            "verification_contract": self.verification_contract,
            "completion_contract": self.completion_contract,
            "failure_policy": self.failure_policy,
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True, slots=True)
class CompiledScientificWorkflow:
    """A validated workflow with exact MCP bindings and an A1 contract."""

    spec: ScientificWorkflowSpec
    nodes: tuple[WorkflowNodeSpec, ...]
    mcp_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    a1_nodes: tuple[str, ...] = ()
    harness_nodes: tuple[str, ...] = ()
    compile_errors: tuple[str, ...] = ()
    catalog_revision: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()

    @property
    def executable(self) -> bool:
        return not self.compile_errors

    @property
    def mixed_execution(self) -> bool:
        executors = {node.executor for node in self.nodes}
        return len(executors) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.spec.workflow_id,
            "version": self.spec.version,
            "qualified_id": self.spec.qualified_id,
            "nodes": [item.to_dict() for item in self.nodes],
            "mcp_bindings": self.mcp_bindings,
            "a1_nodes": list(self.a1_nodes),
            "harness_nodes": list(self.harness_nodes),
            "compile_errors": list(self.compile_errors),
            "catalog_revision": self.catalog_revision,
            "inputs": self.inputs,
            "mixed_execution": self.mixed_execution,
            "evidence_requirements": [item.to_dict() for item in self.evidence_requirements],
            "handoff_selectors": [
                item.to_dict() for item in self.spec.handoff_selectors
            ],
            "verification_contract": self.spec.verification_contract,
            "completion_contract": self.spec.completion_contract,
            "failure_policy": self.spec.failure_policy,
        }

    def to_bound_mcp_workflow(self) -> Any | None:
        """Adapt an all-MCP declaration to the existing bounded MCP executor."""
        if not self.executable or self.mixed_execution:
            return None
        if not self.nodes or any(node.executor != "mcp" for node in self.nodes):
            return None
        return self._to_bound_mcp_workflow(
            self.nodes,
            binding_reason="Compiled from a validated all-MCP domain workflow.",
        )

    def to_a1_fallback_dict(self, *, reason: str) -> dict[str, Any]:
        """Give A1 the provider-neutral workflow when direct MCP is not admitted."""
        payload = self.to_dict()
        nodes = payload.get("nodes", [])
        a1_nodes: list[str] = []
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict) or node.get("executor") == "harness":
                    continue
                node["executor"] = "a1"
                node["capability_id"] = ""
                a1_nodes.append(str(node.get("node_id") or ""))
        payload["mcp_bindings"] = {}
        payload["a1_nodes"] = [value for value in a1_nodes if value]
        payload["compile_errors"] = []
        payload["mixed_execution"] = bool(payload.get("harness_nodes"))
        payload["execution_ownership"] = "a1_full_workflow"
        payload["routing_fallback_reason"] = reason
        return payload

    def to_bound_mcp_stage_workflow(self) -> Any | None:
        """Compile the retrieval prefix of a mixed MCP/A1 workflow."""
        if not self.executable:
            return None
        nodes = tuple(node for node in self.nodes if node.executor == "mcp")
        if not nodes:
            return None
        node_ids = {node.node_id for node in nodes}
        if any(set(node.depends_on) - node_ids for node in nodes):
            return None
        return self._to_bound_mcp_workflow(
            nodes,
            binding_reason=(
                "Compiled as the deterministic MCP retrieval stage of a validated "
                "mixed domain workflow."
            ),
        )

    def _to_bound_mcp_workflow(
        self,
        nodes: tuple[WorkflowNodeSpec, ...],
        *,
        binding_reason: str,
    ) -> Any | None:
        from .execution_models import BoundCapabilityWorkflow, EffectContract, WorkflowCallTemplate

        steps = []
        for node in nodes:
            binding = self.mcp_bindings.get(node.node_id)
            if binding is None:
                return None
            steps.append(
                WorkflowCallTemplate(
                    step_id=node.node_id,
                    tool_name=binding["qualified_name"],
                    arguments=dict(node.arguments),
                    input_schema=dict(binding["input_schema"]),
                    output_schema=dict(binding["output_schema"]),
                    effects=EffectContract(
                        required_paths=tuple(
                            field
                            for requirement in node.evidence
                            for field in requirement.required_fields
                        ),
                        description=node.description,
                    ),
                )
            )
        return BoundCapabilityWorkflow(
            workflow_id=self.spec.qualified_id,
            inputs=dict(self.inputs),
            steps=steps,
            max_steps=max(1, len(steps)),
            binding_reason=binding_reason,
            evidence_purpose="planning_evidence",
        )


class DomainWorkflowRegistry:
    """Versioned registry for reusable scientific workflow specifications."""

    def __init__(self, specs: Iterable[ScientificWorkflowSpec] | None = None) -> None:
        self._specs: dict[str, ScientificWorkflowSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: ScientificWorkflowSpec) -> None:
        spec.validate()
        key = spec.qualified_id
        if key in self._specs:
            raise ScientificWorkflowError(f"duplicate workflow registration: {key}")
        self._specs[key] = spec

    def get(self, workflow_id: str, version: str | None = None) -> ScientificWorkflowSpec | None:
        if version:
            return self._specs.get(f"{workflow_id}@{version}")
        matches = [spec for spec in self._specs.values() if spec.workflow_id == workflow_id]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.version)[-1]

    def list(self) -> list[ScientificWorkflowSpec]:
        return sorted(self._specs.values(), key=lambda item: item.qualified_id)

    def resolve(
        self,
        *,
        intent: Any,
        request: Any,
    ) -> ScientificWorkflowSpec | None:
        def normalize(value: Any) -> str:
            return " ".join(
                str(value or "").casefold().replace("_", " ").replace("-", " ").split()
            )

        def flatten(value: Any) -> list[str]:
            if isinstance(value, dict):
                return [item for child in value.values() for item in flatten(child)]
            if isinstance(value, (list, tuple, set)):
                return [item for child in value for item in flatten(child)]
            text = str(value or "").strip()
            return [text] if text else []

        intent_payload = (
            intent.to_dict() if callable(getattr(intent, "to_dict", None)) else intent
        )
        text = normalize(" ".join(
            item
            for value in (
                getattr(intent, "capability_query", ""),
                getattr(request, "research_goal", ""),
                getattr(getattr(request, "step", None), "objective", ""),
                *getattr(getattr(request, "step", None), "expected_outputs", []),
                intent_payload,
            )
            for item in flatten(value)
        ))
        step = getattr(request, "step", None)
        step_payload = getattr(step, "inputs", {})
        step_text = normalize(" ".join(
            item
            for value in (
                getattr(step, "objective", ""),
                *getattr(step, "expected_outputs", []),
                *getattr(step, "success_criteria", []),
                step_payload,
            )
            for item in flatten(value)
        ))
        hint = str(getattr(intent, "capability_hint", "") or "").casefold().strip()
        operation = str(getattr(getattr(intent, "operation", None), "value", "") or "")
        for spec in self.list():
            match = spec.intent_match
            match_text = step_text if match.get("scope") == "step" else text
            if str(match.get("operation") or "").casefold() not in {"", operation.casefold()}:
                continue
            workflow_hint = str(match.get("capability_hint") or "").casefold().strip()
            if workflow_hint and workflow_hint != hint:
                continue
            all_terms = match.get("all_of", [])
            any_terms = match.get("any_of", [])
            if isinstance(all_terms, str):
                all_terms = [all_terms]
            if isinstance(any_terms, str):
                any_terms = [any_terms]
            if any(normalize(term) not in match_text for term in all_terms or []):
                continue
            if any_terms and not any(normalize(term) in match_text for term in any_terms):
                continue
            return spec
        return None


class DomainWorkflowCompiler:
    """Bind declared MCP nodes to retrieved catalog candidates."""

    def compile(
        self,
        spec: ScientificWorkflowSpec,
        *,
        candidates: Iterable[Any],
        catalog_revision: str = "",
        inputs: dict[str, Any] | None = None,
    ) -> CompiledScientificWorkflow:
        spec.validate()
        candidate_list = list(candidates)
        bindings: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        known_nodes = {node.node_id for node in spec.nodes}
        runtime_inputs = self._project_inputs(spec, dict(inputs or {}))
        resolved_requirements = tuple(
            self._resolve_requirement(item, runtime_inputs, errors)
            for item in spec.evidence_requirements
        )
        resolved_nodes = tuple(
            replace(
                node,
                evidence=tuple(
                    self._resolve_requirement(item, runtime_inputs, errors)
                    for item in node.evidence
                ),
            )
            for node in spec.nodes
        )
        executors = {node.node_id: node.executor for node in resolved_nodes}
        for node in resolved_nodes:
            if node.executor == "mcp" and any(
                executors.get(parent) != "mcp" for parent in node.depends_on
            ):
                errors.append(
                    f"workflow node {node.node_id} requires MCP after a non-MCP stage; "
                    "mixed workflows must retrieve evidence before A1 execution"
                )
            if node.executor == "a1" and any(
                executors.get(parent) == "harness" for parent in node.depends_on
            ):
                errors.append(
                    f"workflow node {node.node_id} depends on a Harness verification stage"
                )
        for node in resolved_nodes:
            errors.extend(
                self._template_errors(
                    node.arguments,
                    node=node,
                    known_nodes=known_nodes,
                    inputs=runtime_inputs,
                )
            )
            if node.executor != "mcp":
                continue
            candidate = self._find_candidate(node, candidate_list)
            if candidate is None:
                errors.append(
                    f"missing capability for workflow node {node.node_id}: "
                    f"{node.capability_id or node.capability_query}"
                )
                continue
            input_schema = getattr(candidate, "input_schema", {})
            output_schema = getattr(candidate, "output_schema", {})
            if not isinstance(input_schema, dict) or not input_schema:
                errors.append(f"capability {node.node_id} has no input schema")
                continue
            if not isinstance(output_schema, dict) or not output_schema:
                errors.append(f"capability {node.node_id} has no output schema")
                continue
            if node.input_schema and node.input_schema != input_schema:
                errors.append(f"schema drift for workflow node {node.node_id}: input_schema")
                continue
            if node.output_schema and node.output_schema != output_schema:
                errors.append(f"schema drift for workflow node {node.node_id}: output_schema")
                continue
            bindings[node.node_id] = {
                "qualified_name": str(getattr(candidate, "qualified_name", "")),
                "input_schema": input_schema,
                "output_schema": output_schema,
                "capability_version": str(getattr(candidate, "capability_version", "")),
                "arguments": node.arguments,
            }
        return CompiledScientificWorkflow(
            spec=spec,
            nodes=resolved_nodes,
            mcp_bindings=bindings,
            a1_nodes=tuple(node.node_id for node in resolved_nodes if node.executor == "a1"),
            harness_nodes=tuple(
                node.node_id for node in resolved_nodes if node.executor == "harness"
            ),
            compile_errors=tuple(errors),
            catalog_revision=catalog_revision,
            inputs=runtime_inputs,
            evidence_requirements=resolved_requirements,
        )

    @classmethod
    def _project_inputs(
        cls,
        spec: ScientificWorkflowSpec,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep only inputs declared or referenced by the executable workflow."""
        referenced: set[str] = set()
        properties = spec.input_schema.get("properties", {})
        if isinstance(properties, dict):
            referenced.update(str(key) for key in properties)
        for node in spec.nodes:
            cls._collect_input_references(node.arguments, referenced)
            for requirement in node.evidence:
                cls._collect_input_references(requirement.to_dict(), referenced)
        for requirement in spec.evidence_requirements:
            cls._collect_input_references(requirement.to_dict(), referenced)
        for contract in (
            spec.verification_contract,
            spec.completion_contract,
            spec.failure_policy,
        ):
            cls._collect_input_references(contract, referenced)
        return {key: inputs[key] for key in sorted(referenced) if key in inputs}

    @classmethod
    def _collect_input_references(cls, value: Any, referenced: set[str]) -> None:
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                parts = [part for part in str(value["$ref"] or "").split(".") if part]
                if len(parts) >= 2 and parts[0] == "inputs":
                    referenced.add(parts[1])
            for child in value.values():
                cls._collect_input_references(child, referenced)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                cls._collect_input_references(child, referenced)
            return
        if isinstance(value, str):
            referenced.update(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value))

    @staticmethod
    def _resolve_requirement(
        requirement: EvidenceRequirement,
        inputs: dict[str, Any],
        errors: list[str],
    ) -> EvidenceRequirement:
        values = {
            "property": requirement.property,
            "context": requirement.context,
            "required_fields": requirement.required_fields,
            "accepted_method_classes": requirement.accepted_method_classes,
            "prohibited_sole_sources": requirement.prohibited_sole_sources,
            "required_provenance": requirement.required_provenance,
        }

        def resolve(value: str) -> str:
            resolved = value
            for key, item in inputs.items():
                resolved = resolved.replace("{" + str(key) + "}", str(item))
            if "{" in resolved or "}" in resolved:
                errors.append(f"missing workflow parameter for evidence requirement: {value}")
            return resolved

        return EvidenceRequirement(
            property=resolve(values["property"]),
            context=resolve(values["context"]),
            required_fields=tuple(resolve(item) for item in values["required_fields"]),
            accepted_method_classes=tuple(
                resolve(item) for item in values["accepted_method_classes"]
            ),
            prohibited_sole_sources=tuple(
                resolve(item) for item in values["prohibited_sole_sources"]
            ),
            required_provenance=tuple(
                resolve(item) for item in values["required_provenance"]
            ),
        )

    @classmethod
    def _template_errors(
        cls,
        value: Any,
        *,
        node: WorkflowNodeSpec,
        known_nodes: set[str],
        inputs: dict[str, Any] | None,
    ) -> list[str]:
        if isinstance(value, list):
            return [
                error
                for item in value
                for error in cls._template_errors(
                    item,
                    node=node,
                    known_nodes=known_nodes,
                    inputs=inputs,
                )
            ]
        if not isinstance(value, dict):
            return []
        if set(value) == {"$ref"}:
            reference = str(value["$ref"] or "")
            parts = [part for part in reference.split(".") if part]
            if len(parts) < 2 or parts[0] not in {"inputs", "steps"}:
                return [f"workflow node {node.node_id} has invalid reference: {reference}"]
            if parts[0] == "inputs":
                if inputs is not None and not cls._input_path_exists(inputs, parts[1:]):
                    return [
                        f"workflow node {node.node_id} references missing input: {reference}"
                    ]
                return []
            if len(parts) < 3 or parts[2] != "result" or parts[1] not in known_nodes:
                return [
                    f"workflow node {node.node_id} references unknown step: {reference}"
                ]
            if parts[1] not in set(node.depends_on):
                return [
                    f"workflow node {node.node_id} references undeclared dependency: {reference}"
                ]
            return []
        if set(value) == {"$select"}:
            spec = value["$select"]
            if not isinstance(spec, dict) or "source" not in spec:
                return [f"workflow node {node.node_id} has an invalid $select template"]
            return cls._template_errors(
                spec["source"],
                node=node,
                known_nodes=known_nodes,
                inputs=inputs,
            )
        if set(value) == {"$map"}:
            spec = value["$map"]
            if not isinstance(spec, dict) or "source" not in spec:
                return [f"workflow node {node.node_id} has an invalid $map template"]
            errors = cls._template_errors(
                spec["source"],
                node=node,
                known_nodes=known_nodes,
                inputs=inputs,
            )
            variables = spec.get("variables", {})
            if isinstance(variables, dict):
                errors.extend(
                    error
                    for item in variables.values()
                    for error in cls._template_errors(
                        item,
                        node=node,
                        known_nodes=known_nodes,
                        inputs=inputs,
                    )
                )
            return errors
        if set(value) == {"$unique"}:
            return cls._template_errors(
                value["$unique"],
                node=node,
                known_nodes=known_nodes,
                inputs=inputs,
            )
        if any(str(key).startswith("$") for key in value):
            return [f"workflow node {node.node_id} has an invalid template operator"]
        return [
            error
            for item in value.values()
            for error in cls._template_errors(
                item,
                node=node,
                known_nodes=known_nodes,
                inputs=inputs,
            )
        ]

    @staticmethod
    def _input_path_exists(inputs: dict[str, Any], path: list[str]) -> bool:
        current: Any = inputs
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    @staticmethod
    def _find_candidate(node: WorkflowNodeSpec, candidates: list[Any]) -> Any | None:
        if node.capability_id:
            exact = next(
                (
                    item
                    for item in candidates
                    if str(getattr(item, "qualified_name", "")) == node.capability_id
                ),
                None,
            )
            if exact is not None:
                return exact
        query = node.capability_query.casefold()
        if not query:
            return None
        tokens = {token for token in query.replace("/", " ").split() if len(token) > 3}
        ranked: list[tuple[int, Any]] = []
        for candidate in candidates:
            text = " ".join(
                (
                    str(getattr(candidate, "qualified_name", "")),
                    str(getattr(candidate, "description", "")),
                )
            ).casefold()
            score = sum(token in text for token in tokens)
            if score:
                ranked.append((score, candidate))
        return max(ranked, key=lambda item: item[0])[1] if ranked else None


def conditioned_property_v1() -> ScientificWorkflowSpec:
    """Return the parameterized evidence workflow for condition-dependent properties."""
    requirement = EvidenceRequirement(
        property="{property_name}",
        context="{context}",
        required_fields=(
            "entity_identity",
            "condition_value",
            "transformed_state",
            "computed_value",
        ),
        accepted_method_classes=("context_aware_computation", "context_aware_reference"),
        prohibited_sole_sources=("context_free_metadata", "raw_descriptor"),
        required_provenance=("source_fields", "method", "method_class"),
    )
    return ScientificWorkflowSpec(
        workflow_id="conditioned_property",
        version="v1",
        intent_match={
            "operation": "retrieve",
            "capability_hint": "conditioned_property.v1",
            "any_of": ["conditioned", "ph-dependent", "crystallization pH", "under pH"],
        },
        input_schema={"type": "object"},
        nodes=(
            WorkflowNodeSpec(
                node_id="entity",
                executor="mcp",
                operator="resolve_entity",
                capability_query="retrieve structure entity and ligand identity",
                arguments={"entity": {"$ref": "inputs.entity"}},
                description="Resolve the requested entity and its provenance.",
            ),
            WorkflowNodeSpec(
                node_id="condition",
                executor="mcp",
                operator="retrieve_record",
                depends_on=("entity",),
                capability_query="retrieve experimental condition pH",
                arguments={"entity": {"$ref": "inputs.entity"}},
                description="Retrieve the condition used for the property.",
            ),
            WorkflowNodeSpec(
                node_id="transform",
                executor="a1",
                operator="transform",
                depends_on=("entity", "condition"),
                arguments={
                    "transformation": {"$ref": "inputs.transformation"},
                    "condition": {"$ref": "steps.condition.result"},
                    "entity": {"$ref": "steps.entity.result"},
                },
                description="Apply a reproducible condition-aware transformation.",
            ),
            WorkflowNodeSpec(
                node_id="compute",
                executor="a1",
                operator="compute",
                depends_on=("transform",),
                arguments={
                    "calculation": {"$ref": "inputs.calculation"},
                    "state": {"$ref": "steps.transform.result"},
                },
                evidence=(requirement,),
                description="Compute the requested property from the transformed state.",
            ),
            WorkflowNodeSpec(
                node_id="verify",
                executor="harness",
                operator="verify",
                depends_on=("compute",),
                arguments={"evidence": {"$ref": "steps.compute.result"}},
                evidence=(requirement,),
                description="Verify method, provenance, and required evidence fields.",
            ),
        ),
        evidence_requirements=(requirement,),
        handoff_selectors=(
            HandoffSelector(
                name="resolved_entity",
                paths=("steps.entity.result",),
                required=True,
                max_matches=1,
            ),
            HandoffSelector(
                name="experimental_condition",
                paths=("steps.condition.result",),
                required=True,
                max_matches=1,
            ),
        ),
        verification_contract={
            "verifier_id": "omniagent.scientific_evidence.v1",
            "reject_prohibited_sole_sources": True,
        },
        completion_contract={
            "required_nodes": ["entity", "condition", "transform", "compute", "verify"],
            "required_evidence": ["{property_name}"],
            "pending_blocks_completion": True,
        },
        failure_policy={
            "missing_capability": "needs_review",
            "schema_drift": "needs_review",
            "a1_pending": "wait_external",
        },
        source_refs=(
            "https://github.com/openJiuwen-ai/sciencediscovery",
            "https://github.com/langchain-ai/langgraph",
        ),
    )


def feature_bounded_proteolysis_v1() -> ScientificWorkflowSpec:
    """Digest a complete protein sequence before filtering by an annotation."""
    requirement = EvidenceRequirement(
        property="feature_bounded_proteolysis",
        context="{feature_name}",
        required_fields=(
            "accession",
            "feature_name",
            "feature_start",
            "feature_end",
            "precursor_sequence",
            "cleavage_residues",
            "missed_cleavages",
            "all_peptides",
            "retained_peptides",
            "longest_peptide_sequence",
            "method",
            "method_class",
            "source_fields",
        ),
        accepted_method_classes=("full_sequence_cleavage_then_interval_filter",),
        prohibited_sole_sources=("isolated_feature_digestion", "feature_subsequence"),
        required_provenance=("source_fields", "method", "method_class"),
    )
    return ScientificWorkflowSpec(
        workflow_id="feature_bounded_proteolysis",
        version="v1",
        intent_match={
            "all_of": ["trypsin", "uniprot"],
            "any_of": ["entirely within", "feature boundaries", "chain", "annotated"],
        },
        input_schema={
            "type": "object",
            "properties": {
                "uniprot_accession": {"type": "string"},
                "uniprot_endpoint": {"type": "string"},
                "feature_name": {"type": "string"},
                "cleavage_residues": {"type": "array"},
                "missed_cleavages": {"type": "integer"},
                "output_key": {"type": "string"},
            },
        },
        nodes=(
            WorkflowNodeSpec(
                node_id="retrieve",
                executor="mcp",
                operator="retrieve_record",
                capability_id="biomni.tool.database.query_uniprot",
                capability_query="retrieve a complete canonical UniProt sequence and feature annotations",
                arguments={"endpoint": {"$ref": "inputs.uniprot_endpoint"}},
                description=(
                    "Retrieve the complete canonical sequence and the annotation that "
                    "defines the requested feature interval."
                ),
            ),
            WorkflowNodeSpec(
                node_id="digest_and_filter",
                executor="a1",
                operator="compute",
                depends_on=("retrieve",),
                arguments={
                    "accession": {"$ref": "inputs.uniprot_accession"},
                    "feature_name": {"$ref": "inputs.feature_name"},
                    "cleavage_residues": {"$ref": "inputs.cleavage_residues"},
                    "missed_cleavages": {"$ref": "inputs.missed_cleavages"},
                    "output_key": {"$ref": "inputs.output_key"},
                    "sequence": {"$ref": "steps.retrieve.result.sequence"},
                    "feature_annotations": {"$ref": "steps.retrieve.result.chain_features"},
                },
                evidence=(requirement,),
                description=(
                    "Digest the complete precursor first using the requested cleavage "
                    "rule, then retain only peptide intervals fully contained in the "
                    "annotated feature. Never digest an isolated feature subsequence."
                ),
            ),
            WorkflowNodeSpec(
                node_id="verify",
                executor="harness",
                operator="verify",
                depends_on=("digest_and_filter",),
                arguments={"evidence": {"$ref": "steps.digest_and_filter.result"}},
                evidence=(requirement,),
                description="Recompute full-sequence cleavage intervals and verify boundaries.",
            ),
        ),
        evidence_requirements=(requirement,),
        handoff_selectors=(
            HandoffSelector(
                name="precursor_sequence",
                paths=(
                    "steps.retrieve.result.sequence",
                    "steps.retrieve.result.data.sequence",
                ),
                required=True,
                max_matches=1,
            ),
            HandoffSelector(
                name="feature_annotations",
                paths=(
                    # Select feature records individually so a long annotation
                    # list cannot be collapsed into an unverifiable preview.
                    "steps.retrieve.result.chain_features.*",
                    "steps.retrieve.result.features.*",
                    "steps.retrieve.result.data.chain_features.*",
                    "steps.retrieve.result.data.features.*",
                ),
                required=True,
                max_matches=64,
            ),
        ),
        verification_contract={
            "verifier_id": "omniagent.scientific_evidence.v1",
            "require_full_sequence_first": True,
            "require_interval_filter": True,
            "derive_evidence_from_verified_output": True,
            "reject_prohibited_sole_sources": True,
        },
        completion_contract={
            "required_nodes": ["retrieve", "digest_and_filter", "verify"],
            "required_evidence": ["feature_bounded_proteolysis"],
            "pending_blocks_completion": True,
        },
        failure_policy={
            "missing_capability": "needs_review",
            "schema_drift": "needs_review",
            "a1_pending": "wait_external",
        },
        source_refs=(
            "https://github.com/openJiuwen-ai/sciencediscovery",
            "https://github.com/langchain-ai/langgraph",
        ),
    )


def context_sensitive_charge_v1() -> ScientificWorkflowSpec:
    """Return the reusable workflow for condition-dependent molecular charge."""
    requirement = EvidenceRequirement(
        property="molecular_net_charge",
        context="protein_bound",
        required_fields=(
            "value",
            "entity_id",
            "method",
            "method_class",
            "source_fields",
            "crystallization_ph",
            "protonation_sites",
            "atom_formal_charges",
        ),
        accepted_method_classes=("context_aware_computation",),
        prohibited_sole_sources=(
            "pdbx_formal_charge",
            "chem_comp.formal_charge",
            "context_free_metadata",
            "raw_descriptor",
        ),
        required_provenance=("source_fields", "method", "method_class"),
    )
    return ScientificWorkflowSpec(
        workflow_id="context_sensitive_charge",
        version="v1",
        intent_match={
            "scope": "step",
            "all_of": ["pdb"],
            "any_of": ["net charge", "molecular charge", "charge state"],
        },
        input_schema={"type": "object"},
        nodes=(
            WorkflowNodeSpec(
                node_id="structure",
                executor="mcp",
                operator="retrieve_artifact",
                capability_id="biomni.tool.database.query_pdb_identifiers",
                capability_query=(
                    "retrieve a PDB entry and its structure file by identifier"
                ),
                arguments={
                    "identifiers": [{"$ref": "inputs.pdb_id"}],
                    "return_type": "entry",
                    "download": True,
                },
                description="Resolve the PDB entry and retrieve the structure context.",
            ),
            WorkflowNodeSpec(
                node_id="chemical_context",
                executor="a1",
                operator="retrieve_record",
                depends_on=("structure",),
                arguments={
                    "pdb_id": {"$ref": "inputs.pdb_id"},
                    "structure": {"$ref": "steps.structure.result"},
                    "required_fields": ["ligand_id", "smiles", "crystallization_ph"],
                },
                description=(
                    "Identify the bound ligand, retrieve its SMILES, and retrieve the "
                    "experimental crystallization pH."
                ),
            ),
            WorkflowNodeSpec(
                node_id="protonate",
                executor="a1",
                operator="transform",
                depends_on=("chemical_context",),
                arguments={
                    "smiles": {"$ref": "steps.chemical_context.result.smiles"},
                    "pH": {"$ref": "steps.chemical_context.result.crystallization_ph"},
                    "transformation": (
                        "enumerate all ionizable sites and generate the protonated "
                        "state at the retrieved pH"
                    ),
                },
                description="Generate the ligand's condition-specific protonation state.",
            ),
            WorkflowNodeSpec(
                node_id="compute",
                executor="a1",
                operator="compute",
                depends_on=("chemical_context", "protonate"),
                arguments={
                    "ligand": {"$ref": "steps.chemical_context.result"},
                    "protonated_state": {"$ref": "steps.protonate.result"},
                    "calculation": (
                        "sum formal charges on every atom in the fully protonated "
                        "ligand state; do not stop at the single most basic site"
                    ),
                },
                evidence=(requirement,),
                description="Calculate net charge by summing atom-level formal charges.",
            ),
            WorkflowNodeSpec(
                node_id="verify",
                executor="harness",
                operator="verify",
                depends_on=("compute",),
                arguments={"evidence": {"$ref": "steps.compute.result"}},
                evidence=(requirement,),
                description="Reject deposited-charge-only results and verify provenance.",
            ),
        ),
        evidence_requirements=(requirement,),
        handoff_selectors=(
            HandoffSelector(
                name="entry_identifiers",
                paths=(
                    "steps.structure.result.**.rcsb_entry_container_identifiers",
                    "steps.structure.result.**.rcsb_id",
                ),
                required=True,
                max_matches=4,
            ),
            HandoffSelector(
                name="experimental_conditions",
                paths=(
                    "steps.structure.result.**.exptl_crystal_grow",
                    "steps.structure.result.**.exptl_crystal_grow_comp",
                ),
                required=True,
                max_matches=4,
            ),
            HandoffSelector(
                name="nonpolymer_entity_identifiers",
                paths=(
                    "steps.structure.result.**.rcsb_nonpolymer_entity_container_identifiers",
                    "steps.structure.result.**.pdbx_entity_nonpoly",
                    "steps.structure.result.**.non_polymer_entity_ids",
                ),
                max_matches=16,
            ),
        ),
        verification_contract={
            "verifier_id": "omniagent.scientific_evidence.v1",
            "require_atom_level_charge_sum": True,
            "reject_prohibited_sole_sources": True,
        },
        completion_contract={
            "required_nodes": [
                "structure",
                "chemical_context",
                "protonate",
                "compute",
                "verify",
            ],
            "required_evidence": ["molecular_net_charge"],
            "pending_blocks_completion": True,
        },
        failure_policy={
            "missing_capability": "needs_review",
            "schema_drift": "needs_review",
            "a1_pending": "wait_external",
        },
        source_refs=(
            "https://github.com/openJiuwen-ai/sciencediscovery",
            "https://github.com/langchain-ai/langgraph",
        ),
    )


def biomedical_kg_evidence_v1() -> ScientificWorkflowSpec:
    """Collect masked structural evidence for a biomedical KG link query."""
    return ScientificWorkflowSpec(
        workflow_id="biomedical_kg_evidence",
        version="v1",
        intent_match={
            "scope": "step",
            "operation": "retrieve",
            "any_of": [
                "kg evidence",
                "knowledge graph evidence",
                "enclosing subgraph",
                "metapath extraction",
            ],
        },
        input_schema={
            "type": "object",
            "properties": {
                "kg_path": {"type": "string"},
                "head_entity": {"type": "string"},
                "tail_entity": {"type": "string"},
                "max_hops": {"type": "integer"},
                "max_paths": {"type": "integer"},
            },
        },
        nodes=(
            WorkflowNodeSpec(
                node_id="load_kg",
                executor="mcp",
                operator="retrieve_artifact",
                capability_id="biomni.tool.knowledge_graph.load_biomedical_kg",
                arguments={
                    "kg_path": {"$ref": "inputs.kg_path"},
                    "format": "csv",
                    "delimiter": ",",
                    "has_header": True,
                    "schema": "primekg",
                    "use_cache": True,
                },
                description="Load the masked observed graph and report graph statistics.",
            ),
            WorkflowNodeSpec(
                node_id="enclosing_subgraph",
                executor="mcp",
                operator="retrieve_record",
                depends_on=("load_kg",),
                capability_id="biomni.tool.knowledge_graph.extract_enclosing_subgraph",
                arguments={
                    "kg_path": {"$ref": "inputs.kg_path"},
                    "head_entity": {"$ref": "inputs.head_entity"},
                    "tail_entity": {"$ref": "inputs.tail_entity"},
                    "max_hops": {"$ref": "inputs.max_hops"},
                    "max_nodes_per_hop": 200,
                    "remove_direct_link": True,
                    "bidirectional": True,
                },
                description=(
                    "Extract a bounded enclosing subgraph with the queried direct link "
                    "removed."
                ),
            ),
            WorkflowNodeSpec(
                node_id="metapaths",
                executor="mcp",
                operator="retrieve_record",
                depends_on=("load_kg",),
                capability_id="biomni.tool.knowledge_graph.extract_metapaths",
                arguments={
                    "kg_path": {"$ref": "inputs.kg_path"},
                    "head_entity": {"$ref": "inputs.head_entity"},
                    "tail_entity": {"$ref": "inputs.tail_entity"},
                    "max_length": {"$ref": "inputs.max_hops"},
                    "max_paths": {"$ref": "inputs.max_paths"},
                    "bidirectional": True,
                },
                description="Extract typed path patterns between the masked endpoint pair.",
            ),
        ),
        handoff_selectors=(
            HandoffSelector(
                name="graph_statistics",
                paths=("steps.load_kg.result",),
                required=True,
                max_matches=1,
            ),
            HandoffSelector(
                name="enclosing_subgraph",
                paths=("steps.enclosing_subgraph.result.result",),
                required=True,
                max_matches=1,
            ),
            HandoffSelector(
                name="metapath_patterns",
                paths=("steps.metapaths.result.result",),
                required=True,
                max_matches=1,
            ),
        ),
        completion_contract={
            "required_nodes": ["load_kg", "enclosing_subgraph", "metapaths"],
            "pending_blocks_completion": True,
        },
        failure_policy={
            "missing_capability": "needs_review",
            "schema_drift": "needs_review",
        },
    )


def biomedical_kg_link_prediction_v1() -> ScientificWorkflowSpec:
    """Infer one masked biomedical relation from previously verified KG evidence."""
    requirement = EvidenceRequirement(
        property="biomedical_kg_link_prediction",
        context="{head_entity} -> {tail_entity}",
        required_fields=(
            "head_entity",
            "tail_entity",
            "predicted_relation",
            "candidate_relations",
            "structural_paths",
            "relation_scores",
            "method",
            "method_class",
            "source_fields",
        ),
        accepted_method_classes=("masked_link_prediction_with_structural_evidence",),
        prohibited_sole_sources=("direct_edge", "memorized_fact", "unsupported_guess"),
        required_provenance=("source_fields", "method", "method_class"),
    )
    return ScientificWorkflowSpec(
        workflow_id="biomedical_kg_link_prediction",
        version="v1",
        intent_match={
            "scope": "step",
            "any_of": [
                "link prediction",
                "predict relation",
                "relation prediction",
                "infer relation",
            ],
        },
        input_schema={
            "type": "object",
            "properties": {
                "kg_path": {"type": "string"},
                "head_entity": {"type": "string"},
                "tail_entity": {"type": "string"},
                "candidate_relations": {"type": "array"},
                "ground_truth_path": {"type": "string"},
            },
        },
        nodes=(
            WorkflowNodeSpec(
                node_id="predict_relation",
                executor="a1",
                operator="compute",
                arguments={
                    "prediction_mode": "head_tail_relation",
                    "head_entity": {"$ref": "inputs.head_entity"},
                    "tail_entity": {"$ref": "inputs.tail_entity"},
                    "candidate_relations": {"$ref": "inputs.candidate_relations"},
                    "instruction": (
                        "Use only prior verified MCP graph evidence. Rank every candidate "
                        "relation, cite at least one observed multi-hop path, and do not use "
                        "a direct head-tail edge or factual recall as proof."
                    ),
                },
                evidence=(requirement,),
                description="Rank candidate relations from masked structural KG evidence.",
            ),
            WorkflowNodeSpec(
                node_id="verify_prediction",
                executor="harness",
                operator="verify",
                depends_on=("predict_relation",),
                arguments={"evidence": {"$ref": "steps.predict_relation.result"}},
                evidence=(requirement,),
                description=(
                    "Verify endpoint identity, observed paths, candidate scores, direct-edge "
                    "masking, and evaluator-private ground truth."
                ),
            ),
        ),
        evidence_requirements=(requirement,),
        verification_contract={
            "verifier_id": "omniagent.kg_link_prediction.v1",
            "derive_evidence_from_verified_output": True,
            "require_prior_mcp_evidence": True,
            "require_masked_direct_link": True,
            "min_structural_paths": 1,
            "private_input_keys": ["ground_truth_path"],
            "preconditions": {
                "prior_evidence": {
                    "source_backend": "mcp",
                    "capability_contains": "knowledge_graph",
                    "fallback_workflow_id": "biomedical_kg_evidence",
                }
            },
        },
        completion_contract={
            "required_nodes": ["predict_relation", "verify_prediction"],
            "required_evidence": ["biomedical_kg_link_prediction"],
            "pending_blocks_completion": True,
        },
        failure_policy={"a1_pending": "wait_external"},
    )


def default_domain_workflow_registry() -> DomainWorkflowRegistry:
    return DomainWorkflowRegistry(
        [
            biomedical_kg_evidence_v1(),
            biomedical_kg_link_prediction_v1(),
            feature_bounded_proteolysis_v1(),
            context_sensitive_charge_v1(),
            conditioned_property_v1(),
        ]
    )


def workflow_fingerprint(workflow: CompiledScientificWorkflow) -> str:
    encoded = json.dumps(workflow.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def biomni_result_contract(
    workflow_payload: dict[str, Any] | None,
    *,
    fallback_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the subset of Biomni's result contract that its gateway enforces."""
    required_paths: list[str] = []
    requirements = (
        workflow_payload.get("evidence_requirements", [])
        if isinstance(workflow_payload, dict)
        else []
    )
    if isinstance(requirements, list) and requirements:
        required_paths.append("completed_nodes")
        for index, requirement in enumerate(requirements, start=1):
            if not isinstance(requirement, dict):
                continue
            prefix = f"evidence.requirement_{index}"
            required_paths.extend((f"{prefix}.property", f"{prefix}.context"))
            for field_name in (
                *requirement.get("required_fields", []),
                *requirement.get("required_provenance", []),
            ):
                field_name = str(field_name or "").strip()
                if field_name:
                    required_paths.append(f"{prefix}.{field_name}")
    else:
        required_paths.extend(
            str(field).strip() for field in fallback_fields if str(field).strip()
        )
    return {
        "required_output_fields": list(dict.fromkeys(required_paths)),
    } if required_paths else {}


def _iter_handoff_values(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if path:
        yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_handoff_values(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_handoff_values(child, (*path, str(index)))


def _handoff_path_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    cache: dict[tuple[int, int], bool] = {}

    def match(pattern_index: int, path_index: int) -> bool:
        key = (pattern_index, path_index)
        if key in cache:
            return cache[key]
        if pattern_index == len(pattern):
            result = path_index == len(path)
        elif pattern[pattern_index] == "**":
            result = match(pattern_index + 1, path_index) or (
                path_index < len(path) and match(pattern_index, path_index + 1)
            )
        else:
            result = (
                path_index < len(path)
                and pattern[pattern_index] in {"*", path[path_index]}
                and match(pattern_index + 1, path_index + 1)
            )
        cache[key] = result
        return result

    return match(0, 0)


def _format_handoff_path(path: tuple[str, ...]) -> str:
    rendered = ""
    for part in path:
        if part.isdigit():
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + part
    return rendered


def project_workflow_evidence(
    workflow_payload: dict[str, Any],
    value: Any,
    *,
    artifacts: Iterable[str] = (),
    max_chars: int = 16000,
) -> Any:
    """Project declared MCP evidence into a bounded, provenance-carrying handoff."""
    if isinstance(value, dict) and value.get("schema_version") == "omniagent.handoff.v1":
        projected = dict(value)
        projected["artifact_refs"] = list(
            dict.fromkeys(
                [
                    *(str(item) for item in projected.get("artifact_refs", []) if str(item)),
                    *(str(item) for item in artifacts if str(item)),
                ]
            )
        )
        return projected

    raw_selectors = workflow_payload.get("handoff_selectors", [])
    if not isinstance(raw_selectors, list) or not raw_selectors:
        return compact_workflow_evidence(value, max_chars=max_chars)

    nodes = list(_iter_handoff_values(value))
    selections: dict[str, list[dict[str, Any]]] = {}
    missing_required: list[str] = []
    selector_required: dict[str, bool] = {}
    selector_count = max(1, len(raw_selectors))
    value_budget = max(512, min(4000, max_chars // (selector_count * 2)))
    for raw_selector in raw_selectors:
        if not isinstance(raw_selector, dict):
            continue
        name = str(raw_selector.get("name") or "").strip()
        raw_paths = raw_selector.get("paths", [])
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        patterns = [
            tuple(part for part in str(item).split(".") if part)
            for item in raw_paths
            if str(item).strip()
        ] if isinstance(raw_paths, list) else []
        if not name or not patterns:
            continue
        required = bool(raw_selector.get("required", False))
        selector_required[name] = required
        max_matches = max(1, int(raw_selector.get("max_matches", 8)))
        matched_paths: set[tuple[str, ...]] = set()
        records: list[dict[str, Any]] = []
        for path, selected_value in nodes:
            if path in matched_paths or not any(
                _handoff_path_matches(pattern, path) for pattern in patterns
            ):
                continue
            matched_paths.add(path)
            records.append(
                {
                    "source_path": _format_handoff_path(path),
                    "value": compact_workflow_evidence(
                        selected_value,
                        max_chars=value_budget,
                    ),
                }
            )
            if len(records) >= max_matches:
                break
        selections[name] = records
        if required and not records:
            missing_required.append(name)

    projected: dict[str, Any] = {
        "schema_version": "omniagent.handoff.v1",
        "workflow_id": str(
            workflow_payload.get("qualified_id")
            or workflow_payload.get("workflow_id")
            or "domain-workflow"
        ),
        "selections": selections,
        "artifact_refs": list(
            dict.fromkeys(str(item) for item in artifacts if str(item))
        ),
        "missing_required_selectors": missing_required,
    }

    def encoded_size() -> int:
        return len(json.dumps(projected, ensure_ascii=False, default=str))

    while encoded_size() > max_chars:
        reducible = [
            (name, records)
            for name, records in selections.items()
            if len(records) > 1
        ]
        if not reducible:
            break
        name, records = max(
            reducible,
            key=lambda item: (
                not selector_required.get(item[0], False),
                len(item[1]),
            ),
        )
        records.pop()
        projected["truncated"] = True

    if encoded_size() > max_chars:
        preview_budget = max(128, max_chars // max(1, sum(map(len, selections.values()))) - 160)
        for records in selections.values():
            for record in records:
                encoded = json.dumps(record["value"], ensure_ascii=False, default=str)
                if len(encoded) > preview_budget:
                    record["value"] = {
                        "json_preview": encoded[:preview_budget] + "...[truncated]"
                    }
                    projected["truncated"] = True
    return projected


def workflow_execution_instruction(
    workflow_payload: dict[str, Any],
    *,
    mcp_evidence: Any | None = None,
) -> str:
    """Render provider-neutral workflow metadata into a bounded A1 instruction."""
    workflow_id = str(workflow_payload.get("qualified_id") or "domain-workflow")
    nodes = workflow_payload.get("nodes", [])
    full_a1_workflow = workflow_payload.get("execution_ownership") == "a1_full_workflow"
    a1_nodes = [
        node
        for node in nodes
        if isinstance(node, dict)
        and (node.get("executor") == "a1" or (full_a1_workflow and node.get("executor") != "harness"))
    ] if isinstance(nodes, list) else []
    harness_nodes = [
        str(node.get("node_id"))
        for node in nodes
        if isinstance(node, dict) and node.get("executor") == "harness"
    ] if isinstance(nodes, list) else []
    requirements = workflow_payload.get("evidence_requirements", [])
    expected_evidence: dict[str, Any] = {}
    if isinstance(requirements, list):
        for index, requirement in enumerate(requirements, start=1):
            if not isinstance(requirement, dict):
                continue
            record = {
                "property": requirement.get("property"),
                "context": requirement.get("context"),
            }
            for field_name in (
                *requirement.get("required_fields", []),
                *requirement.get("required_provenance", []),
            ):
                field_name = str(field_name or "").strip()
                if field_name:
                    record.setdefault(field_name, f"<required:{field_name}>")
            record["_constraints"] = {
                "accepted_method_classes": requirement.get(
                    "accepted_method_classes", []
                ),
                "prohibited_sole_sources": requirement.get(
                    "prohibited_sole_sources", []
                ),
            }
            expected_evidence[f"requirement_{index}"] = record
    instruction = {
        "workflow_id": workflow_id,
        "a1_nodes": [
            {
                "node_id": node.get("node_id"),
                "operator": node.get("operator"),
                "depends_on": node.get("depends_on", []),
                "arguments": node.get("arguments", {}),
                "description": node.get("description", ""),
            }
            for node in a1_nodes
        ],
        "harness_owned_nodes": harness_nodes,
        "output_guidance": {
            "completed_nodes": [node.get("node_id") for node in a1_nodes],
            "evidence": expected_evidence,
        },
    }
    ownership_instruction = (
        "Execute all declared non-Harness nodes in this workflow using A1's available "
        "tools and internal planning. No MCP retrieval stage was admitted by OmniAgent. "
        if full_a1_workflow
        else "Execute only the declared A1 nodes in this compiled workflow. MCP retrieval "
        "nodes have already been executed by the Harness; use their structured evidence "
        "and do not repeat them. "
    )
    parts = [
        ownership_instruction
        + "Harness-owned verification nodes must not be simulated. "
        "Do not inspect Harness event or checkpoint logs to recover omitted data. If "
        "missing_required_selectors is non-empty, return structured INSUFFICIENT_EVIDENCE "
        "instead of guessing. Return the best structured JSON evidence you can establish "
        "from the executed nodes; use output_guidance as a suggestion, omit unknown fields, "
        "and never fabricate missing evidence. OmniAgent performs the final completeness "
        "verification locally.",
        json.dumps(instruction, ensure_ascii=False, default=str),
    ]
    if mcp_evidence is not None:
        projected_evidence = project_workflow_evidence(
            workflow_payload,
            mcp_evidence,
        )
        parts.append(
            "MCP_EVIDENCE:\n"
            + json.dumps(
                projected_evidence,
                ensure_ascii=False,
                default=str,
            )
        )
    return "\n".join(parts)


def compact_workflow_evidence(value: Any, *, max_chars: int = 16000) -> Any:
    """Bound external evidence before it enters the A1 model context."""
    def compact(item: Any, depth: int = 0) -> Any:
        if depth >= 8:
            return "...[max depth]"
        if isinstance(item, dict):
            pairs = list(item.items())
            result = {
                str(key): compact(child, depth + 1)
                for key, child in pairs[:40]
            }
            if len(pairs) > 40:
                result["_truncated_fields"] = len(pairs) - 40
            return result
        if isinstance(item, (list, tuple)):
            values = list(item)
            result = [compact(child, depth + 1) for child in values[:20]]
            if len(values) > 20:
                result.append({"_truncated_items": len(values) - 20})
            return result
        if isinstance(item, str) and len(item) > 2000:
            return item[:2000] + "...[truncated]"
        return item

    compacted = compact(value)
    encoded = json.dumps(compacted, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return compacted
    return {
        "truncated": True,
        "json_preview": encoded[:max_chars] + "...[truncated]",
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        for key in ("value", "position", "residue", "index"):
            parsed = _as_int(value.get(key))
            if parsed is not None:
                return parsed
    try:
        text = str(value).strip()
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def _normalize_sequence(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("value", "sequence", "seq"):
            if key in value:
                return _normalize_sequence(value[key])
    return re.sub(r"\s+", "", str(value or "")).upper()


def _feature_label(feature: dict[str, Any]) -> str:
    values = [
        feature.get(key, "")
        for key in ("name", "type", "description", "id", "feature", "label")
    ]
    return " ".join(str(value) for value in values if value).casefold()


def _normalize_label(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
    )


def _label_matches(value: Any, expected: Any) -> bool:
    actual_label = _normalize_label(value)
    expected_label = _normalize_label(expected)
    if not actual_label or not expected_label:
        return actual_label == expected_label
    return (
        actual_label == expected_label
        or f" {expected_label} " in f" {actual_label} "
        or f" {actual_label} " in f" {expected_label} "
    )


def _feature_interval(
    value: Any,
    feature_name: str,
) -> tuple[int, int] | None:
    """Extract one named 1-based inclusive feature interval from API variants."""
    expected = _normalize_label(feature_name)
    candidates: list[tuple[dict[str, Any], str]] = []

    def collect(item: Any, inherited_label: str = "") -> None:
        if isinstance(item, dict):
            label = " ".join(part for part in (inherited_label, _feature_label(item)) if part)
            if any(
                key in item
                for key in ("start", "begin", "feature_start", "start_1based")
            ) and any(
                key in item
                for key in ("end", "stop", "feature_end", "end_1based")
            ):
                candidates.append((item, label))
            for child in item.values():
                collect(child, label)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child, inherited_label)

    collect(value)
    labeled = [
        (item, label)
        for item, label in candidates
        if expected and _label_matches(label, expected)
    ]
    selected = labeled if labeled else candidates if len(candidates) == 1 else []
    for item, _label in selected:
        start = _as_int(
            item.get(
                "start",
                item.get(
                    "begin",
                    item.get("feature_start", item.get("start_1based")),
                ),
            )
        )
        end = _as_int(
            item.get(
                "end",
                item.get(
                    "stop",
                    item.get("feature_end", item.get("end_1based")),
                ),
            )
        )
        if start is not None and end is not None and 1 <= start <= end:
            return start, end
    return None


def _handoff_selection(handoff: Any, names: Iterable[str]) -> Any:
    if not isinstance(handoff, dict):
        return None
    selections = handoff.get("selections")
    if not isinstance(selections, dict):
        return None
    for name in names:
        records = selections.get(name)
        if not isinstance(records, list) or not records:
            continue
        values = [
            record["value"] if isinstance(record, dict) and "value" in record else record
            for record in records
        ]
        return values[0] if len(values) == 1 else values
    return None


def _cleavage_intervals(
    sequence: str,
    cleavage_residues: Iterable[Any],
    missed_cleavages: int,
) -> list[dict[str, Any]]:
    residues = {str(item).strip().upper() for item in cleavage_residues if str(item).strip()}
    boundaries = [
        0,
        *(
            index
            for index, residue in enumerate(sequence, start=1)
            if residue in residues
        ),
        len(sequence),
    ]
    boundaries = list(dict.fromkeys(boundaries))
    peptides: list[dict[str, Any]] = []
    max_segments = max(1, missed_cleavages + 1)
    for start_index in range(len(boundaries) - 1):
        for segment_count in range(1, max_segments + 1):
            end_index = start_index + segment_count
            if end_index >= len(boundaries):
                break
            start = boundaries[start_index] + 1
            end = boundaries[end_index]
            if start > end:
                continue
            peptides.append(
                {"sequence": sequence[start - 1 : end], "start": start, "end": end}
            )
    return peptides


def _normalize_peptide_records(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        sequence = _normalize_sequence(item.get("sequence", item.get("peptide")))
        start = _as_int(
            item.get("start", item.get("begin", item.get("start_1based")))
        )
        end = _as_int(item.get("end", item.get("stop", item.get("end_1based"))))
        if not sequence or start is None or end is None:
            return None
        normalized.append({"sequence": sequence, "start": start, "end": end})
    return normalized


def _normalize_kg_prediction_result(
    workflow_payload: dict[str, Any],
    result: Any,
) -> bool:
    if not getattr(result, "success", False) or not isinstance(
        getattr(result, "output", None), dict
    ):
        return False

    def find_prediction(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            if any(key in value for key in ("predicted_relation", "prediction", "relation")):
                return value
            for child in value.values():
                found = find_prediction(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = find_prediction(child)
                if found is not None:
                    return found
        return None

    output = dict(result.output)
    candidate = find_prediction(output)
    if candidate is None:
        return False
    record = dict(candidate)
    predicted = record.get(
        "predicted_relation",
        record.get("prediction", record.get("relation")),
    )
    if not str(predicted or "").strip():
        return False
    inputs = workflow_payload.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    head = str(inputs.get("head_entity", "")).strip()
    tail = str(inputs.get("tail_entity", "")).strip()
    record.setdefault("property", "biomedical_kg_link_prediction")
    record.setdefault("context", f"{head} -> {tail}")
    record.setdefault("head_entity", head)
    record.setdefault("tail_entity", tail)
    record["predicted_relation"] = predicted
    record.setdefault("candidate_relations", inputs.get("candidate_relations", []))
    if "structural_paths" not in record:
        for alias in ("paths", "metapaths", "supporting_paths"):
            if alias in record:
                record["structural_paths"] = record[alias]
                break
    if "relation_scores" not in record:
        for alias in ("scores", "candidate_scores", "ranked_relations"):
            if alias in record:
                record["relation_scores"] = record[alias]
                break

    existing = output.get("evidence", [])
    evidence = list(existing) if isinstance(existing, list) else [existing] if existing else []
    if not any(
        isinstance(item, dict)
        and item.get("property") == "biomedical_kg_link_prediction"
        for item in evidence
    ):
        evidence.append(record)
    output["evidence"] = evidence
    output.setdefault("completed_nodes", ["predict_relation"])
    result.output = output
    if getattr(result, "verification_payload", None) is not None:
        result.verification_payload = output
    return True


def normalize_domain_result(workflow_payload: dict[str, Any], result: Any) -> bool:
    """Add verifier-derived evidence when a workflow can prove a compact output.

    Biomni is allowed to return the final computed value without echoing every
    input and intermediate field. A workflow-specific normalizer may derive the
    evidence only from the persisted MCP handoff and then compare the returned
    value with a deterministic recomputation.
    """
    contract = workflow_payload.get("verification_contract", {})
    if not isinstance(contract, dict) or not contract.get(
        "derive_evidence_from_verified_output"
    ):
        return False
    if workflow_payload.get("workflow_id") == "biomedical_kg_link_prediction":
        return _normalize_kg_prediction_result(workflow_payload, result)
    if workflow_payload.get("workflow_id") != "feature_bounded_proteolysis":
        return False
    if not getattr(result, "success", False) or not isinstance(
        getattr(result, "output", None), dict
    ):
        return False

    output = dict(result.output)
    inputs = workflow_payload.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    output_key = str(inputs.get("output_key", "")).strip()
    candidate = output.get(output_key) if output_key else None
    if isinstance(candidate, dict):
        candidate = candidate.get("sequence", candidate.get("value"))
    candidate = _normalize_sequence(candidate)
    if not candidate:
        return False

    metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
    stage = metadata.get("domain_workflow_stage")
    handoff = stage.get("mcp_evidence") if isinstance(stage, dict) else None
    sequence = _normalize_sequence(
        _handoff_selection(handoff, ("precursor_sequence",))
    )
    features = _handoff_selection(handoff, ("feature_annotations",))
    feature_name = str(inputs.get("feature_name", "")).strip()
    interval = _feature_interval(features, feature_name)
    cleavage_residues = inputs.get("cleavage_residues", [])
    missed_cleavages = _as_int(inputs.get("missed_cleavages"))
    missed_cleavages = 0 if missed_cleavages is None else missed_cleavages
    if (
        not sequence
        or interval is None
        or not isinstance(cleavage_residues, (list, tuple))
        or missed_cleavages < 0
    ):
        return False

    all_peptides = _cleavage_intervals(
        sequence,
        cleavage_residues,
        missed_cleavages,
    )
    retained_peptides = [
        item
        for item in all_peptides
        if item["start"] >= interval[0] and item["end"] <= interval[1]
    ]
    expected = max(
        (item["sequence"] for item in retained_peptides),
        key=len,
        default="",
    )
    if candidate != expected or not retained_peptides:
        return False

    source_fields: list[str] = []
    selections = handoff.get("selections") if isinstance(handoff, dict) else None
    if isinstance(selections, dict):
        for name in ("precursor_sequence", "feature_annotations"):
            records = selections.get(name, [])
            if not isinstance(records, list):
                continue
            source_fields.extend(
                str(record.get("source_path"))
                for record in records
                if isinstance(record, dict) and str(record.get("source_path", "")).strip()
            )
    source_fields = list(dict.fromkeys(source_fields))
    if not source_fields:
        return False

    method_class = "full_sequence_cleavage_then_interval_filter"
    method = (
        "Digest the complete precursor at the C-terminus of K/R, then retain "
        "peptides fully contained in the annotated feature interval."
    )
    derived_record = {
        "property": "feature_bounded_proteolysis",
        "context": feature_name,
        "accession": str(inputs.get("uniprot_accession", "")).strip(),
        "feature_name": feature_name,
        "feature_start": interval[0],
        "feature_end": interval[1],
        "precursor_sequence": sequence,
        "cleavage_residues": list(cleavage_residues),
        "missed_cleavages": missed_cleavages,
        "all_peptides": all_peptides,
        "retained_peptides": retained_peptides,
        "longest_peptide_sequence": expected,
        "method": method,
        "method_class": method_class,
        "source_fields": source_fields,
        "derived_by": "omniagent.feature_bounded_proteolysis.verifier.v1",
    }
    existing = output.get("evidence", [])
    evidence = list(existing) if isinstance(existing, list) else [existing] if existing else []
    if not any(
        isinstance(item, dict)
        and item.get("property") == derived_record["property"]
        and item.get("context") == derived_record["context"]
        for item in evidence
    ):
        evidence.append(derived_record)
    output["evidence"] = evidence
    result.output = output
    if getattr(result, "verification_payload", None) is not None:
        result.verification_payload = output
    metadata = dict(metadata)
    metadata["domain_evidence_normalization"] = {
        "workflow_id": workflow_payload.get("qualified_id", "feature_bounded_proteolysis@v1"),
        "method": "deterministic_recomputation_from_mcp_handoff",
        "source_fields": source_fields,
    }
    result.task_metadata = metadata
    return True


def _verify_feature_bounded_proteolysis(
    workflow_payload: dict[str, Any],
    result: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    inputs = workflow_payload.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    expected_accession = str(inputs.get("uniprot_accession", "")).strip().upper()
    expected_feature = str(inputs.get("feature_name", "")).strip()
    expected_residues = inputs.get("cleavage_residues", [])
    expected_missed = _as_int(inputs.get("missed_cleavages"))
    expected_missed = 0 if expected_missed is None else expected_missed
    errors: list[str] = []
    evidence = next(
        (
            record
            for record in records
            if str(record.get("property", "")).strip() == "feature_bounded_proteolysis"
            and _label_matches(record.get("context", ""), expected_feature)
        ),
        None,
    )
    if evidence is None:
        return ["feature-bounded evidence record is missing"]

    sequence = _normalize_sequence(evidence.get("precursor_sequence"))
    if not sequence:
        errors.append("precursor_sequence is empty")
    if expected_accession and str(evidence.get("accession", "")).strip().upper() != expected_accession:
        errors.append("evidence accession does not match the MCP request")
    if not _label_matches(evidence.get("feature_name", ""), expected_feature):
        errors.append("evidence feature_name does not match the requested feature")
    feature_start = _as_int(evidence.get("feature_start"))
    feature_end = _as_int(evidence.get("feature_end"))
    if feature_start is None or feature_end is None or feature_start > feature_end:
        errors.append("feature interval is missing or invalid")
    elif sequence and feature_end > len(sequence):
        errors.append("feature interval falls outside the precursor sequence")

    handoff = None
    metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
    stage = metadata.get("domain_workflow_stage")
    if isinstance(stage, dict):
        handoff = stage.get("mcp_evidence")
    full_a1_workflow = (
        workflow_payload.get("execution_ownership") == "a1_full_workflow"
    )
    if not full_a1_workflow:
        mcp_sequence = _normalize_sequence(
            _handoff_selection(handoff, ("precursor_sequence",))
        )
        if mcp_sequence and sequence and mcp_sequence != sequence:
            errors.append("A1 precursor_sequence does not match the MCP evidence")
        mcp_features = _handoff_selection(handoff, ("feature_annotations",))
        mcp_interval = _feature_interval(mcp_features, expected_feature)
        if mcp_interval is None:
            errors.append("MCP evidence does not contain the requested feature interval")
        elif (feature_start, feature_end) != mcp_interval:
            errors.append("A1 feature interval does not match the MCP evidence")

    if not isinstance(expected_residues, (list, tuple)) or not expected_residues:
        errors.append("cleavage_residues is missing")
    if expected_missed < 0:
        errors.append("missed_cleavages must be non-negative")
    if sequence and feature_start is not None and feature_end is not None and expected_missed >= 0:
        expected_peptides = _cleavage_intervals(sequence, expected_residues, expected_missed)
        expected_retained = [
            item
            for item in expected_peptides
            if item["start"] >= feature_start and item["end"] <= feature_end
        ]
        actual_all = _normalize_peptide_records(evidence.get("all_peptides"))
        actual_retained = _normalize_peptide_records(evidence.get("retained_peptides"))
        if actual_all != expected_peptides:
            errors.append("all_peptides is not the full-sequence cleavage result")
        if actual_retained != expected_retained:
            errors.append("retained_peptides does not equal interval-filtered full peptides")
        longest = str(evidence.get("longest_peptide_sequence", "")).strip().upper()
        expected_longest = max(
            (item["sequence"] for item in expected_retained),
            key=len,
            default="",
        )
        if longest != expected_longest:
            errors.append("longest_peptide_sequence is not the longest retained peptide")
        if not actual_all or not actual_retained:
            errors.append("full peptide enumeration and retained peptide list are required")

    method_class = str(evidence.get("method_class", "")).casefold().strip()
    if method_class != "full_sequence_cleavage_then_interval_filter":
        errors.append("method_class does not prove full-sequence cleavage before filtering")
    method = str(evidence.get("method", "")).casefold()
    if "isolated" in method or "subsequence" in method:
        errors.append("method indicates isolated-feature digestion")
    source_fields = evidence.get("source_fields", [])
    if isinstance(source_fields, str):
        source_fields = [source_fields]
    if not isinstance(source_fields, list | tuple) or not source_fields:
        errors.append("source_fields provenance is missing")
    elif full_a1_workflow:
        source_text = " ".join(str(item) for item in source_fields).casefold()
        if expected_accession and expected_accession.casefold() not in source_text:
            errors.append("A1 provenance does not identify the requested accession")
        if "sequence" not in source_text:
            errors.append("A1 provenance does not identify the precursor sequence source")
        if not any(token in source_text for token in ("feature", "chain", "coordinate")):
            errors.append("A1 provenance does not identify the feature interval source")
    return errors


def _iter_structural_paths(value: Any) -> Iterable[list[str]]:
    if isinstance(value, dict):
        for key in ("nodes", "path", "entity_path"):
            path = value.get(key)
            if isinstance(path, (list, tuple)) and path and all(
                isinstance(item, str) for item in path
            ):
                yield [str(item).strip() for item in path]
        for child in value.values():
            yield from _iter_structural_paths(child)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, str) for item in value):
            yield [str(item).strip() for item in value]
        else:
            for child in value:
                yield from _iter_structural_paths(child)


def _kg_relation_scores(value: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(value, dict):
        for relation, raw_score in value.items():
            score = raw_score.get("score") if isinstance(raw_score, dict) else raw_score
            try:
                scores[str(relation).strip().casefold()] = float(score)
            except (TypeError, ValueError):
                continue
    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, dict):
                continue
            relation = item.get("relation", item.get("label", item.get("candidate")))
            score = item.get("score", item.get("confidence", item.get("probability")))
            try:
                scores[str(relation).strip().casefold()] = float(score)
            except (TypeError, ValueError):
                continue
    return scores


def _load_observed_kg(
    kg_path: str,
) -> tuple[set[tuple[str, str]], dict[str, set[str]], list[str]]:
    edges: set[tuple[str, str]] = set()
    adjacency: dict[str, set[str]] = {}
    errors: list[str] = []
    path = Path(kg_path).expanduser()
    if not path.is_file():
        return edges, adjacency, ["observed KG file is missing"]
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            required = {"x_id", "x_type", "relation", "y_id", "y_type"}
            if not required.issubset(fields):
                return edges, adjacency, ["observed KG does not use the declared PrimeKG schema"]
            for row in reader:
                head = f"{str(row.get('x_type', '')).strip()}:{str(row.get('x_id', '')).strip()}"
                tail = f"{str(row.get('y_type', '')).strip()}:{str(row.get('y_id', '')).strip()}"
                if head == ":" or tail == ":":
                    continue
                edges.add((head, tail))
                adjacency.setdefault(head, set()).add(tail)
                adjacency.setdefault(tail, set()).add(head)
    except (OSError, csv.Error) as exc:
        errors.append(f"observed KG could not be parsed: {type(exc).__name__}")
    return edges, adjacency, errors


def _verify_biomedical_kg_link_prediction(
    workflow_payload: dict[str, Any],
    records: list[dict[str, Any]],
    request: Any | None,
) -> list[str]:
    inputs = workflow_payload.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    head = str(inputs.get("head_entity", "")).strip()
    tail = str(inputs.get("tail_entity", "")).strip()
    allowed_relations = {
        str(item).strip().casefold()
        for item in inputs.get("candidate_relations", [])
        if str(item).strip()
    }
    evidence = next(
        (
            record
            for record in records
            if str(record.get("property", record.get("property_name", ""))).strip()
            == "biomedical_kg_link_prediction"
        ),
        None,
    )
    if evidence is None:
        return ["KG link-prediction evidence record is missing"]

    errors: list[str] = []
    if str(evidence.get("head_entity", "")).strip() != head:
        errors.append("prediction head_entity does not match the requested entity")
    if str(evidence.get("tail_entity", "")).strip() != tail:
        errors.append("prediction tail_entity does not match the requested entity")
    predicted = str(
        evidence.get("predicted_relation", evidence.get("relation", ""))
    ).strip()
    predicted_key = predicted.casefold()
    if not predicted:
        errors.append("predicted_relation is missing")
    elif allowed_relations and predicted_key not in allowed_relations:
        errors.append("predicted_relation is outside candidate_relations")

    scores = _kg_relation_scores(evidence.get("relation_scores"))
    if not scores:
        errors.append("relation_scores contains no numeric candidate scores")
    elif predicted_key not in scores:
        errors.append("relation_scores does not score the predicted relation")

    kg_path = str(inputs.get("kg_path", "")).strip()
    edges, adjacency, graph_errors = _load_observed_kg(kg_path)
    errors.extend(graph_errors)
    if (head, tail) in edges or (tail, head) in edges:
        errors.append("observed KG leaks the queried direct head-tail edge")
    valid_paths = 0
    for path in _iter_structural_paths(evidence.get("structural_paths")):
        if len(path) < 3 or (path[0], path[-1]) not in {(head, tail), (tail, head)}:
            continue
        if all(right in adjacency.get(left, set()) for left, right in zip(path, path[1:])):
            valid_paths += 1
    contract = workflow_payload.get("verification_contract", {})
    contract = contract if isinstance(contract, dict) else {}
    minimum_paths = max(1, int(contract.get("min_structural_paths", 1)))
    if valid_paths < minimum_paths:
        errors.append(
            f"fewer than {minimum_paths} structural paths are present in the observed KG"
        )

    if contract.get("require_prior_mcp_evidence"):
        prior = getattr(request, "prior_observations", None) if request is not None else None
        prior_text = json.dumps(prior or [], ensure_ascii=False, default=str).casefold()
        if "mcp" not in prior_text or "knowledge_graph" not in prior_text:
            errors.append("no prior verified KG MCP evidence was supplied to the prediction round")

    truth_path = str(
        inputs.get("ground_truth_path")
        or os.getenv("OMNIAGENT_KG_LINK_GROUND_TRUTH_PATH", "")
    ).strip()
    if not truth_path:
        errors.append("evaluator-private KG ground truth is not configured")
    else:
        try:
            truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"KG ground truth could not be read: {type(exc).__name__}")
        else:
            if not isinstance(truth, dict):
                errors.append("KG ground truth is not an object")
            else:
                if str(truth.get("head_entity", "")).strip() != head or str(
                    truth.get("tail_entity", "")
                ).strip() != tail:
                    errors.append("KG ground truth endpoints do not match the request")
                expected = str(truth.get("relation", "")).strip().casefold()
                if not expected or predicted_key != expected:
                    errors.append("predicted_relation does not match evaluator ground truth")
    return errors


def verify_domain_evidence(
    workflow_payload: dict[str, Any],
    result: Any,
    request: Any | None = None,
) -> list[str]:
    """Check explicit domain evidence without treating free-form answers as proof."""
    raw_requirements = workflow_payload.get("evidence_requirements", [])
    if not isinstance(raw_requirements, list):
        return ["domain workflow evidence_requirements is not a list"]
    records: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "property" in value or "property_name" in value:
                records.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(getattr(result, "output", None))
    collect(getattr(result, "observations", None))
    collect(getattr(result, "raw", None))
    errors: list[str] = []
    for raw in raw_requirements:
        if not isinstance(raw, dict):
            errors.append("domain evidence requirement is not an object")
            continue
        property_name = str(raw.get("property", "")).strip()
        context = str(raw.get("context", "")).strip()
        required_fields = [str(item) for item in raw.get("required_fields", [])]
        accepted_methods = {str(item).casefold() for item in raw.get("accepted_method_classes", [])}
        prohibited_sources = {str(item).casefold() for item in raw.get("prohibited_sole_sources", [])}
        provenance_fields = [str(item) for item in raw.get("required_provenance", [])]
        matched = False
        for record in records:
            record_property = str(record.get("property", record.get("property_name", ""))).strip()
            record_context = str(record.get("context", "")).strip()
            if record_property != property_name or not _label_matches(
                record_context, context
            ):
                continue
            if any(field not in record for field in required_fields):
                continue
            method_class = str(record.get("method_class", "")).casefold()
            if accepted_methods and method_class not in accepted_methods:
                continue
            provenance = record.get("provenance", record)
            if not isinstance(provenance, dict) or any(
                field not in record and field not in provenance for field in provenance_fields
            ):
                continue
            sources = record.get("source_fields", record.get("sources", []))
            if isinstance(sources, str):
                sources = [sources]
            normalized_sources = {str(item).casefold() for item in sources or []}
            if normalized_sources and normalized_sources.issubset(prohibited_sources):
                continue
            matched = True
            break
        if not matched:
            errors.append(
                f"domain evidence requirement not satisfied: {property_name} in {context}"
            )
    contract = workflow_payload.get("verification_contract", {})
    if (
        isinstance(contract, dict)
        and contract.get("require_full_sequence_first")
        and workflow_payload.get("workflow_id") == "feature_bounded_proteolysis"
    ):
        errors.extend(_verify_feature_bounded_proteolysis(workflow_payload, result, records))
    if (
        isinstance(contract, dict)
        and contract.get("verifier_id") == "omniagent.kg_link_prediction.v1"
        and workflow_payload.get("workflow_id") == "biomedical_kg_link_prediction"
    ):
        errors.extend(
            _verify_biomedical_kg_link_prediction(workflow_payload, records, request)
        )
    return errors
