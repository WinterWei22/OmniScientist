from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def stage_request_id(
    parent_request_id: str,
    *,
    stage: str,
    identity: str,
) -> str:
    """Derive a stable external request identity without replacing its parent."""
    parent = str(parent_request_id or "").strip()
    stage_name = str(stage or "").strip().lower()
    if not parent:
        raise ValueError("parent_request_id is required")
    if not stage_name:
        raise ValueError("stage is required")
    digest = hashlib.sha256(
        f"{parent}\x1f{stage_name}\x1f{identity}".encode("utf-8")
    ).hexdigest()[:20]
    suffix = f":stage:{stage_name}:{digest}"
    return f"{parent[: 256 - len(suffix)]}{suffix}"


class ExecutionBackend(str, Enum):
    A1 = "a1"
    MCP = "mcp"
    UNAVAILABLE = "unavailable"


class SemanticOperation(str, Enum):
    RETRIEVE = "retrieve"
    VALIDATE = "validate"
    ANALYZE = "analyze"
    EXPERIMENT = "experiment"
    GENERATE_ARTIFACT = "generate_artifact"
    SYNTHESIZE = "synthesize"

    @classmethod
    def normalize(cls, value: Any) -> SemanticOperation:
        normalized = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "compute": cls.ANALYZE.value,
            "calculate": cls.ANALYZE.value,
            "calculation": cls.ANALYZE.value,
            "analysis": cls.ANALYZE.value,
            "lookup": cls.RETRIEVE.value,
            "search": cls.RETRIEVE.value,
            "verification": cls.VALIDATE.value,
            "verify": cls.VALIDATE.value,
        }
        return cls(aliases.get(normalized, normalized))


class ExecutionShape(str, Enum):
    SINGLE_CAPABILITY = "single_capability"
    MULTI_CAPABILITY = "multi_capability"
    ADAPTIVE = "adaptive"


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"


@dataclass(frozen=True, slots=True)
class SemanticCapabilityIntent:
    operation: SemanticOperation
    capability_query: str
    execution_shape: ExecutionShape
    schema_bound: bool
    side_effect: SideEffect
    rationale: str = ""
    required_output_fields: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    entity_context: dict[str, str] = field(default_factory=dict)
    capability_hint: str = ""

    @classmethod
    def from_inputs(
        cls, inputs: dict[str, Any]
    ) -> SemanticCapabilityIntent | None:
        raw = inputs.get("semantic_intent")
        if not isinstance(raw, dict):
            return None
        try:
            operation = SemanticOperation.normalize(raw.get("operation"))
            execution_shape = ExecutionShape(
                str(raw.get("execution_shape", "")).lower()
            )
            side_effect = SideEffect(str(raw.get("side_effect", "read_only")).lower())
        except ValueError:
            return None
        schema_bound_value = raw.get("schema_bound", False)
        if isinstance(schema_bound_value, bool):
            schema_bound = schema_bound_value
        elif isinstance(schema_bound_value, str) and schema_bound_value.lower() in {
            "true",
            "false",
        }:
            schema_bound = schema_bound_value.lower() == "true"
        else:
            return None
        query = str(
            raw.get("capability_query") or inputs.get("tool_query") or ""
        ).strip()
        if not query:
            return None
        return cls(
            operation=operation,
            capability_query=query,
            execution_shape=execution_shape,
            schema_bound=schema_bound,
            side_effect=side_effect,
            rationale=str(raw.get("rationale", "")).strip(),
            required_output_fields=cls._strings(raw.get("required_output_fields")),
            expected_artifacts=cls._strings(raw.get("expected_artifacts")),
            entity_context=cls._string_mapping(raw.get("entity_context")),
            capability_hint=str(raw.get("capability_hint", "")).strip(),
        )

    @staticmethod
    def _strings(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list | tuple):
            return ()
        return tuple(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )

    @staticmethod
    def _string_mapping(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key).strip(): str(item).strip()
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "capability_query": self.capability_query,
            "execution_shape": self.execution_shape.value,
            "schema_bound": self.schema_bound,
            "side_effect": self.side_effect.value,
            "rationale": self.rationale,
            "required_output_fields": list(self.required_output_fields),
            "expected_artifacts": list(self.expected_artifacts),
            "entity_context": dict(self.entity_context),
            "capability_hint": self.capability_hint,
        }


@dataclass(slots=True)
class ResourceCandidate:
    qualified_name: str
    description: str = ""
    score: float | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    capability_version: str = ""
    effect_contract: dict[str, Any] = field(default_factory=dict)
    result_adapter: str = "generic"
    execution_mode: str = "sync"
    lifecycle: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    timeout_policy: dict[str, Any] = field(default_factory=dict)
    idempotency_policy: dict[str, Any] = field(default_factory=dict)
    provenance_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PathValueRequirement:
    """Require an execution result path to contain one of the declared values."""

    path: str
    expected_values: tuple[str, ...]
    case_sensitive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected_values": list(self.expected_values),
            "case_sensitive": self.case_sensitive,
        }


@dataclass(frozen=True, slots=True)
class EffectContract:
    required_paths: tuple[str, ...] = ()
    any_of_paths: tuple[str, ...] = ()
    required_value_matches: tuple[PathValueRequirement, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_intent(cls, intent: SemanticCapabilityIntent | None) -> EffectContract:
        if intent is None:
            return cls()
        return cls(
            required_paths=intent.required_output_fields,
            required_artifacts=intent.expected_artifacts,
            description=intent.capability_query,
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.required_paths,
                self.any_of_paths,
                self.required_value_matches,
                self.required_artifacts,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_paths": list(self.required_paths),
            "any_of_paths": list(self.any_of_paths),
            "required_value_matches": [
                item.to_dict() for item in self.required_value_matches
            ],
            "required_artifacts": list(self.required_artifacts),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ConditionCoverage:
    """Conditions a binding can produce and the verifier must still establish."""

    required_conditions: tuple[str, ...] = ()
    binding_covered_conditions: tuple[str, ...] = ()
    verification_required_conditions: tuple[str, ...] = ()
    uncovered_conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoundCapabilityCall:
    tool_name: str
    arguments: dict[str, Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    effects: EffectContract = field(default_factory=EffectContract)
    binding_reason: str = ""
    capability_version: str = ""
    result_adapter: str = "generic"
    execution_mode: str = "sync"
    lifecycle: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    timeout_policy: dict[str, Any] = field(default_factory=dict)
    idempotency_policy: dict[str, Any] = field(default_factory=dict)
    provenance_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "effects": self.effects.to_dict(),
            "binding_reason": self.binding_reason,
            "capability_version": self.capability_version,
            "result_adapter": self.result_adapter,
            "execution_mode": self.execution_mode,
            "lifecycle": self.lifecycle,
            "retry_policy": self.retry_policy,
            "timeout_policy": self.timeout_policy,
            "idempotency_policy": self.idempotency_policy,
            "provenance_policy": self.provenance_policy,
        }


@dataclass(slots=True)
class WorkflowCallTemplate:
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    effects: EffectContract = field(default_factory=EffectContract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "effects": self.effects.to_dict(),
        }


@dataclass(slots=True)
class BoundCapabilityWorkflow:
    workflow_id: str
    inputs: dict[str, Any]
    steps: list[WorkflowCallTemplate]
    effects: EffectContract = field(default_factory=EffectContract)
    max_steps: int = 5
    binding_reason: str = ""
    evidence_purpose: str = "claim_evidence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "inputs": self.inputs,
            "steps": [item.to_dict() for item in self.steps],
            "effects": self.effects.to_dict(),
            "max_steps": self.max_steps,
            "binding_reason": self.binding_reason,
            "evidence_purpose": self.evidence_purpose,
        }


@dataclass(slots=True)
class RouteDecision:
    backend: ExecutionBackend
    reason_code: str
    rationale: str
    query: str
    candidates: list[ResourceCandidate] = field(default_factory=list)
    previous_backend: str = ""
    requested_backend: str = ""
    admitted_capability: str = ""
    selected_capability: str = ""
    route_signature: str = ""
    execution_signature: str = ""
    execution_payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    request_id: str = ""
    retry_blocked: bool = False
    binding_id: str = ""
    catalog_revision: str = ""
    condition_coverage: ConditionCoverage = field(default_factory=ConditionCoverage)
    output_contract: dict[str, Any] = field(default_factory=dict)
    parameterization: dict[str, Any] = field(default_factory=dict)
    semantic_intent: SemanticCapabilityIntent | None = None
    bound_call: BoundCapabilityCall | None = None
    bound_workflow: BoundCapabilityWorkflow | None = None
    domain_workflow: dict[str, Any] = field(default_factory=dict)
    evidence_purpose: str = "claim_evidence"
    policy_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["backend"] = self.backend.value
        value["semantic_intent"] = (
            self.semantic_intent.to_dict() if self.semantic_intent else None
        )
        value["bound_call"] = self.bound_call.to_dict() if self.bound_call else None
        value["bound_workflow"] = (
            self.bound_workflow.to_dict() if self.bound_workflow else None
        )
        value["condition_coverage"] = self.condition_coverage.to_dict()
        return value
