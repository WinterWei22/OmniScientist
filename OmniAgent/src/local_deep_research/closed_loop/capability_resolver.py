from __future__ import annotations

import re
from typing import Any

from .execution_models import (
    ExecutionBackend,
    ExecutionShape,
    BoundCapabilityWorkflow,
    ResourceCandidate,
    RouteDecision,
    SemanticCapabilityIntent,
    SemanticOperation,
    SideEffect,
)
from .execution_validation import validate_schema_instance


class CapabilityCatalogResolver:
    """Resolve a model-authored semantic intent against executable capabilities."""

    _MCP_OPERATIONS = {
        SemanticOperation.RETRIEVE,
        SemanticOperation.VALIDATE,
    }

    @staticmethod
    def requires_catalog(intent: SemanticCapabilityIntent) -> bool:
        return (
            intent.execution_shape
            in {ExecutionShape.SINGLE_CAPABILITY, ExecutionShape.MULTI_CAPABILITY}
            and intent.operation
            in {SemanticOperation.RETRIEVE, SemanticOperation.VALIDATE}
            and intent.schema_bound
        )

    def resolve(
        self,
        intent: SemanticCapabilityIntent,
        *,
        candidates: list[ResourceCandidate],
        previous_backend: str = "",
        arguments: dict[str, Any] | None = None,
        workflow: BoundCapabilityWorkflow | None = None,
    ) -> RouteDecision:
        base = {
            "query": intent.capability_query,
            "candidates": candidates,
            "previous_backend": previous_backend,
            "semantic_intent": intent,
        }
        bounded_lookup = (
            intent.execution_shape
            in {ExecutionShape.SINGLE_CAPABILITY, ExecutionShape.MULTI_CAPABILITY}
            and intent.operation in self._MCP_OPERATIONS
            and intent.schema_bound
        )
        if bounded_lookup:
            if workflow is not None:
                return RouteDecision(
                    backend=ExecutionBackend.MCP,
                    reason_code="semantic_workflow_bound",
                    rationale=workflow.binding_reason,
                    admitted_capability=workflow.workflow_id,
                    bound_workflow=workflow,
                    **base,
                )
            executable = [item for item in candidates if item.input_schema]
            if not executable:
                return RouteDecision(
                    backend=ExecutionBackend.UNAVAILABLE,
                    reason_code="semantic_capability_unavailable",
                    rationale="No retrieved capability exposed a schema for this semantic intent.",
                    **base,
                )
            exact = self._exact_candidate(intent.capability_query, executable)
            selected = exact or self._semantic_candidate(
                intent,
                executable,
                arguments=arguments,
            )
            if selected is None:
                return RouteDecision(
                    backend=ExecutionBackend.UNAVAILABLE,
                    reason_code="semantic_capability_unavailable",
                    rationale=(
                        "Retrieved candidates were not proven equivalent to the requested "
                        "capability and effect contract."
                    ),
                    **base,
                )
            argument_errors = validate_schema_instance(
                arguments or {}, selected.input_schema, strict_objects=True
            )
            if argument_errors:
                return RouteDecision(
                    backend=ExecutionBackend.UNAVAILABLE,
                    reason_code="semantic_arguments_unbound",
                    rationale="; ".join(argument_errors[:3]),
                    admitted_capability=selected.qualified_name,
                    **base,
                )
            return RouteDecision(
                backend=ExecutionBackend.MCP,
                reason_code=(
                    "semantic_exact_capability"
                    if exact is not None
                    else "semantic_catalog_match"
                ),
                rationale=(
                    "The capability name, schema, arguments, and requested effects are "
                    "compatible; the Harness owns workspace materialization."
                ),
                admitted_capability=selected.qualified_name,
                **base,
            )
        if intent.side_effect is SideEffect.WORKSPACE_WRITE:
            if intent.operation in {
                SemanticOperation.ANALYZE,
                SemanticOperation.EXPERIMENT,
                SemanticOperation.GENERATE_ARTIFACT,
                SemanticOperation.SYNTHESIZE,
            }:
                return RouteDecision(
                    backend=ExecutionBackend.A1,
                    reason_code="semantic_adaptive_workflow",
                    rationale=(
                        "A1 may compute an intermediate structured artifact, but it must "
                        "return a verifiable result or artifact manifest; the Harness owns "
                        "all final workspace materialization."
                    ),
                    **base,
                )
            return RouteDecision(
                backend=ExecutionBackend.UNAVAILABLE,
                reason_code="workspace_write_requires_harness_materialization",
                rationale=(
                    "Workspace writes do not select an execution backend. Bind a read-only "
                    "capability for retrieval/validation or let the Harness materialize the "
                    "structured result."
                ),
                **base,
            )
        if intent.execution_shape is not ExecutionShape.SINGLE_CAPABILITY:
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="semantic_adaptive_workflow",
                rationale=(
                    "The semantic intent requires adaptive or multi-capability execution; "
                    "the Harness owns any workspace materialization."
                ),
                **base,
            )
        return RouteDecision(
            backend=ExecutionBackend.A1,
            reason_code="semantic_open_execution",
            rationale="The semantic intent is not a schema-bound retrieval or validation.",
            **base,
        )

    @staticmethod
    def _exact_candidate(
        query: str,
        candidates: list[ResourceCandidate],
    ) -> ResourceCandidate | None:
        requested = query.strip().casefold()
        if not requested or any(character.isspace() for character in requested):
            return None
        return next(
            (
                item
                for item in candidates
                if requested
                in {
                    item.qualified_name.casefold(),
                    item.qualified_name.rsplit(".", 1)[-1].casefold(),
                }
            ),
            None,
        )

    @classmethod
    def _semantic_candidate(
        cls,
        intent: SemanticCapabilityIntent,
        candidates: list[ResourceCandidate],
        *,
        arguments: dict[str, Any] | None,
    ) -> ResourceCandidate | None:
        if arguments is None:
            return None
        compatible = []
        query_tokens = cls._distinctive_tokens(intent.capability_query)
        for item in candidates:
            if validate_schema_instance(
                arguments, item.input_schema, strict_objects=True
            ):
                continue
            candidate_text = f"{item.qualified_name} {item.description}"
            candidate_tokens = cls._distinctive_tokens(candidate_text)
            overlap = query_tokens.intersection(candidate_tokens)
            if not query_tokens or len(overlap) / len(query_tokens) < 0.5:
                continue
            # Required evidence is checked after task-specific reduction, not here.
            compatible.append(item)
        if not compatible:
            return None
        return max(
            compatible,
            key=lambda item: item.score if item.score is not None else float("-inf"),
        )

    @staticmethod
    def _distinctive_tokens(value: str) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "by",
            "data",
            "fetch",
            "file",
            "for",
            "from",
            "get",
            "given",
            "lookup",
            "official",
            "query",
            "record",
            "retrieve",
            "single",
            "structured",
            "the",
            "to",
            "tool",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))
            if len(token) > 1 and token not in stop_words
        }
