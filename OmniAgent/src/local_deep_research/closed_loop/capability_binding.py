from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .capability_resolver import CapabilityCatalogResolver
from .contracts import A1TaskRequest
from .execution_models import (
    BoundCapabilityCall,
    ConditionCoverage,
    EffectContract,
    ExecutionBackend,
    ResourceCandidate,
    RouteDecision,
    SemanticCapabilityIntent,
)
from .execution_validation import schema_declares_path, validate_schema_instance
from .mcp_workflow import CapabilityWorkflowRegistry
from .scientific_workflow import (
    DomainWorkflowCompiler,
    DomainWorkflowRegistry,
    default_domain_workflow_registry,
    workflow_fingerprint,
)
from .routing_policy import RoutingPolicy


class CapabilityBindingRegistry:
    """Compile semantic intent into an auditable executable capability contract.

    A binding is the only place that couples an intent to a catalog capability,
    concrete arguments, post-execution effect checks, and the selected verifier.
    """

    verifier_id = "omniagent.structured_result.v1"

    def __init__(
        self,
        *,
        resolver: CapabilityCatalogResolver | None = None,
        workflow_registry: CapabilityWorkflowRegistry | None = None,
        domain_workflow_registry: DomainWorkflowRegistry | None = None,
        domain_workflow_compiler: DomainWorkflowCompiler | None = None,
        routing_policy: RoutingPolicy | None = None,
    ) -> None:
        self.resolver = resolver or CapabilityCatalogResolver()
        self.workflow_registry = workflow_registry or CapabilityWorkflowRegistry()
        self.domain_workflow_registry = (
            domain_workflow_registry or default_domain_workflow_registry()
        )
        self.domain_workflow_compiler = domain_workflow_compiler or DomainWorkflowCompiler()
        self.routing_policy = routing_policy or RoutingPolicy.from_environment()

    def bind(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
        *,
        candidates: list[ResourceCandidate],
        catalog_revision: str = "",
        previous_backend: str = "",
    ) -> RouteDecision:
        # Only stable, read-only direct profiles may use MCP. Other catalog
        # capabilities remain available to A1 without one-by-one adaptation.
        candidates = self.routing_policy.filter_candidates(candidates)
        base = {
            "query": intent.capability_query,
            "candidates": candidates,
            "previous_backend": previous_backend,
            "semantic_intent": intent,
            "catalog_revision": catalog_revision,
            "policy_metadata": self.routing_policy.describe(),
        }
        domain_spec = self.resolve_domain_workflow(request, intent)
        if domain_spec is not None:
            compiled = self.domain_workflow_compiler.compile(
                domain_spec,
                candidates=candidates,
                catalog_revision=catalog_revision,
                inputs=self._domain_workflow_inputs(request, intent),
            )

            domain_payload = compiled.to_dict()
            domain_payload["fingerprint"] = workflow_fingerprint(compiled)
            if not compiled.executable:
                reason = "; ".join(compiled.compile_errors[:4])
                return self._domain_a1_fallback(
                    compiled,
                    intent=intent,
                    request=request,
                    reason_code="domain_workflow_mcp_unavailable_fallback_to_a1",
                    rationale=reason,
                    base=base,
                )
            bound_mcp = compiled.to_bound_mcp_workflow()
            if bound_mcp is not None:
                errors = self.routing_policy.authorize_workflow(
                    bound_mcp, domain_workflow=True
                )
                if errors:
                    return self._domain_a1_fallback(
                        compiled,
                        intent=intent,
                        request=request,
                        reason_code="domain_workflow_policy_fallback_to_a1",
                        rationale="; ".join(errors[:4]),
                        base=base,
                    )
                return RouteDecision(
                    backend=ExecutionBackend.MCP,
                    reason_code="domain_workflow_bound",
                    rationale="Validated domain workflow compiled to the bounded MCP executor.",
                    admitted_capability=domain_spec.qualified_id,
                    binding_id=f"domain:{domain_spec.qualified_id}",
                    condition_coverage=ConditionCoverage(
                        required_conditions=self._required_conditions(intent, request),
                        binding_covered_conditions=tuple(
                            f"domain_node:{node.node_id}" for node in compiled.nodes
                        ),
                        verification_required_conditions=tuple(
                            f"evidence:{item.property}"
                            for item in domain_spec.evidence_requirements
                        ),
                    ),
                    output_contract={
                        **self._output_contract(
                            EffectContract(),
                            evidence_purpose="planning_evidence",
                            requested_effects=EffectContract.from_intent(intent),
                        ),
                        "domain_workflow": domain_payload,
                    },
                    bound_workflow=bound_mcp,
                    domain_workflow=domain_payload,
                    evidence_purpose="planning_evidence",
                    **base,
                )
            # A mixed workflow is an A1 job contract; A1 must execute only the
            # validated nodes after Harness executes the deterministic MCP stage.
            mcp_stage = compiled.to_bound_mcp_stage_workflow()
            if mcp_stage is not None:
                errors = self.routing_policy.authorize_workflow(
                    mcp_stage, domain_workflow=True
                )
                if errors:
                    return self._domain_a1_fallback(
                        compiled,
                        intent=intent,
                        request=request,
                        reason_code="domain_workflow_policy_fallback_to_a1",
                        rationale="; ".join(errors[:4]),
                        base=base,
                    )
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="domain_workflow_bound",
                rationale="Validated mixed MCP/A1 workflow compiled to a structured A1 contract.",
                admitted_capability=domain_spec.qualified_id,
                binding_id=f"domain:{domain_spec.qualified_id}",
                condition_coverage=ConditionCoverage(
                    required_conditions=self._required_conditions(intent, request),
                    binding_covered_conditions=tuple(
                        f"domain_node:{node.node_id}" for node in compiled.nodes
                    ),
                    verification_required_conditions=tuple(
                        f"evidence:{item.property}"
                        for item in domain_spec.evidence_requirements
                    ),
                ),
                output_contract={
                    **self._output_contract(
                        EffectContract.from_intent(intent),
                        evidence_purpose="claim_evidence",
                    ),
                    "domain_workflow": domain_payload,
                },
                bound_workflow=mcp_stage,
                domain_workflow=domain_payload,
                evidence_purpose="claim_evidence",
                **base,
            )
        if (
            intent.side_effect.value == "workspace_write"
            and intent.operation.value in {"retrieve", "validate"}
            and not self.resolver.requires_catalog(intent)
        ):
            required = self._required_conditions(intent, request)
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="workspace_write_fallback_to_a1",
                rationale=(
                    "A retrieval or validation intent cannot be routed solely because it "
                    "mentions a workspace write; the Harness owns materialization and "
                    "requires a separately bound capability; A1 owns the adaptive work."
                ),
                condition_coverage=ConditionCoverage(
                    required_conditions=required,
                    uncovered_conditions=required,
                ),
                output_contract=self._output_contract(EffectContract.from_intent(intent)),
                **base,
            )
        if not self.resolver.requires_catalog(intent):
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="semantic_adaptive_execution",
                rationale=(
                    "The intent is not a schema-bound retrieval or validation; "
                    "use A1 while the Harness owns final materialization."
                ),
                binding_id="adaptive.a1.v1",
                condition_coverage=self._adaptive_coverage(intent),
                output_contract=self._output_contract(EffectContract.from_intent(intent)),
                **base,
            )

        unavailable_reason = self.workflow_registry.unavailability_reason(
            request, intent, candidates
        )
        if unavailable_reason:
            required = self._required_conditions(intent, request)
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="semantic_condition_fallback_to_a1",
                rationale=unavailable_reason + "; A1 owns the adaptive workflow.",
                binding_id="adaptive.a1_fallback.v1",
                condition_coverage=ConditionCoverage(
                    required_conditions=required,
                    uncovered_conditions=required,
                ),
                output_contract=self._output_contract(EffectContract.from_intent(intent)),
                **base,
            )

        workflow = self.workflow_registry.bind(request, intent, candidates)
        if workflow is not None:
            policy_errors = self.routing_policy.authorize_workflow(
                workflow, domain_workflow=False
            )
            if policy_errors:
                return RouteDecision(
                    backend=ExecutionBackend.A1,
                    reason_code="generic_workflow_policy_fallback_to_a1",
                    rationale="; ".join(policy_errors[:4]),
                    binding_id="adaptive.a1_fallback.v1",
                    condition_coverage=self._adaptive_coverage(intent),
                    output_contract=self._output_contract(
                        EffectContract.from_intent(intent)
                    ),
                    **base,
                )
            coverage = ConditionCoverage(
                required_conditions=self._required_conditions(intent, request),
                binding_covered_conditions=tuple(
                    f"workflow_step:{step.step_id}" for step in workflow.steps
                ),
                verification_required_conditions=self._effect_conditions(workflow.effects),
            )
            return RouteDecision(
                backend=ExecutionBackend.MCP,
                reason_code="semantic_workflow_bound",
                rationale=workflow.binding_reason,
                admitted_capability=workflow.workflow_id,
                binding_id=f"workflow:{workflow.workflow_id}",
                condition_coverage=coverage,
                output_contract=self._output_contract(
                    EffectContract(),
                    evidence_purpose="planning_evidence",
                    requested_effects=workflow.effects,
                ),
                bound_workflow=workflow,
                evidence_purpose="planning_evidence",
                **base,
            )

        if self.workflow_registry.requires_adaptive_execution(request, intent):
            return RouteDecision(
                backend=ExecutionBackend.A1,
                reason_code="semantic_structural_analysis",
                rationale=(
                    "The step requires atom-level structural interpretation or geometry "
                    "rather than a catalog-proven structured retrieval. Route it to A1 and "
                    "verify its structured result at the Harness boundary."
                ),
                binding_id="adaptive.structural_analysis.v1",
                condition_coverage=self._adaptive_coverage(intent),
                output_contract=self._output_contract(EffectContract.from_intent(intent)),
                **base,
            )

        arguments = request.step.inputs.get("arguments")
        effects = EffectContract.from_intent(intent)
        derived_arguments = False
        if not isinstance(arguments, dict):
            executable = [
                item
                for item in candidates
                if self.routing_policy.derive_arguments(request, item, intent) is not None
            ]
            if executable:
                # Fill simple provider arguments from the semantic request;
                # do not invoke another LLM just to translate a known shape.
                arguments = self.routing_policy.derive_arguments(
                    request, executable[0], intent
                )
                derived_arguments = True
            else:
                if candidates:
                    required = self._required_conditions(intent, request)
                    return RouteDecision(
                        backend=ExecutionBackend.MCP,
                        reason_code="schema_parameters_required",
                        rationale=(
                            "The catalog exposed an allowlisted input schema, but the "
                            "request lacks a deterministic argument mapping; disclose the "
                            "schema and parameterize the call before binding."
                        ),
                        binding_id="catalog.schema_parameterization.v1",
                        condition_coverage=ConditionCoverage(
                            required_conditions=required,
                            uncovered_conditions=required,
                        ),
                        output_contract=self._output_contract(effects),
                        **base,
                    )
                return RouteDecision(
                    backend=ExecutionBackend.A1,
                    reason_code="semantic_capability_fallback_to_a1",
                    rationale=(
                        "No allowlisted direct MCP capability has a deterministic argument "
                        "mapping for this intent; A1 owns the adaptive execution."
                    ),
                    binding_id="adaptive.a1_fallback.v1",
                    condition_coverage=ConditionCoverage(
                        required_conditions=self._required_conditions(intent, request),
                        uncovered_conditions=self._required_conditions(intent, request),
                    ),
                    output_contract=self._output_contract(effects),
                    **base,
                )

        provisional = self.resolver.resolve(
            intent,
            candidates=candidates,
            previous_backend=previous_backend,
            arguments=arguments,
        )
        provisional.catalog_revision = catalog_revision
        if provisional.backend is not ExecutionBackend.MCP:
            provisional.backend = ExecutionBackend.A1
            provisional.reason_code = "catalog_fallback_to_a1"
            provisional.rationale = provisional.rationale + "; A1 owns adaptive execution."
            provisional.binding_id = "adaptive.a1_fallback.v1"
            provisional.condition_coverage = ConditionCoverage(
                required_conditions=self._required_conditions(intent, request),
                uncovered_conditions=self._required_conditions(intent, request),
            )
            provisional.output_contract = self._output_contract(
                EffectContract.from_intent(intent)
            )
            return provisional

        candidate = next(
            (
                item
                for item in candidates
                if item.qualified_name == provisional.admitted_capability
            ),
            None,
        )
        if candidate is None:
            provisional.backend = ExecutionBackend.A1
            provisional.reason_code = "semantic_capability_fallback_to_a1"
            provisional.rationale = "Catalog admission did not resolve a direct profile; A1 owns execution."
            provisional.binding_id = "adaptive.a1_fallback.v1"
            return provisional
        if not self._catalog_supports_effect_verification(candidate, EffectContract()):
            provisional.backend = ExecutionBackend.A1
            provisional.reason_code = "semantic_output_contract_fallback_to_a1"
            provisional.rationale = (
                "The selected capability does not expose a usable result schema; A1 owns execution."
            )
            provisional.binding_id = "adaptive.a1_fallback.v1"
            provisional.condition_coverage = ConditionCoverage(
                required_conditions=self._required_conditions(intent, request),
                uncovered_conditions=self._required_conditions(intent, request),
            )
            provisional.output_contract = self._output_contract(effects)
            return provisional
        errors = validate_schema_instance(
            arguments, candidate.input_schema, strict_objects=True
        )
        if errors:
            provisional.backend = ExecutionBackend.A1
            provisional.reason_code = "semantic_arguments_unbound"
            provisional.rationale = "; ".join(errors[:3]) + "; A1 owns adaptive execution."
            provisional.binding_id = "adaptive.a1_fallback.v1"
            return provisional

        # Keep the intent's effect contract on the bound call so untyped provider
        # output is still checked against the requested evidence at runtime.
        bound_effects = effects
        if derived_arguments:
            provisional.reason_code = "policy_parameterized"
            provisional.rationale = (
                "An allowlisted direct MCP capability was bound from protocol-neutral "
                "inputs without nested model routing."
            )
        provisional.binding_id = "catalog.single_capability.v1"
        provisional.selected_capability = candidate.qualified_name
        provisional.bound_call = BoundCapabilityCall(
            tool_name=candidate.qualified_name,
            arguments=dict(arguments),
            input_schema=candidate.input_schema,
            output_schema=candidate.output_schema,
            effects=bound_effects,
            binding_reason=provisional.reason_code,
            capability_version=candidate.capability_version,
            result_adapter=candidate.result_adapter,
            execution_mode=candidate.execution_mode,
            lifecycle=dict(candidate.lifecycle),
            retry_policy=dict(candidate.retry_policy),
            timeout_policy=dict(candidate.timeout_policy),
            idempotency_policy=dict(candidate.idempotency_policy),
            provenance_policy=dict(candidate.provenance_policy),
        )
        provisional.condition_coverage = ConditionCoverage(
            required_conditions=self._required_conditions(intent, request),
            binding_covered_conditions=(
                f"catalog_capability:{candidate.qualified_name}",
                "input_schema",
            ),
            verification_required_conditions=(),
        )
        provisional.evidence_purpose = "planning_evidence"
        provisional.output_contract = self._output_contract(
            bound_effects,
            candidate.output_schema,
            evidence_purpose="planning_evidence",
            requested_effects=effects,
        )
        return provisional

    def resolve_domain_workflow(
        self,
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> Any | None:
        """Apply declarative prior-evidence preconditions before capability discovery."""
        spec = self.domain_workflow_registry.resolve(intent=intent, request=request)
        if spec is None:
            return None
        contract = spec.verification_contract
        preconditions = contract.get("preconditions", {}) if isinstance(contract, dict) else {}
        prior_rule = (
            preconditions.get("prior_evidence", {})
            if isinstance(preconditions, dict)
            else {}
        )
        if not isinstance(prior_rule, dict) or not prior_rule:
            return spec
        backend = str(prior_rule.get("source_backend", "")).casefold().strip()
        capability = str(prior_rule.get("capability_contains", "")).casefold().strip()
        for record in request.prior_observations:
            encoded = json.dumps(record, ensure_ascii=False, default=str).casefold()
            if (not backend or backend in encoded) and (
                not capability or capability in encoded
            ):
                return spec
        fallback_id = str(prior_rule.get("fallback_workflow_id", "")).strip()
        return self.domain_workflow_registry.get(fallback_id) if fallback_id else spec

    @staticmethod
    def _domain_workflow_inputs(
        request: A1TaskRequest,
        intent: SemanticCapabilityIntent,
    ) -> dict[str, Any]:
        """Normalize planner semantics into reusable workflow parameters."""
        inputs = dict(request.step.inputs)
        context = dict(intent.entity_context)
        text = " ".join(
            str(value or "")
            for value in (
                request.research_goal,
                request.step.objective,
                intent.capability_query,
            )
        )
        pdb_match = re.search(r"\b[0-9][A-Za-z0-9]{3}\b", text)
        pdb_id = str(context.get("pdb_id") or (pdb_match.group(0) if pdb_match else ""))
        accession_match = re.search(
            r"\b[A-Z][0-9][A-Z0-9]{3,6}(?:-[0-9]+)?\b", text
        )
        accession = str(
            inputs.get("uniprot_accession")
            or context.get("uniprot_accession")
            or (accession_match.group(0) if accession_match else "")
        )
        feature_match = re.search(
            r"\b([A-Za-z0-9]+(?:[- ]chain))\b", text, flags=re.IGNORECASE
        )
        feature_name = str(
            inputs.get("feature_name")
            or context.get("feature_name")
            or (feature_match.group(1) if feature_match else "")
        ).replace(" chain", "-chain")
        cleavage_residues = inputs.get("cleavage_residues")
        if not isinstance(cleavage_residues, (list, tuple)):
            cleavage_residues = (
                ["K", "R"]
                if "trypsin" in text.casefold() or "tryptic" in text.casefold()
                else []
            )
        missed_cleavages = inputs.get("missed_cleavages")
        if not isinstance(missed_cleavages, int) or isinstance(missed_cleavages, bool):
            zero_missed = re.search(r"zero\s+missed\s+cleavages", text, re.IGNORECASE)
            missed_cleavages = 0 if zero_missed or "trypsin" in text.casefold() else 0
        output_match = re.search(
            r"key named [`']?([A-Za-z_][A-Za-z0-9_]*)[`']?", text, flags=re.IGNORECASE
        )
        required_fields = {item.casefold() for item in intent.required_output_fields}
        property_name = (
            "molecular_net_charge"
            if any("charge" in item for item in required_fields)
            else str(inputs.get("property_name") or "conditioned_property")
        )
        context_name = (
            "protein_bound"
            if "bound" in text.casefold()
            else str(inputs.get("context") or "molecular_state")
        )
        entity_ids = re.findall(
            r"\b(?:drug|disease|gene/protein|gene|protein|pathway|phenotype):[A-Za-z0-9_.-]+\b",
            text,
            flags=re.IGNORECASE,
        )
        head_entity = str(
            inputs.get("head_entity")
            or context.get("head_entity")
            or (entity_ids[0] if entity_ids else "")
        ).strip()
        tail_entity = str(
            inputs.get("tail_entity")
            or context.get("tail_entity")
            or (entity_ids[1] if len(entity_ids) > 1 else "")
        ).strip()

        raw_relations = inputs.get("candidate_relations", context.get("candidate_relations"))
        if isinstance(raw_relations, str):
            try:
                parsed_relations = ast.literal_eval(raw_relations)
            except (SyntaxError, ValueError):
                parsed_relations = None
            if isinstance(parsed_relations, (list, tuple)):
                candidate_relations = [
                    str(item).strip()
                    for item in parsed_relations
                    if str(item).strip()
                ]
            else:
                candidate_relations = [
                    item.strip(" []`\"'")
                    for item in re.split(r"[,|]", raw_relations)
                    if item.strip(" []`\"'")
                ]
        elif isinstance(raw_relations, (list, tuple)):
            candidate_relations = [str(item).strip() for item in raw_relations if str(item).strip()]
        else:
            relation_match = re.search(
                r"candidate relations?\s*(?::|=|are)\s*\[?([^\]\n.;]+)",
                text,
                flags=re.IGNORECASE,
            )
            candidate_relations = [
                item.strip(" `\"'")
                for item in re.split(r"[,|]", relation_match.group(1))
                if item.strip(" `\"'")
            ] if relation_match else []

        configured_kg_path = str(
            inputs.get("kg_path") or context.get("kg_path") or ""
        ).strip()
        allowed_roots = [
            Path(item).expanduser().resolve()
            for item in request.allowed_paths
            if str(item).strip()
        ]

        def admitted_path(value: str) -> str:
            if not value:
                return ""
            path = Path(value).expanduser().resolve()
            if any(path == root or root.is_dir() and path.is_relative_to(root) for root in allowed_roots):
                return str(path)
            return ""

        kg_path = admitted_path(configured_kg_path)
        if not kg_path:
            for root in allowed_roots:
                candidates = (
                    [root]
                    if root.is_file() and root.suffix.casefold() == ".csv"
                    else [root / "kg_observed.csv", root / "observed_kg.csv"]
                    if root.is_dir()
                    else []
                )
                existing = next((item for item in candidates if item.is_file()), None)
                if existing is not None:
                    kg_path = str(existing)
                    break

        inputs.pop("ground_truth_path", None)
        inputs.setdefault("pdb_id", pdb_id)
        inputs.setdefault("uniprot_accession", accession)
        inputs.setdefault(
            "uniprot_endpoint",
            f"https://rest.uniprot.org/uniprotkb/{accession}.json" if accession else "",
        )
        inputs.setdefault("feature_name", feature_name)
        inputs.setdefault("cleavage_residues", list(cleavage_residues))
        inputs.setdefault("missed_cleavages", missed_cleavages)
        inputs.setdefault("output_key", str(inputs.get("output_key") or (output_match.group(1) if output_match else "")))
        inputs.setdefault(
            "entity",
            context.get("ligand_descriptor") or context.get("pdb_id") or pdb_id,
        )
        inputs.setdefault("property_name", property_name)
        inputs.setdefault("context", context_name)
        inputs.setdefault("transformation", "protonate at the retrieved condition")
        inputs.setdefault("calculation", "sum atom formal charges")
        inputs["kg_path"] = kg_path
        inputs["head_entity"] = head_entity
        inputs["tail_entity"] = tail_entity
        inputs["candidate_relations"] = candidate_relations
        inputs.setdefault("max_hops", 2)
        inputs.setdefault("max_paths", 100)
        return inputs

    @staticmethod
    def _catalog_proves_effect(
        candidate: ResourceCandidate,
        effects: EffectContract,
    ) -> bool:
        if effects.is_empty:
            return True
        schema = candidate.output_schema
        if not isinstance(schema, dict) or str(schema.get("type", "")).lower() in {
            "",
            "any",
        }:
            return False
        for path in effects.required_paths:
            if not schema_declares_path(schema, path):
                return False
        if effects.required_artifacts and not schema_declares_path(schema, "artifacts"):
            return False
        return True

    @classmethod
    def _catalog_supports_effect_verification(
        cls,
        candidate: ResourceCandidate,
        effects: EffectContract,
    ) -> bool:
        if cls._catalog_proves_effect(candidate, effects):
            return True
        schema = candidate.output_schema
        schema_type = (
            str(schema.get("type", "")).lower() if isinstance(schema, dict) else ""
        )
        return schema_type in {"", "any"} and not effects.required_artifacts

    @staticmethod
    def _required_conditions(
        intent: SemanticCapabilityIntent,
        request: A1TaskRequest | None = None,
    ) -> tuple[str, ...]:
        workflow_conditions = (
            CapabilityWorkflowRegistry.required_conditions(request, intent)
            if request is not None
            else ()
        )
        return tuple(
            dict.fromkeys(
                [
                    *(f"field:{value}" for value in intent.required_output_fields),
                    *(f"artifact:{value}" for value in intent.expected_artifacts),
                    *workflow_conditions,
                ]
            )
        )

    @staticmethod
    def _effect_conditions(contract: EffectContract) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *(f"field:{value}" for value in contract.required_paths),
                    *(f"any_of:{value}" for value in contract.any_of_paths),
                    *(
                        "value:"
                        + item.path
                        + "="
                        + "|".join(item.expected_values)
                        for item in contract.required_value_matches
                    ),
                    *(f"artifact:{value}" for value in contract.required_artifacts),
                ]
            )
        )

    def _adaptive_coverage(self, intent: SemanticCapabilityIntent) -> ConditionCoverage:
        required = self._required_conditions(intent)
        return ConditionCoverage(
            required_conditions=required,
            verification_required_conditions=required,
        )

    def _domain_a1_fallback(
        self,
        compiled: Any,
        *,
        intent: SemanticCapabilityIntent,
        request: A1TaskRequest,
        reason_code: str,
        rationale: str,
        base: dict[str, Any],
    ) -> RouteDecision:
        payload = compiled.to_a1_fallback_dict(reason=rationale)
        return RouteDecision(
            backend=ExecutionBackend.A1,
            reason_code=reason_code,
            rationale=rationale + "; A1 owns the complete workflow.",
            admitted_capability=compiled.spec.qualified_id,
            binding_id="adaptive.a1_fallback.v1",
            condition_coverage=self._adaptive_coverage(intent),
            output_contract=self._output_contract(EffectContract.from_intent(intent)),
            domain_workflow=payload,
            **base,
        )

    def _output_contract(
        self,
        effects: EffectContract,
        output_schema: dict[str, Any] | None = None,
        *,
        evidence_purpose: str = "claim_evidence",
        requested_effects: EffectContract | None = None,
    ) -> dict[str, Any]:
        return {
            "output_schema": dict(output_schema or {}),
            "effects": effects.to_dict(),
            "verifier_id": self.verifier_id,
            "evidence_purpose": evidence_purpose,
            "requested_effects": (
                requested_effects.to_dict() if requested_effects is not None else None
            ),
        }
