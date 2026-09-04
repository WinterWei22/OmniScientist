from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable

from .execution_models import (
    BoundCapabilityCall,
    BoundCapabilityWorkflow,
    ExecutionShape,
    ResourceCandidate,
    SemanticCapabilityIntent,
    SemanticOperation,
    SideEffect,
)
from .execution_validation import validate_schema_instance


class RoutingMode(str, Enum):
    A1_FIRST = "a1_first"
    A1_ONLY = "a1_only"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class DirectMCPProfile:
    qualified_name: str
    allowed_argument_keys: frozenset[str]
    required_any_of: tuple[str, ...]
    forbidden_argument_keys: frozenset[str] = frozenset()
    fixed_arguments: tuple[tuple[str, Any], ...] = ()
    trusted_argument_schemas: tuple[tuple[str, dict[str, Any]], ...] = ()
    evidence_purpose: str = "planning_evidence"

    def argument_schema(self, key: str) -> dict[str, Any] | None:
        schema = dict(self.trusted_argument_schemas).get(key)
        return dict(schema) if isinstance(schema, dict) else None

    @classmethod
    def _contains_reference(cls, value: Any) -> bool:
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                return True
            return any(cls._contains_reference(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_reference(item) for item in value)
        return False

    def restrict_candidate(self, candidate: ResourceCandidate) -> ResourceCandidate | None:
        schema = candidate.input_schema
        if not isinstance(schema, dict) or str(schema.get("type", "")).casefold() != "object":
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None
        selected: dict[str, dict[str, Any]] = {}
        for key, value in properties.items():
            if (
                key not in self.allowed_argument_keys
                or key in self.forbidden_argument_keys
                or not isinstance(value, dict)
            ):
                continue
            provider_type = str(value.get("type", "")).casefold()
            if provider_type not in {"", "any"}:
                selected[key] = dict(value)
                continue
            trusted = self.argument_schema(key)
            if trusted is not None:
                selected[key] = trusted
        if not self.required_any_of or not any(key in selected for key in self.required_any_of):
            return None
        required = schema.get("required", [])
        restricted_schema = {
            "type": "object",
            "properties": selected,
            "required": [
                key
                for key in required
                if key in selected and key not in dict(self.fixed_arguments)
            ]
            if isinstance(required, list)
            else [],
            "additionalProperties": False,
        }
        return replace(candidate, input_schema=restricted_schema)

    def normalize_arguments(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        normalized = dict(arguments)
        errors: list[str] = []
        unexpected = sorted(set(normalized) - self.allowed_argument_keys)
        if unexpected:
            errors.append("arguments are outside the direct MCP profile: " + ", ".join(unexpected))
        forbidden = sorted(
            key
            for key in self.forbidden_argument_keys
            if key in normalized and normalized[key] not in (None, "", False)
        )
        if forbidden:
            errors.append("forbidden direct MCP arguments: " + ", ".join(forbidden))
        for key, value in self.fixed_arguments:
            if key in normalized and normalized[key] != value:
                errors.append(f"direct MCP argument {key!r} must equal {value!r}")
            normalized[key] = value
        for key, value in normalized.items():
            schema = self.argument_schema(key)
            if schema is None or self._contains_reference(value):
                continue
            schema_errors = validate_schema_instance(
                value,
                schema,
                strict_objects=True,
            )
            errors.extend(
                f"direct MCP argument {key!r} {error}" for error in schema_errors[:3]
            )
        if self.required_any_of and not any(
            key in normalized and normalized[key] not in (None, "", [], {})
            for key in self.required_any_of
        ):
            errors.append(
                "direct MCP call requires one of: " + ", ".join(self.required_any_of)
            )
        return normalized, errors


def _profile(
    qualified_name: str,
    *,
    allowed: Iterable[str],
    required_any_of: Iterable[str],
    forbidden: Iterable[str] = (),
    fixed: dict[str, Any] | None = None,
    schemas: dict[str, dict[str, Any]] | None = None,
) -> DirectMCPProfile:
    return DirectMCPProfile(
        qualified_name=qualified_name,
        allowed_argument_keys=frozenset(allowed),
        required_any_of=tuple(required_any_of),
        forbidden_argument_keys=frozenset(forbidden),
        fixed_arguments=tuple((fixed or {}).items()),
        trusted_argument_schemas=tuple((schemas or {}).items()),
    )


DEFAULT_DIRECT_MCP_PROFILES = (
    _profile(
        "biomni.tool.database.query_uniprot",
        allowed=("endpoint", "max_results"),
        required_any_of=("endpoint",),
        forbidden=("prompt",),
        schemas={
            "endpoint": {"type": "string"},
            "max_results": {"type": "integer"},
        },
    ),
    _profile(
        "biomni.tool.database.query_pdb",
        allowed=(
            "gene_symbol",
            "organism",
            "experimental_method",
            "released_before",
            "max_results",
        ),
        required_any_of=(
            "gene_symbol",
            "organism",
            "experimental_method",
            "released_before",
        ),
        forbidden=("prompt", "query"),
        schemas={
            "gene_symbol": {"type": "string"},
            "organism": {"type": "string"},
            "experimental_method": {"type": "string"},
            "released_before": {"type": "string"},
            "max_results": {"type": "integer"},
        },
    ),
    _profile(
        "biomni.tool.database.query_stringdb",
        allowed=("identifiers", "species", "required_score", "download_image"),
        required_any_of=("identifiers",),
        forbidden=("prompt", "endpoint", "output_dir"),
        fixed={"download_image": False},
        schemas={
            "identifiers": {"type": "string"},
            "species": {"type": "integer"},
            "required_score": {"type": "integer"},
            "download_image": {"type": "boolean"},
        },
    ),
    _profile(
        "biomni.tool.database.query_opentarget",
        allowed=("disease_name", "disease_id", "max_results", "verbose"),
        required_any_of=("disease_name", "disease_id"),
        forbidden=("prompt", "query", "variables"),
        fixed={"verbose": False},
        schemas={
            "disease_name": {"type": "string"},
            "disease_id": {"type": "string"},
            "max_results": {"type": "integer"},
            "verbose": {"type": "boolean"},
        },
    ),
    _profile(
        "biomni.tool.literature.query_pubmed",
        allowed=("query", "max_papers", "max_retries"),
        required_any_of=("query",),
        forbidden=("email",),
        schemas={
            "query": {"type": "string"},
            "max_papers": {"type": "integer"},
            "max_retries": {"type": "integer"},
        },
    ),
    _profile(
        "biomni.tool.database.query_alphafold",
        allowed=(
            "uniprot_id",
            "endpoint",
            "residue_range",
            "download",
            "model_version",
            "model_number",
        ),
        required_any_of=("uniprot_id",),
        forbidden=("output_dir", "file_format"),
        fixed={"download": False},
        schemas={
            "uniprot_id": {"type": "string"},
            "endpoint": {"type": "string"},
            "residue_range": {"type": "string"},
            "download": {"type": "boolean"},
            "model_version": {"type": "string"},
            "model_number": {"type": "integer"},
        },
    ),
    _profile(
        "biomni.tool.knowledge_graph.load_biomedical_kg",
        allowed=("kg_path", "format", "delimiter", "has_header", "schema", "use_cache"),
        required_any_of=("kg_path",),
        fixed={
            "format": "csv",
            "delimiter": ",",
            "has_header": True,
            "schema": "primekg",
            "use_cache": True,
        },
        schemas={
            "kg_path": {"type": "string"},
            "format": {"type": "string"},
            "delimiter": {"type": "string"},
            "has_header": {"type": "boolean"},
            "schema": {"type": "string"},
            "use_cache": {"type": "boolean"},
        },
    ),
    _profile(
        "biomni.tool.knowledge_graph.extract_enclosing_subgraph",
        allowed=(
            "kg_path",
            "head_entity",
            "tail_entity",
            "max_hops",
            "max_nodes_per_hop",
            "remove_direct_link",
            "bidirectional",
        ),
        required_any_of=("kg_path",),
        fixed={"remove_direct_link": True, "bidirectional": True},
        schemas={
            "kg_path": {"type": "string"},
            "head_entity": {"type": "string"},
            "tail_entity": {"type": "string"},
            "max_hops": {"type": "integer"},
            "max_nodes_per_hop": {"type": "integer"},
            "remove_direct_link": {"type": "boolean"},
            "bidirectional": {"type": "boolean"},
        },
    ),
    _profile(
        "biomni.tool.knowledge_graph.extract_metapaths",
        allowed=(
            "kg_path",
            "head_entity",
            "tail_entity",
            "max_length",
            "max_paths",
            "bidirectional",
        ),
        required_any_of=("kg_path",),
        fixed={"bidirectional": True},
        schemas={
            "kg_path": {"type": "string"},
            "head_entity": {"type": "string"},
            "tail_entity": {"type": "string"},
            "max_length": {"type": "integer"},
            "max_paths": {"type": "integer"},
            "bidirectional": {"type": "boolean"},
        },
    ),
)


class RoutingPolicy:
    """Own the deterministic boundary between planning retrieval and A1 execution."""

    version = "omniagent.routing.a1-only.v1"

    def __init__(
        self,
        *,
        mode: RoutingMode | str = RoutingMode.A1_ONLY,
        profiles: Iterable[DirectMCPProfile] | None = None,
        enabled_tools: Iterable[str] | None = None,
    ) -> None:
        self.mode = RoutingMode(str(getattr(mode, "value", mode)).strip().casefold())
        configured = tuple(profiles or DEFAULT_DIRECT_MCP_PROFILES)
        enabled = {
            str(value).strip() for value in (enabled_tools or ()) if str(value).strip()
        }
        if enabled:
            configured = tuple(item for item in configured if item.qualified_name in enabled)
        self._profiles = {item.qualified_name: item for item in configured}

    @classmethod
    def from_environment(cls) -> RoutingPolicy:
        mode = os.getenv("OMNIAGENT_ROUTING_POLICY", RoutingMode.A1_ONLY.value)
        enabled = os.getenv("OMNIAGENT_DIRECT_MCP_TOOLS", "")
        return cls(
            mode=mode,
            enabled_tools=(item.strip() for item in enabled.split(",") if item.strip()),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode.value,
            "direct_mcp_tools": sorted(self._profiles),
            "default_mcp_evidence_purpose": "planning_evidence",
        }

    def profile_for(self, qualified_name: str) -> DirectMCPProfile | None:
        return self._profiles.get(str(qualified_name or "").strip())

    def should_inspect(
        self,
        intent: SemanticCapabilityIntent | None,
        *,
        workflow_spec: Any | None = None,
    ) -> bool:
        if self.mode is RoutingMode.A1_ONLY:
            return False
        if workflow_spec is not None:
            mcp_nodes = [
                node
                for node in getattr(workflow_spec, "nodes", ())
                if str(getattr(node, "executor", "")) == "mcp"
            ]
            if not mcp_nodes:
                return False
            return all(
                not str(getattr(node, "capability_id", "") or "").strip()
                or self.profile_for(str(getattr(node, "capability_id", ""))) is not None
                for node in mcp_nodes
            )
        return bool(
            intent is not None
            and intent.operation in {SemanticOperation.RETRIEVE, SemanticOperation.VALIDATE}
            and intent.execution_shape is ExecutionShape.SINGLE_CAPABILITY
            and intent.schema_bound
            and intent.side_effect is SideEffect.READ_ONLY
        )

    def filter_candidates(
        self, candidates: Iterable[ResourceCandidate]
    ) -> list[ResourceCandidate]:
        if self.mode is RoutingMode.A1_ONLY:
            return []
        admitted: list[ResourceCandidate] = []
        for candidate in candidates:
            profile = self.profile_for(candidate.qualified_name)
            restricted = profile.restrict_candidate(candidate) if profile else None
            if restricted is not None:
                admitted.append(restricted)
        return admitted

    def authorize_call(self, call: BoundCapabilityCall) -> list[str]:
        profile = self.profile_for(call.tool_name)
        if profile is None:
            return [f"capability {call.tool_name!r} is not in the direct MCP profile set"]
        arguments, errors = profile.normalize_arguments(call.arguments)
        if not errors:
            call.arguments = arguments
        return errors

    def authorize_workflow(
        self,
        workflow: BoundCapabilityWorkflow,
        *,
        domain_workflow: bool,
    ) -> list[str]:
        if not workflow.steps:
            return ["direct MCP workflow has no executable steps"]
        if not domain_workflow and len(workflow.steps) != 1:
            return ["generic direct MCP execution must contain exactly one step"]
        errors: list[str] = []
        for step in workflow.steps:
            profile = self.profile_for(step.tool_name)
            if profile is None:
                errors.append(f"workflow capability {step.tool_name!r} is not allowlisted")
                continue
            arguments, step_errors = profile.normalize_arguments(step.arguments)
            if not step_errors:
                step.arguments = arguments
            errors.extend(f"{step.step_id}: {error}" for error in step_errors)
        return errors

    def derive_arguments(
        self,
        request: Any,
        candidate: ResourceCandidate,
        intent: SemanticCapabilityIntent | None = None,
    ) -> dict[str, Any] | None:
        """Compile common protocol-neutral step fields into one safe MCP call.

        This deliberately handles only argument shapes whose meaning is stable
        across runs. An unrecognised shape stays with A1 instead of asking a
        nested model to guess a provider-specific request body.
        """
        profile = self.profile_for(candidate.qualified_name)
        if profile is None:
            return None
        step_inputs = getattr(getattr(request, "step", None), "inputs", {})
        inputs = dict(step_inputs) if isinstance(step_inputs, dict) else {}
        context = dict(intent.entity_context) if intent is not None else {}
        context.update(
            {
                str(key): value
                for key, value in inputs.items()
                if key not in {"arguments", "semantic_intent"}
            }
        )
        explicit = inputs.get("arguments")
        if isinstance(explicit, dict):
            arguments = dict(explicit)
        else:
            name = candidate.qualified_name
            arguments: dict[str, Any] = {}
            if name.endswith(".query_uniprot"):
                endpoint = (
                    context.get("endpoint")
                    or context.get("uniprot_endpoint")
                    or context.get("accession")
                    or context.get("uniprot_accession")
                )
                if endpoint:
                    endpoint = str(endpoint).strip()
                    if not endpoint.startswith(("http://", "https://", "/")):
                        endpoint = f"https://rest.uniprot.org/uniprotkb/{endpoint}.json"
                    arguments = {"endpoint": endpoint}
            elif name.endswith(".query_pdb"):
                aliases = {
                    "gene_symbol": ("gene_symbol", "gene", "target_gene"),
                    "organism": ("organism", "species"),
                    "experimental_method": ("experimental_method", "method"),
                    "released_before": (
                        "released_before",
                        "date_cutoff",
                        "release_date_before",
                    ),
                }
                arguments = {
                    key: context[source]
                    for key, sources in aliases.items()
                    for source in sources
                    if context.get(source) not in (None, "")
                }
                if "max_results" in context and context["max_results"] not in (None, ""):
                    arguments["max_results"] = context["max_results"]
                elif "limit" in context and context["limit"] not in (None, ""):
                    arguments["max_results"] = context["limit"]
            elif name.endswith(".query_stringdb"):
                identifiers = (
                    context.get("identifiers")
                    or context.get("gene_symbol")
                    or context.get("gene")
                    or context.get("target_gene")
                )
                if identifiers:
                    arguments = {"identifiers": identifiers}
                    if context.get("species") not in (None, ""):
                        arguments["species"] = context["species"]
                    if context.get("required_score") not in (None, ""):
                        arguments["required_score"] = context["required_score"]
                    arguments["download_image"] = False
            elif name.endswith(".query_opentarget"):
                disease_id = context.get("disease_id")
                disease_name = context.get("disease_name") or context.get("disease")
                if disease_id or disease_name:
                    arguments = {
                        ("disease_id" if disease_id else "disease_name"):
                        (disease_id or disease_name),
                        "verbose": False,
                    }
                    if context.get("max_results") not in (None, ""):
                        arguments["max_results"] = context["max_results"]
                    elif context.get("limit") not in (None, ""):
                        arguments["max_results"] = context["limit"]
            elif name.endswith(".query_pubmed"):
                query = context.get("query") or context.get("literature_query")
                if query:
                    arguments = {"query": query}
                    if context.get("max_papers") not in (None, ""):
                        arguments["max_papers"] = context["max_papers"]
            elif name.endswith(".query_alphafold"):
                accession = (
                    context.get("uniprot_id")
                    or context.get("uniprot_accession")
                    or context.get("accession")
                )
                if accession:
                    arguments = {"uniprot_id": accession, "download": False}
                    if context.get("endpoint") in {"prediction", "summary", "annotations"}:
                        arguments["endpoint"] = context["endpoint"]
            else:
                return None
        normalized, errors = profile.normalize_arguments(arguments)
        if errors:
            return None
        return normalized
