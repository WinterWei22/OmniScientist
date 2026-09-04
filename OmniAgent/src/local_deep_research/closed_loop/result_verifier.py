from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from .answer_validation import (
    assess_result_semantics,
    infer_answer_semantic_contract,
)
from .result_payload import compact_evidence_payload, material_result_leaves
from .scientific_workflow import normalize_domain_result, verify_domain_evidence
from .scientific_state import (
    AttemptRecord,
    AttemptStatus,
    ClaimRecord,
    ClaimStatus,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    ScientificState,
    VerifiedStateTransition,
)
from .task_metadata import external_task_summary

if TYPE_CHECKING:
    from .contracts import A1TaskRequest, A1TaskResult


class TaskResultVerifier(Protocol):
    verifier_id: str

    def verify(
        self,
        state: ScientificState,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> VerifiedStateTransition: ...


class StructuredResultVerifier:
    verifier_id = "omniagent.structured_result.v1"
    evidence_type = "structured_execution_result"

    def verify(
        self,
        state: ScientificState,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> VerifiedStateTransition:
        route = self._route_metadata(result)
        backend = str(route.get("backend", "unknown"))
        capability_id = self._trace_capability(result.tool_trace)
        if not capability_id:
            capability_id = str(
                route.get("selected_capability")
                or route.get("admitted_capability")
                or ""
            )
        attempt_id = self._stable_id(
            "attempt",
            request.run_id,
            str(request.iteration),
            request.step.step_id,
            str(state.state_version),
        )
        domain_workflow = route.get("domain_workflow")
        if isinstance(domain_workflow, dict):
            normalize_domain_result(domain_workflow, result)
        accepted, reason = self._admit(request, result)
        evidence: tuple[EvidenceRecord, ...] = ()
        claims: tuple[ClaimRecord, ...] = ()
        hypotheses: tuple[HypothesisRecord, ...] = ()
        if accepted:
            evidence_id = self._stable_id("evidence", attempt_id)
            semantic_validation = self._semantic_validation(request, result)
            evidence = (
                EvidenceRecord(
                    evidence_id=evidence_id,
                    evidence_type=self.evidence_type,
                    summary=self._summary(result),
                    source_attempt_id=attempt_id,
                    source_backend=backend,
                    source_capability_id=capability_id,
                    provenance=self._provenance(
                        result,
                        route,
                        semantic_validation=semantic_validation,
                    ),
                    payload=self._payload(result),
                    artifact_refs=tuple(dict.fromkeys(result.artifacts)),
                    verifier_id=self.verifier_id,
                ),
            )
            # Retrieval/workflow-prefix results are evidence for the next
            # executor. They are not conclusions until the complete scientific
            # workflow has passed its final verifier.
            claims = (
                ()
                if str(route.get("evidence_purpose", "claim_evidence"))
                == "planning_evidence"
                else self._proposed_claims(
                    state,
                    result,
                    attempt_id,
                    evidence_id,
                )
            )
            hypotheses = self._working_hypothesis(state, request)

        attempt = AttemptRecord(
            attempt_id=attempt_id,
            iteration=request.iteration,
            step_id=request.step.step_id,
            objective=request.step.objective,
            backend=backend,
            capability_id=capability_id,
            status=(
                AttemptStatus.SUCCEEDED
                if accepted
                else AttemptStatus.FAILED
                if not result.success
                else AttemptStatus.REJECTED
            ),
            result_status=result.result_status,
            verifier_id=self.verifier_id,
            reason=reason,
            evidence_ids=tuple(item.evidence_id for item in evidence),
            artifact_refs=tuple(dict.fromkeys(result.artifacts)),
        )
        transition_id = self._stable_id("transition", attempt_id)
        return VerifiedStateTransition(
            transition_id=transition_id,
            expected_state_version=state.state_version,
            accepted=accepted,
            reason=reason,
            verifier_id=self.verifier_id,
            attempt=attempt,
            evidence=evidence,
            claims=claims,
            hypotheses=hypotheses,
        )

    def _admit(
        self,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> tuple[bool, str]:
        route = self._route_metadata(result)
        if not result.success:
            return False, "; ".join(result.errors) or "execution reported failure"
        if result.result_status not in {"success", "completed"}:
            return False, f"execution returned non-success status: {result.result_status}"
        has_structured_result = bool(
            result.observations or result.metrics or result.output is not None
        )
        artifact_manifest = (
            result.task_metadata.get("artifact_manifest", [])
            if isinstance(result.task_metadata, dict)
            else []
        )
        has_verified_artifacts = bool(result.artifacts) and isinstance(
            artifact_manifest, list
        ) and len(artifact_manifest) == len(result.artifacts)
        if not has_structured_result and not has_verified_artifacts:
            return False, (
                "OUTPUT_CONTRACT_FAILED: execution returned neither structured output "
                "nor verified artifacts"
            )
        domain_workflow = route.get("domain_workflow")
        if isinstance(domain_workflow, dict):
            domain_errors = verify_domain_evidence(domain_workflow, result, request)
            if domain_errors:
                return False, "DOMAIN_EVIDENCE_CONTRACT_FAILED: " + "; ".join(domain_errors[:3])
        entity_errors = self._entity_identity_errors(request, result)
        if entity_errors:
            return False, "ENTITY_IDENTITY_CONTRACT_FAILED: " + "; ".join(
                entity_errors[:3]
            )
        return True, "structured execution result admitted"

    @staticmethod
    def _entity_identity_errors(
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> list[str]:
        if not request.entity_corrections:
            return []
        material = json.dumps(
            {
                "answer": result.answer,
                "output": result.output,
                "observations": result.observations,
            },
            ensure_ascii=False,
            default=str,
        )
        errors: list[str] = []
        for rejected, expected in request.entity_corrections.items():
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(rejected)}(?![A-Za-z0-9])",
                flags=re.IGNORECASE,
            )
            if pattern.search(material):
                errors.append(
                    f"result contains rejected entity identifier {rejected!r}; "
                    f"expected {expected!r}"
                )
        return errors

    @staticmethod
    def _route_metadata(result: A1TaskResult) -> dict[str, Any]:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        route = metadata.get("omniagent_route", {})
        return dict(route) if isinstance(route, dict) else {}

    @staticmethod
    def _trace_capability(trace: list[dict[str, Any]]) -> str:
        for item in reversed(trace):
            if not isinstance(item, dict):
                continue
            value = item.get("tool_name") or item.get("gateway_tool")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _summary(result: A1TaskResult) -> str:
        if result.observations:
            encoded = json.dumps(
                result.observations[:3], ensure_ascii=False, default=str
            )
            return encoded[:1600]
        if result.output is not None:
            return json.dumps(result.output, ensure_ascii=False, default=str)[:1600]
        if result.answer.strip():
            return result.answer.strip()[:1600]
        return "Execution produced structured metrics or artifacts."

    @staticmethod
    def _payload(result: A1TaskResult) -> dict[str, Any]:
        output = result.output
        if output is None:
            output = (
                result.verification_payload
                if result.verification_payload is not None
                else result.raw
            )
        value = {
            "answer": result.answer,
            "observations": result.observations[:12],
            "metrics": dict(result.metrics),
            "output": output,
            "result_status": result.result_status,
        }
        return compact_evidence_payload(value)

    @staticmethod
    def _provenance(
        result: A1TaskResult,
        route: dict[str, Any],
        *,
        semantic_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_metadata = external_task_summary(result.task_metadata)
        compact_route = task_metadata.pop("omniagent_route", None)
        if not isinstance(compact_route, dict):
            compact_route = external_task_summary(
                {"omniagent_route": route}
            ).get("omniagent_route", {})
        provenance = {
            "route": compact_route,
            "biomni_task": task_metadata,
            "effect_verification": task_metadata.get("effect_verification"),
            "trace_events": [
                {
                    key: item.get(key)
                    for key in ("event", "backend", "gateway_tool", "tool_name", "ok")
                    if key in item
                }
                for item in result.tool_trace[-12:]
                if isinstance(item, dict)
            ],
        }
        if semantic_validation is not None:
            provenance["semantic_validation"] = semantic_validation
        return provenance

    def _semantic_validation(
        self,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> dict[str, Any] | None:
        return None

    def _proposed_claims(
        self,
        state: ScientificState,
        result: A1TaskResult,
        attempt_id: str,
        evidence_id: str,
    ) -> tuple[ClaimRecord, ...]:
        statements: list[str] = []
        for observation in result.observations[:12]:
            if not isinstance(observation, dict):
                continue
            statement = str(
                observation.get("claim") or observation.get("statement") or ""
            ).strip()
            if statement and statement not in statements:
                statements.append(statement)

        if isinstance(result.output, dict):
            answer = str(result.output.get("answer", "")).strip()
            if answer and answer not in statements:
                statements.append(answer)
            rows = result.output.get("metabolism_table")
            if isinstance(rows, list):
                for row in rows[:12]:
                    if not isinstance(row, dict):
                        continue
                    compound = str(row.get("compound", "")).strip()
                    percentage = row.get("cyp3a4_metabolism_percentage")
                    if compound and isinstance(percentage, int | float) and not isinstance(
                        percentage, bool
                    ):
                        output_statement = (
                            f"{compound} has a reported CYP3A4 metabolism percentage "
                            f"of {float(percentage):g}%."
                        )
                        if output_statement not in statements:
                            statements.append(output_statement)
            for path, value in material_result_leaves(result.output):
                if path == "answer" or path.startswith("metabolism_table"):
                    continue
                output_statement = f"Final output {path}: {value}"
                if output_statement not in statements:
                    statements.append(output_statement)
        referenced_evidence = self._referenced_evidence_ids(state, result)
        claim_evidence_ids = tuple(
            dict.fromkeys((evidence_id, *referenced_evidence))
        )
        claims = []
        for index, statement in enumerate(statements[:12]):
            claims.append(
                ClaimRecord(
                    claim_id=self._stable_id(
                        "claim", attempt_id, str(index), statement
                    ),
                    statement=statement,
                    evidence_ids=claim_evidence_ids,
                    status=ClaimStatus.PROPOSED,
                    source_attempt_id=attempt_id,
                    verifier_id=self.verifier_id,
                )
            )
        return tuple(claims)

    @staticmethod
    def _referenced_evidence_ids(
        state: ScientificState,
        result: A1TaskResult,
    ) -> tuple[str, ...]:
        if not isinstance(result.output, dict):
            return ()
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        values: list[Any] = []
        if result.output.get("evidence_id") is not None:
            values.append(result.output.get("evidence_id"))
        evidence_ids = result.output.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            values.append(evidence_ids)
        elif isinstance(evidence_ids, list):
            values.extend(evidence_ids)
        if metadata.get("harness_finalization") is True:
            supporting = metadata.get("supporting_evidence_ids", [])
            if isinstance(supporting, str):
                values.append(supporting)
            elif isinstance(supporting, list):
                values.extend(supporting)
        return tuple(
            dict.fromkeys(
                str(value)
                for value in values
                if str(value) in state.evidence
            )
        )

    def _working_hypothesis(
        self,
        state: ScientificState,
        request: A1TaskRequest,
    ) -> tuple[HypothesisRecord, ...]:
        statement = request.hypothesis.strip()
        if not statement:
            return ()
        hypothesis_id = self._stable_id("hypothesis", statement.casefold())
        if hypothesis_id in state.hypotheses:
            return ()
        return (
            HypothesisRecord(
                hypothesis_id=hypothesis_id,
                statement=statement,
                status=HypothesisStatus.PROPOSED,
                uncertainty="Planner-authored working assumption; not a verified fact.",
                version=1,
            ),
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        payload = "\x1f".join(parts).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


class DrugDiscoveryResultVerifier(StructuredResultVerifier):
    verifier_id = "omniagent.drug_discovery_result.v1"
    evidence_type = "drug_discovery_execution_result"

    def _semantic_validation(
        self,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> dict[str, Any] | None:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        if metadata.get("harness_finalization") is True:
            return None
        return assess_result_semantics(
            infer_answer_semantic_contract(request.research_goal),
            result,
        )

    def _admit(
        self,
        request: A1TaskRequest,
        result: A1TaskResult,
    ) -> tuple[bool, str]:
        accepted, reason = super()._admit(request, result)
        if not accepted:
            return accepted, reason
        route = self._route_metadata(result)
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        if route.get("backend") == "mcp":
            bound_call = route.get("bound_call")
            bound_workflow = route.get("bound_workflow")
            if not isinstance(bound_call, dict) and not isinstance(bound_workflow, dict):
                return False, "MCP result lacks a validated execution binding"
            actual_tools = {
                str(item.get("tool_name"))
                for item in result.tool_trace
                if isinstance(item, dict) and item.get("tool_name")
            }
            if isinstance(bound_call, dict):
                expected_tool = str(bound_call.get("tool_name", ""))
                if not expected_tool or expected_tool not in actual_tools:
                    return False, "actual MCP tool does not match the bound capability"
                actual_binding = metadata.get("bound_call")
                if isinstance(actual_binding, dict) and (
                    actual_binding.get("tool_name") != bound_call.get("tool_name")
                    or actual_binding.get("arguments") != bound_call.get("arguments")
                ):
                    return False, "executed MCP call differs from the validated binding"
                effects = bound_call.get("effects", {})
            else:
                expected_tools = {
                    str(item.get("tool_name"))
                    for item in bound_workflow.get("steps", [])
                    if isinstance(item, dict) and item.get("tool_name")
                }
                if not expected_tools or not expected_tools.issubset(actual_tools):
                    return False, "actual MCP workflow does not match its bound capabilities"
                effects = bound_workflow.get("effects", {})
            effect_required = bool(
                isinstance(effects, dict)
                and any(
                    effects.get(key)
                    for key in ("required_paths", "any_of_paths", "required_artifacts")
                )
            )
            effect_verification = metadata.get("effect_verification")
            if effect_required and not (
                isinstance(effect_verification, dict)
                and effect_verification.get("passed") is True
            ):
                return False, "MCP result did not satisfy the planned effect contract"
            output_errors = metadata.get("output_schema_errors", [])
            if output_errors:
                return False, "MCP result failed its declared output schema"
        has_provenance = bool(route or result.task_metadata or result.tool_trace)
        has_structure = bool(
            result.observations or result.metrics or result.output is not None
        )
        if not has_provenance:
            return False, "drug-discovery result lacks execution provenance"
        if not has_structure:
            return False, "drug-discovery result lacks scientific content"
        semantic_contract = infer_answer_semantic_contract(request.research_goal)
        if semantic_contract is not None and route.get("backend") == "a1":
            semantic = self._semantic_validation(request, result)
            if isinstance(semantic, dict) and semantic.get("passed") is True:
                if not self._has_method_provenance(result, semantic):
                    return False, (
                        "A1 method-sensitive evidence lacks a method provenance record "
                        "linked to an executed tool or verified artifact"
                    )
        return True, "drug-discovery execution contract and effects admitted"

    @staticmethod
    def _has_method_provenance(
        result: A1TaskResult,
        semantic: dict[str, Any],
    ) -> bool:
        metadata = result.task_metadata if isinstance(result.task_metadata, dict) else {}
        records = metadata.get("method_provenance", [])
        if not isinstance(records, list):
            return False
        actual_tools = {
            str(item.get("tool_name") or item.get("gateway_tool") or "")
            for item in result.tool_trace
            if isinstance(item, dict)
        }
        actual_tools.discard("")
        artifacts = {str(item) for item in result.artifacts}
        assertions = semantic.get("accepted_assertions", [])
        if not isinstance(assertions, list) or not assertions:
            return False
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            method = str(assertion.get("method", "")).strip()
            method_class = str(assertion.get("method_class", "")).strip()
            for record in records:
                if not isinstance(record, dict):
                    continue
                if str(record.get("method", "")).strip() != method:
                    continue
                if str(record.get("method_class", "")).strip() != method_class:
                    continue
                source_tool = str(
                    record.get("source_tool") or record.get("tool_name") or ""
                ).strip()
                source_artifact = str(
                    record.get("source_artifact") or record.get("artifact") or ""
                ).strip()
                if source_tool and source_tool in actual_tools:
                    return True
                if source_artifact and source_artifact in artifacts:
                    return True
        return False


class SMDDResultVerifier(StructuredResultVerifier):
    verifier_id = "omniagent.smdd_result.v1"
    evidence_type = "smdd_execution_result"


def build_task_result_verifier(
    task_manifest: dict[str, Any],
) -> TaskResultVerifier:
    task_id = str(task_manifest.get("id", "")).lower()
    parameters = task_manifest.get("task_parameters", {})
    benchmark = (
        str(parameters.get("benchmark", "")).lower()
        if isinstance(parameters, dict)
        else ""
    )
    if benchmark == "drugdiscoverybench":
        return DrugDiscoveryResultVerifier()
    if benchmark == "smdd" or task_id.startswith("smdd_"):
        return SMDDResultVerifier()
    return StructuredResultVerifier()
