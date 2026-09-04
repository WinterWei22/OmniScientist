from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .answer_validation import (
    infer_answer_semantic_contract,
    validate_final_answer_semantics,
)
from .grounding import GroundingIndex
from .scientific_state import ClaimRecord, ClaimStatus

if TYPE_CHECKING:
    from .contracts import Critique, ResearchState, WorkflowEvaluation


@dataclass(frozen=True, slots=True)
class FinalizationDecision:
    eligible: bool
    blockers: tuple[str, ...] = ()
    output_valid: bool = False
    evidence_coverage: float = 0.0
    covered_claim_ids: tuple[str, ...] = ()
    required_claim_ids: tuple[str, ...] = ()
    semantic_valid: bool = True
    semantic_evidence_ids: tuple[str, ...] = ()
    objective_satisfied: bool = False
    mode: str = "result"
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinalizationGate:
    """Deterministic completion gate over output, claims, and admitted evidence."""

    def evaluate(
        self,
        state: ResearchState,
        evaluation: WorkflowEvaluation | None,
        critique: Critique | None = None,
    ) -> FinalizationDecision:
        scientific = state.scientific_state
        finalization = state.task_manifest.get("finalization", {})
        config = dict(finalization) if isinstance(finalization, dict) else {}
        completion_contract = state.task_manifest.get("completion_contract")
        has_explicit_contract = bool(config)
        if isinstance(completion_contract, dict) and completion_contract:
            config.update(completion_contract)
            has_explicit_contract = True
        benchmark = self._benchmark(state.task_manifest)
        blockers: list[str] = []
        if critique is not None and critique.requires_consumption:
            blockers.append(
                "Verifier feedback requires another iteration before completion."
            )
        ledger = getattr(state, "action_ledger", None)
        unresolved_actions = ledger.unresolved() if ledger is not None else []
        for action in unresolved_actions:
            blockers.append(
                f"Action {action.action_id} is unresolved ({action.status}); "
                "completion requires reconciliation."
            )
        if state.pending_execution is not None:
            blockers.append(
                "An external Biomni task is still pending: "
                + state.pending_execution.task_id
            )
        if ledger is not None:
            for action in ledger.records.values():
                if action.status == "timed_out":
                    blockers.append(
                        f"Action {action.action_id} timed out; a timeout cannot produce SUCCESS."
                    )

        output_config = state.task_manifest.get("output_config", {})
        has_output_contract = isinstance(output_config, dict) and bool(output_config)
        if (
            has_output_contract
            and not has_explicit_contract
            and benchmark not in {"drugdiscoverybench", "smdd"}
        ):
            blockers.append(
                "Finalization requires task_manifest.completion_contract for "
                "output-configured tasks."
            )
        require_grounded = bool(
            config.get(
                "require_grounded_output",
                has_output_contract
                and (
                    has_explicit_contract
                    or benchmark in {"drugdiscoverybench", "smdd"}
                ),
            )
        )

        payload, output_valid, output_blockers = self._validate_output(
            state, evaluation, config
        )
        blockers.extend(output_blockers)
        insufficient = bool(
            isinstance(payload, dict)
            and payload.get("status") == "INSUFFICIENT_EVIDENCE"
        )
        if insufficient:
            reason = str(payload.get("reason", "")).strip()
            allow_insufficient = bool(
                config.get(
                    "allow_insufficient_evidence",
                    benchmark in {"drugdiscoverybench", "smdd"},
                )
            )
            if not allow_insufficient:
                blockers.append(
                    "INSUFFICIENT_EVIDENCE is not allowed by the completion contract."
                )
            if not reason:
                blockers.append("INSUFFICIENT_EVIDENCE requires a concise reason.")
            if not scientific.attempts:
                blockers.append(
                    "INSUFFICIENT_EVIDENCE requires at least one recorded acquisition attempt."
                )
            return FinalizationDecision(
                eligible=not blockers,
                blockers=tuple(dict.fromkeys(blockers)),
                output_valid=output_valid,
                evidence_coverage=1.0 if scientific.attempts else 0.0,
                objective_satisfied=False,
                mode="insufficient_evidence",
                metrics={
                    "attempt_count": float(len(scientific.attempts)),
                    "evidence_count": float(len(scientific.evidence)),
                    "unresolved_action_count": float(len(unresolved_actions)),
                },
            )

        minimum_evidence = int(
            config.get(
                "minimum_evidence_records",
                1
                if has_explicit_contract or benchmark in {"drugdiscoverybench", "smdd"}
                else 0,
            )
        )
        if len(scientific.evidence) < minimum_evidence:
            blockers.append(
                f"Finalization requires at least {minimum_evidence} verifier-admitted "
                "evidence record(s)."
            )

        required_claims, missing_claims = self._required_claims(
            scientific.claims, config
        )
        blockers.extend(
            f"Required claim is missing from scientific state: {item}"
            for item in missing_claims
        )
        ddb_claim_blockers: list[str] = []
        if benchmark == "drugdiscoverybench" and not config.get("required_claims"):
            required_claims, missing_output_claims = self._ddb_output_claims(
                payload, scientific.claims, scientific.evidence
            )
            ddb_claim_blockers = missing_output_claims
            blockers.extend(missing_output_claims)

        require_verified = bool(config.get("require_verified_claims", False))
        covered: list[str] = []
        for claim in required_claims:
            if require_verified and claim.status is not ClaimStatus.VERIFIED:
                continue
            if claim.evidence_ids and set(claim.evidence_ids).issubset(scientific.evidence):
                covered.append(claim.claim_id)
        coverage = len(covered) / len(required_claims) if required_claims else 1.0
        if benchmark == "drugdiscoverybench" and (
            not output_valid or ddb_claim_blockers
        ):
            coverage = 0.0
        minimum_coverage = float(
            config.get(
                "minimum_claim_evidence_coverage",
                1.0 if required_claims else 0.0,
            )
        )
        if coverage < minimum_coverage:
            blockers.append(
                "Claim evidence coverage is below the required threshold: "
                f"{coverage:.3f} < {minimum_coverage:.3f}."
            )
        if require_verified and len(covered) < len(required_claims):
            blockers.append("One or more required claims are not verified.")

        if require_grounded:
            blockers.extend(
                self._grounded_output_blockers(
                    payload,
                    scientific.claims,
                    scientific.evidence,
                )
            )

        semantic = validate_final_answer_semantics(
            infer_answer_semantic_contract(state.goal),
            payload,
            required_claims,
            scientific.evidence,
            self._payload_evidence_ids(payload),
        )
        blockers.extend(semantic["blockers"])

        return FinalizationDecision(
            eligible=not blockers,
            blockers=tuple(dict.fromkeys(blockers)),
            output_valid=output_valid,
            evidence_coverage=coverage,
            covered_claim_ids=tuple(covered),
            required_claim_ids=tuple(item.claim_id for item in required_claims),
            semantic_valid=bool(semantic["passed"]),
            semantic_evidence_ids=tuple(semantic["evidence_ids"]),
            objective_satisfied=not blockers,
            metrics={
                "evidence_count": float(len(scientific.evidence)),
                "required_claim_count": float(len(required_claims)),
                "covered_claim_count": float(len(covered)),
                "minimum_claim_evidence_coverage": minimum_coverage,
                "grounded_output_required": float(require_grounded),
                "semantic_answer_required": float(bool(semantic["required"])),
                "semantic_answer_valid": float(bool(semantic["passed"])),
                "unresolved_action_count": float(len(unresolved_actions)),
            },
        )

    def _validate_output(
        self,
        state: ResearchState,
        evaluation: WorkflowEvaluation | None,
        config: dict[str, Any],
    ) -> tuple[Any, bool, list[str]]:
        output_config = state.task_manifest.get("output_config", {})
        if not isinstance(output_config, dict) or not output_config:
            return None, True, []
        blockers: list[str] = []
        name = str(output_config.get("file_path", "")).strip()
        if not name:
            return None, False, ["Output contract is missing output_config.file_path."]
        workspace = Path(state.workspace).resolve()
        output = (workspace / name).resolve()
        if not output.is_relative_to(workspace):
            return None, False, ["Configured final output escapes the workspace."]
        if not output.is_file():
            return None, False, [f"Required final output is missing: {name}"]
        if not getattr(state, "final_output_materialized", False):
            blockers.append(
                "Required final output was not materialized by the Harness."
            )

        payload: Any = None
        output_format = str(output_config.get("format", "")).lower()
        if output_format == "json" or output.suffix.lower() == ".json":
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
            except Exception as exc:
                return None, False, [f"Final JSON output is invalid: {type(exc).__name__}: {exc}"]
            schema = output_config.get("schema")
            if isinstance(schema, dict):
                blockers.extend(self._validate_schema(payload, schema, path="$"))
            required_fields = config.get("required_output_fields", [])
            if (
                isinstance(payload, dict)
                and payload.get("status") == "INSUFFICIENT_EVIDENCE"
                and config.get("allow_insufficient_evidence", False)
            ):
                required_fields = ["status", "reason"]
            if isinstance(required_fields, str):
                required_fields = [required_fields]
            for field_name in required_fields if isinstance(required_fields, list) else []:
                if not self._has_path(payload, str(field_name)):
                    blockers.append(f"Final output is missing required field: {field_name}")

        if evaluation is not None:
            contract_valid = evaluation.metrics.get("contract_valid")
            if contract_valid is not None and contract_valid < 1.0:
                blockers.append("The task-specific deterministic output contract failed.")
            terminal_contract = evaluation.metrics.get("terminal_contract")
            if terminal_contract is not None and terminal_contract >= 1.0:
                if not (
                    isinstance(payload, dict)
                    and payload.get("status") == "INSUFFICIENT_EVIDENCE"
                ):
                    blockers.append("Terminal contract metric lacks a matching output status.")
        return payload, not blockers, blockers

    @staticmethod
    def _payload_evidence_ids(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []
        values: list[Any] = []
        for key in ("evidence_id", "evidence_ids"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                values.extend(value)
            elif value not in (None, ""):
                values.append(value)
        semantic = payload.get("semantic_evidence")
        items = semantic if isinstance(semantic, list) else [semantic]
        for item in items:
            if isinstance(item, dict) and item.get("evidence_id_ref") not in (None, ""):
                values.append(item["evidence_id_ref"])
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))

    @staticmethod
    def _grounded_output_blockers(
        payload: Any,
        claims: dict[str, ClaimRecord],
        evidence: dict[str, Any],
    ) -> list[str]:
        if not isinstance(payload, dict):
            return ["Grounded output validation requires a JSON object."]
        _matches, blockers = GroundingIndex(claims, evidence).output_matches(payload)
        return blockers

    @staticmethod
    def _required_claims(
        claims: dict[str, ClaimRecord],
        config: dict[str, Any],
    ) -> tuple[list[ClaimRecord], list[str]]:
        requirements = config.get("required_claims", [])
        if isinstance(requirements, str | dict):
            requirements = [requirements]
        if not isinstance(requirements, list):
            return [], []
        selected: list[ClaimRecord] = []
        missing: list[str] = []
        for requirement in requirements:
            if isinstance(requirement, str):
                match = claims.get(requirement)
                if match is not None and match not in selected:
                    selected.append(match)
                elif match is None:
                    missing.append(requirement)
                continue
            if not isinstance(requirement, dict):
                continue
            claim_id = str(requirement.get("claim_id", "")).strip()
            contains = str(requirement.get("statement_contains", "")).casefold().strip()
            matched = False
            for item in claims.values():
                if claim_id and item.claim_id != claim_id:
                    continue
                if contains and contains not in item.statement.casefold():
                    continue
                matched = True
                if item not in selected:
                    selected.append(item)
            if not matched:
                missing.append(claim_id or contains or json.dumps(requirement))
        return selected, missing

    @staticmethod
    def _ddb_output_claims(
        payload: Any,
        claims: dict[str, ClaimRecord],
        evidence: dict[str, Any],
    ) -> tuple[list[ClaimRecord], list[str]]:
        if not isinstance(payload, dict):
            return [], ["DrugDiscoveryBench final output is not a JSON object."]
        index = GroundingIndex(claims, evidence)
        matches, blockers = index.output_matches(payload)
        if blockers:
            return [], blockers
        selected: list[ClaimRecord] = []
        for _path, match in matches:
            for claim_id in match.claim_ids:
                claim = claims[claim_id]
                if claim not in selected:
                    selected.append(claim)
        return selected, []

    @classmethod
    def _validate_schema(
        cls,
        value: Any,
        schema: dict[str, Any],
        *,
        path: str,
    ) -> list[str]:
        blockers: list[str] = []
        expected_type = schema.get("type")
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected is not None and (
            not isinstance(value, expected)
            or expected_type in {"number", "integer"} and isinstance(value, bool)
        ):
            return [f"{path} must have JSON type {expected_type}."]
        if "enum" in schema and value not in schema["enum"]:
            blockers.append(f"{path} is not one of the allowed values.")
        if isinstance(value, dict):
            required = schema.get("required", [])
            if isinstance(required, list):
                for key in required:
                    if str(key) not in value:
                        blockers.append(f"{path}.{key} is required.")
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                for key, child_schema in properties.items():
                    if key in value and isinstance(child_schema, dict):
                        blockers.extend(
                            cls._validate_schema(
                                value[key], child_schema, path=f"{path}.{key}"
                            )
                        )
        if isinstance(value, list):
            minimum = schema.get("minItems")
            if isinstance(minimum, int) and len(value) < minimum:
                blockers.append(f"{path} must contain at least {minimum} item(s).")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    blockers.extend(
                        cls._validate_schema(item, item_schema, path=f"{path}[{index}]")
                    )
        return blockers

    @staticmethod
    def _has_path(value: Any, dotted_path: str) -> bool:
        cursor = value
        for part in dotted_path.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return False
            cursor = cursor[part]
        return cursor is not None and (not isinstance(cursor, str) or bool(cursor.strip()))

    @staticmethod
    def _benchmark(manifest: dict[str, Any]) -> str:
        parameters = manifest.get("task_parameters", {})
        if not isinstance(parameters, dict):
            return ""
        return str(parameters.get("benchmark", "")).casefold()
