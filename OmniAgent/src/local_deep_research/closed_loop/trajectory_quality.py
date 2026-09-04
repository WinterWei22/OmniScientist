from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class TrajectoryQualityReport:
    route_count: int
    mcp_route_count: int
    a1_route_count: int
    unavailable_route_count: int
    binding_rate: float
    schema_valid_rate: float
    effect_completion_rate: float
    duplicate_invocation_rate: float
    evidence_admission_precision: float
    invalid_evidence_admissions: int
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceReplayEvaluator:
    """Evaluate saved Harness events without re-running models or tools."""

    def evaluate(self, events: Iterable[dict[str, Any]]) -> TrajectoryQualityReport:
        decisions: dict[tuple[Any, Any], dict[str, Any]] = {}
        executions: dict[tuple[Any, Any], dict[str, Any]] = {}
        verifications: dict[tuple[Any, Any], dict[str, Any]] = {}
        completed_routes: list[dict[str, Any]] = []
        issues: list[str] = []
        for record in events:
            if not isinstance(record, dict):
                continue
            event = str(record.get("event", ""))
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                continue
            key = (payload.get("iteration"), payload.get("step_id"))
            if event == "execution_route_decided":
                decisions[key] = payload
            elif event == "execution_route_completed":
                completed_routes.append(payload)
            elif event in {"execution_completed", "a1_execution_completed"}:
                executions[key] = payload
            elif event == "execution_result_verified":
                verifications[key] = payload

        route_count = len(decisions)
        mcp = [item for item in decisions.values() if item.get("backend") == "mcp"]
        a1 = [item for item in decisions.values() if item.get("backend") == "a1"]
        unavailable = [
            item for item in decisions.values() if item.get("backend") == "unavailable"
        ]
        bound = [
            item
            for item in mcp
            if isinstance(item.get("bound_call"), dict)
            or isinstance(item.get("bound_workflow"), dict)
        ]
        if len(bound) < len(mcp):
            issues.append("One or more MCP routes lacked a validated execution binding.")

        schema_checks = 0
        schema_valid = 0
        effect_checks = 0
        effect_passed = 0
        invalid_admission_keys: set[tuple[Any, Any]] = set()
        for key, execution in executions.items():
            metadata = execution.get("biomni_task")
            if not isinstance(metadata, dict):
                continue
            output_errors = metadata.get("output_schema_errors")
            if isinstance(output_errors, list):
                schema_checks += 1
                schema_valid += int(not output_errors)
            effect = metadata.get("effect_verification")
            if isinstance(effect, dict):
                required = any(
                    effect.get(name)
                    for name in (
                        "required_paths",
                        "any_of_paths",
                        "required_artifacts",
                        "missing_paths",
                        "missing_artifacts",
                    )
                )
                if required:
                    effect_checks += 1
                    effect_passed += int(effect.get("passed") is True)
                verification = verifications.get(key)
                if (
                    isinstance(verification, dict)
                    and verification.get("accepted") is True
                    and effect.get("passed") is not True
                ):
                    invalid_admission_keys.add(key)
        for key, verification in verifications.items():
            if verification.get("accepted") is not True:
                continue
            decision = decisions.get(key, {})
            if decision.get("backend") == "mcp" and not (
                isinstance(decision.get("bound_call"), dict)
                or isinstance(decision.get("bound_workflow"), dict)
            ):
                invalid_admission_keys.add(key)
        invalid_admissions = len(invalid_admission_keys)
        if invalid_admissions:
            issues.append(
                "Verifier admitted evidence from an unbound MCP route or an execution "
                "with unmet effects."
            )

        invoked_fingerprints = [
            self._canonical_route_fingerprint(item)
            for item in completed_routes
            if item.get("execution_invoked") is True
        ]
        invoked_fingerprints = [item for item in invoked_fingerprints if item]
        duplicate_count = len(invoked_fingerprints) - len(set(invoked_fingerprints))
        if duplicate_count:
            issues.append("The same canonical execution fingerprint was invoked repeatedly.")

        accepted_count = sum(
            item.get("accepted") is True for item in verifications.values()
        )
        evidence_precision = (
            (accepted_count - invalid_admissions) / accepted_count
            if accepted_count
            else 1.0
        )
        return TrajectoryQualityReport(
            route_count=route_count,
            mcp_route_count=len(mcp),
            a1_route_count=len(a1),
            unavailable_route_count=len(unavailable),
            binding_rate=len(bound) / len(mcp) if mcp else 1.0,
            schema_valid_rate=(
                schema_valid / schema_checks if schema_checks else 1.0
            ),
            effect_completion_rate=(
                effect_passed / effect_checks if effect_checks else 1.0
            ),
            duplicate_invocation_rate=(
                duplicate_count / len(invoked_fingerprints)
                if invoked_fingerprints
                else 0.0
            ),
            evidence_admission_precision=evidence_precision,
            invalid_evidence_admissions=invalid_admissions,
            issues=tuple(issues),
        )

    @staticmethod
    def _canonical_route_fingerprint(route: dict[str, Any]) -> str:
        bound_call = route.get("bound_call")
        bound_workflow = route.get("bound_workflow")
        if isinstance(bound_call, dict):
            payload: dict[str, Any] = {
                "backend": route.get("backend"),
                "bound_call": bound_call,
            }
        elif isinstance(bound_workflow, dict):
            payload = {
                "backend": route.get("backend"),
                "bound_workflow": bound_workflow,
            }
        else:
            intent = route.get("semantic_intent")
            semantic = intent if isinstance(intent, dict) else {}
            capability = route.get("selected_capability") or route.get(
                "admitted_capability"
            )
            if not capability:
                return str(route.get("route_signature") or "")
            payload = {
                "backend": route.get("backend"),
                "capability": capability,
                "reason_code": route.get("reason_code"),
                "semantic_contract": {
                    "operation": semantic.get("operation"),
                    "execution_shape": semantic.get("execution_shape"),
                    "side_effect": semantic.get("side_effect"),
                    "required_output_fields": semantic.get(
                        "required_output_fields", []
                    ),
                    "expected_artifacts": semantic.get("expected_artifacts", []),
                },
            }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def load_jsonl_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL event at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL event at line {line_number} is not an object")
        events.append(value)
    return events


def evaluate_trace_file(path: str | Path) -> TrajectoryQualityReport:
    return TraceReplayEvaluator().evaluate(load_jsonl_events(path))
