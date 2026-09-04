from __future__ import annotations

from typing import Any


_NON_RETRYABLE_CONTRACT_MARKERS = (
    "capability_unavailable",
    "capability_binding",
    "duplicate_route_blocked",
    "duplicate_action_blocked",
    "idempotency_conflict",
    "effect_verification",
    "effect_verification_failed",
    "invalid_arguments",
    "invalid_workflow",
    "invalid_workflow_arguments",
    "workflow_binding_failed",
    "workflow_output_schema_failed",
    "workflow_effect_unmet",
    "output_contract_failed",
    "verification_rejected",
    "domain_evidence_contract_failed",
    "method provenance record",
)


def execution_failure_retryable(result: Any) -> bool:
    """Classify exact-action retries independently from Planner replanning."""
    metadata = getattr(result, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("idempotency_conflict"):
        return False
    text = " ".join(
        [
            str(getattr(result, "result_status", "")),
            *(str(item) for item in getattr(result, "errors", []) or []),
        ]
    ).lower()
    if any(marker in text for marker in _NON_RETRYABLE_CONTRACT_MARKERS):
        return False
    explicit = metadata.get("error_retryable")
    if not isinstance(explicit, bool):
        explicit = metadata.get("retryable")
    if isinstance(explicit, bool):
        return explicit
    return True
