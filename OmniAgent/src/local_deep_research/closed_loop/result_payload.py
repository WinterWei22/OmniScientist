from __future__ import annotations

import json
from typing import Any


_METADATA_KEYS = frozenset(
    {
        "action_id",
        "analysis",
        "attempt_id",
        "backend",
        "capability_id",
        "confidence",
        "evidence",
        "evidence_id",
        "evidence_ids",
        "execution_parameters",
        "explanation",
        "filters_applied",
        "input_parameters",
        "iteration",
        "metadata",
        "provenance",
        "projection",
        "projection_metadata",
        "query_parameters",
        "rationale",
        "reason",
        "reasoning",
        "request_id",
        "request_parameters",
        "run_id",
        "semantic_evidence",
        "sources",
        "status",
        "step_id",
        "supporting_evidence_ids",
        "task_id",
        "verifier_id",
        "workflow_phase",
    }
)

DEFAULT_EVIDENCE_MAX_BYTES = 131072
DEFAULT_EVIDENCE_LEAF_LIMIT = 256


def extract_envelope_answer(value: Any) -> str:
    """Return a textual answer from a result envelope without treating output as prose."""
    if isinstance(value, dict):
        for key in ("answer", "content", "text"):
            if key in value:
                answer = extract_envelope_answer(value[key])
                if answer:
                    return answer
        return ""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, dict):
        return extract_envelope_answer(decoded)
    return text


def merge_envelope_answer(payload: Any, envelope_answer: Any) -> Any:
    """Add a missing envelope answer to a canonical final object, preserving explicit data."""
    if not isinstance(payload, dict):
        return payload
    if extract_envelope_answer(payload.get("answer")):
        return payload
    answer = extract_envelope_answer(envelope_answer)
    if not answer:
        return payload
    merged = dict(payload)
    merged["answer"] = answer
    return merged


def material_result_values(
    payload: Any,
    *,
    limit: int | None = 64,
) -> list[tuple[str, str | bool | int | float]]:
    """Return typed scientific result leaves, excluding control and provenance data."""
    if not isinstance(payload, dict):
        return []

    leaves: list[tuple[str, str | bool | int | float]] = []

    def visit(value: Any, path: str) -> None:
        if limit is not None and len(leaves) >= limit:
            return
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                child = value[key]
                key_text = str(key).strip()
                if not key_text or key_text.casefold() in _METADATA_KEYS:
                    continue
                child_path = f"{path}.{key_text}" if path else key_text
                visit(child, child_path)
            return
        if isinstance(value, list | tuple):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
                if limit is not None and len(leaves) >= limit:
                    break
            return
        if isinstance(value, str):
            scalar: str | bool | int | float = value.strip()
        elif isinstance(value, bool | int | float):
            if isinstance(value, float):
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            scalar = value
        else:
            return
        if scalar != "":
            leaves.append((path, scalar[:4000] if isinstance(scalar, str) else scalar))

    visit(payload, "")
    return leaves


def material_result_leaves(payload: Any, *, limit: int = 64) -> list[tuple[str, str]]:
    """Return bounded string projections of scientific result fields."""
    leaves: list[tuple[str, str]] = []
    for path, value in material_result_values(payload, limit=limit):
        if isinstance(value, str):
            text = value
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        leaves.append((path, text))
    return leaves


def compact_evidence_payload(
    payload: Any,
    *,
    max_bytes: int = DEFAULT_EVIDENCE_MAX_BYTES,
    leaf_limit: int = DEFAULT_EVIDENCE_LEAF_LIMIT,
) -> Any:
    """Keep small evidence exact and project large values into scalar facts."""
    normalized = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) <= max(1, int(max_bytes)):
        return normalized
    leaves = material_result_values(normalized, limit=max(1, int(leaf_limit)))
    projection = [
        {"path": path, "value": value}
        for path, value in leaves
    ]
    compacted = {
        "projection": projection,
        "projection_metadata": {
            "truncated": True,
            "original_size_bytes": len(encoded),
            "projected_scalar_count": len(projection),
            "projected_scalar_limit": max(1, int(leaf_limit)),
        },
    }
    while projection:
        projected_size = len(
            json.dumps(
                compacted,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        if projected_size <= max(1, int(max_bytes)):
            break
        projection.pop()
        compacted["projection_metadata"]["projected_scalar_count"] = len(projection)
    return compacted


def has_material_result(payload: Any) -> bool:
    return bool(material_result_leaves(payload))
