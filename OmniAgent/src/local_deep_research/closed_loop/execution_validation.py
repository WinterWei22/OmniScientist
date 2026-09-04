from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .execution_models import EffectContract


def normalize_mcp_schema(schema: dict[str, Any], *, strict_objects: bool) -> dict[str, Any]:
    """Normalize Biomni's permissive ``type: any`` into JSON Schema 2020-12."""
    normalized = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "any":
                node.pop("type", None)
            if (
                strict_objects
                and node.get("type") == "object"
                and isinstance(node.get("properties"), dict)
                and "additionalProperties" not in node
            ):
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(normalized)
    return normalized


def validate_schema_instance(
    value: Any,
    schema: dict[str, Any],
    *,
    strict_objects: bool = False,
) -> list[str]:
    if not schema:
        return []
    normalized = normalize_mcp_schema(schema, strict_objects=strict_objects)
    try:
        validator = Draft202012Validator(normalized)
        validator.check_schema(normalized)
    except SchemaError as exc:
        return [f"invalid tool schema: {exc.message}"]
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    messages: list[str] = []
    for error in errors:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path
        )
        messages.append(f"{path}: {error.message}")
    return messages


def schema_declares_path(schema: Any, path: str) -> bool:
    """Return whether a JSON Schema declares a concrete nested output path.

    This is intentionally stricter than merely checking the root property.  A
    capability is admitted only when its schema follows every object and array
    segment needed by the effect contract.
    """
    parts = [part for part in str(path).split(".") if part]
    if not parts or not isinstance(schema, dict):
        return False

    def branches(node: dict[str, Any]) -> list[dict[str, Any]]:
        selected = [node]
        for keyword in ("allOf", "anyOf", "oneOf"):
            values = node.get(keyword)
            if isinstance(values, list):
                nested = [item for item in values if isinstance(item, dict)]
                if keyword == "allOf":
                    selected.extend(nested)
                elif nested:
                    selected = nested
        return selected

    def follows(node: Any, remaining: list[str]) -> bool:
        if not isinstance(node, dict) or str(node.get("type", "")).lower() == "any":
            return False
        if not remaining:
            return True
        for branch in branches(node):
            if branch is not node and follows(branch, remaining):
                return True
        part = remaining[0]
        if part.endswith("[*]"):
            property_name = part[:-3]
            properties = node.get("properties")
            child = properties.get(property_name) if isinstance(properties, dict) else None
            if not isinstance(child, dict) or str(child.get("type", "")).lower() == "any":
                return False
            items = child.get("items")
            return follows(items, remaining[1:]) if isinstance(items, dict) else False
        properties = node.get("properties")
        child = properties.get(part) if isinstance(properties, dict) else None
        return follows(child, remaining[1:]) if isinstance(child, dict) else False

    return follows(schema, parts)


def select_path(value: Any, path: str) -> list[Any]:
    """Select values using a bounded dotted path with optional ``[*]`` wildcards."""
    cursors = [value]
    for raw_part in (part for part in path.split(".") if part):
        wildcard = raw_part.endswith("[*]")
        part = raw_part[:-3] if wildcard else raw_part
        next_values: list[Any] = []
        for cursor in cursors:
            selected = cursor.get(part) if part and isinstance(cursor, dict) else cursor
            if wildcard:
                if isinstance(selected, list):
                    next_values.extend(selected)
            elif selected is not None:
                next_values.append(selected)
        cursors = next_values
        if not cursors:
            break
    return cursors


def has_material_value(values: list[Any]) -> bool:
    return any(
        value is not None
        and value != ""
        and value != []
        and value != {}
        for value in values
    )


@dataclass(frozen=True, slots=True)
class EffectVerification:
    passed: bool
    required_paths: tuple[str, ...]
    matched_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    any_of_paths: tuple[str, ...]
    matched_any_of_paths: tuple[str, ...]
    required_value_matches: tuple[dict[str, Any], ...]
    missing_value_matches: tuple[dict[str, Any], ...]
    required_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_effects(
    payload: Any,
    contract: EffectContract,
    *,
    artifacts: list[str] | tuple[str, ...] = (),
    allowed_paths: list[str] | tuple[str, ...] = (),
) -> EffectVerification:
    matched = tuple(
        path for path in contract.required_paths if has_material_value(select_path(payload, path))
    )
    missing = tuple(path for path in contract.required_paths if path not in matched)
    matched_any = tuple(
        path for path in contract.any_of_paths if has_material_value(select_path(payload, path))
    )
    matched_values: list[dict[str, Any]] = []
    missing_values: list[dict[str, Any]] = []
    for requirement in contract.required_value_matches:
        observed = select_path(payload, requirement.path)
        expected = tuple(str(item) for item in requirement.expected_values)
        if not requirement.case_sensitive:
            expected = tuple(item.casefold() for item in expected)
            observed_values = tuple(str(item).casefold() for item in observed)
        else:
            observed_values = tuple(str(item) for item in observed)
        record = {
            "path": requirement.path,
            "expected_values": list(requirement.expected_values),
            "observed_values": list(observed_values),
        }
        if any(value in expected for value in observed_values):
            matched_values.append(record)
        else:
            missing_values.append(record)
    roots = [Path(item).resolve() for item in allowed_paths]
    resolved_artifacts: list[Path] = []
    for raw in artifacts:
        supplied = Path(raw)
        if supplied.is_absolute():
            resolved_artifacts.append(supplied.resolve())
        else:
            resolved_artifacts.extend((root / supplied).resolve() for root in roots)
    missing_artifacts = []
    for expected in contract.required_artifacts:
        if not any(
            (candidate.name == expected or candidate.as_posix().endswith(expected))
            and candidate.is_file()
            and (not roots or any(candidate.is_relative_to(root) for root in roots))
            for candidate in resolved_artifacts
        ):
            missing_artifacts.append(expected)
    passed = (
        not missing
        and (not contract.any_of_paths or bool(matched_any))
        and not missing_values
        and not missing_artifacts
    )
    return EffectVerification(
        passed=passed,
        required_paths=contract.required_paths,
        matched_paths=matched,
        missing_paths=missing,
        any_of_paths=contract.any_of_paths,
        matched_any_of_paths=matched_any,
        required_value_matches=tuple(matched_values),
        missing_value_matches=tuple(missing_values),
        required_artifacts=contract.required_artifacts,
        missing_artifacts=tuple(missing_artifacts),
    )
