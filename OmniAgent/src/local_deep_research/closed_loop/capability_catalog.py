from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _first_mapping(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Provider-neutral declaration for one executable capability.

    The manifest is deliberately data-only.  It describes what a tool accepts
    and returns, while the Harness still owns admission, invocation and
    verification policy.
    """

    canonical_name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    capability_version: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    effect_contract: dict[str, Any] = field(default_factory=dict)
    result_adapter: str = "generic"
    execution_mode: str = "sync"
    lifecycle: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    timeout_policy: dict[str, Any] = field(default_factory=dict)
    idempotency_policy: dict[str, Any] = field(default_factory=dict)
    provenance_policy: dict[str, Any] = field(default_factory=dict)
    result_kind: str = ""
    module: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CapabilityManifest | None:
        manifest = record.get("manifest")
        source = dict(record) | dict(manifest) if isinstance(manifest, dict) else dict(record)
        canonical_name = _first_text(
            source, "canonical_name", "qualified_name", "capability_id", "name", "id"
        )
        if not canonical_name:
            return None
        aliases = _string_tuple(source.get("aliases"))
        aliases = tuple(
            dict.fromkeys(
                (*aliases,)
                + tuple(
                    item
                    for item in (
                        _first_text(source, "name"),
                        _first_text(source, "qualified_name"),
                    )
                    if item and item != canonical_name
                )
            )
        )
        return cls(
            canonical_name=canonical_name,
            description=_first_text(source, "description", "summary"),
            aliases=aliases,
            capability_version=_first_text(
                source, "capability_version", "version", "schema_version"
            ),
            input_schema=_first_mapping(
                source, "input_schema", "inputSchema", "parameters_schema", "parameters"
            ),
            output_schema=_first_mapping(
                source, "output_schema", "outputSchema", "result_schema", "response_schema"
            ),
            effect_contract=_first_mapping(
                source, "effect_contract", "effects", "output_contract"
            ),
            result_adapter=_first_text(
                source, "result_adapter", "normalizer", "normalizer_id"
            )
            or "generic",
            execution_mode=_first_text(source, "execution_mode", "mode") or "sync",
            lifecycle=_first_mapping(source, "lifecycle", "task_lifecycle"),
            retry_policy=_first_mapping(source, "retry_policy", "retry"),
            timeout_policy=_first_mapping(source, "timeout_policy", "timeouts"),
            idempotency_policy=_first_mapping(source, "idempotency_policy", "idempotency"),
            provenance_policy=_first_mapping(source, "provenance_policy", "provenance"),
            result_kind=_first_text(source, "result_kind", "result_type"),
            module=_first_text(source, "module", "provider", "source"),
            metadata=(
                dict(source.get("metadata"))
                if isinstance(source.get("metadata"), dict)
                else {}
            ),
        )

    @property
    def schema_bound(self) -> bool:
        return bool(self.input_schema or self.output_schema)

    def to_record(self, *, score: float | None = None) -> dict[str, Any]:
        value = {
            "name": self.canonical_name.rsplit(".", 1)[-1],
            "qualified_name": self.canonical_name,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "description": self.description,
            "capability_version": self.capability_version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "effect_contract": self.effect_contract,
            "result_adapter": self.result_adapter,
            "execution_mode": self.execution_mode,
            "lifecycle": self.lifecycle,
            "retry_policy": self.retry_policy,
            "timeout_policy": self.timeout_policy,
            "idempotency_policy": self.idempotency_policy,
            "provenance_policy": self.provenance_policy,
            "result_kind": self.result_kind,
            "module": self.module,
        }
        if self.metadata:
            value["metadata"] = self.metadata
        if score is not None:
            value["score"] = round(float(score), 6)
        return value


def catalog_revision(manifests: list[CapabilityManifest]) -> str:
    """Derive a stable revision when Biomni did not publish one."""
    payload = [
        item.to_record()
        for item in sorted(manifests, key=lambda item: item.canonical_name)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "derived:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def manifest_metadata(manifest: CapabilityManifest) -> dict[str, Any]:
    """Return only execution metadata safe to persist in a route snapshot."""
    value = asdict(manifest)
    value.pop("metadata", None)
    return value
