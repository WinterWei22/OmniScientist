from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .execution_validation import validate_schema_instance


class ArtifactContractError(ValueError):
    """Raised when an execution result references an unverifiable artifact."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    path: str
    media_type: str
    size_bytes: int
    sha256: str
    declared_media_type: str | None = None
    content_schema: dict[str, Any] | None = None
    declared_sha256: str | None = None
    declared_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "path": self.path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.declared_media_type:
            value["declared_media_type"] = self.declared_media_type
        if self.content_schema is not None:
            value["content_schema"] = self.content_schema
        if self.declared_sha256 is not None:
            value["declared_sha256"] = self.declared_sha256
        if self.declared_size_bytes is not None:
            value["declared_size_bytes"] = self.declared_size_bytes
        return value


def normalize_artifact_declarations(value: Any) -> list[dict[str, Any]]:
    """Preserve artifact contract fields while accepting legacy path-only results."""
    items = value if isinstance(value, list | tuple) else [value]
    declarations: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            declarations.append(
                {str(key): value for key, value in item.items()}
            )
        else:
            path = str(item).strip()
            if path:
                declarations.append({"path": path})
    return declarations


def verify_artifacts(
    artifacts: list[Any],
    allowed_paths: list[str],
) -> list[VerifiedArtifact]:
    """Resolve and verify artifacts before they enter scientific state."""
    if not artifacts:
        return []
    roots = [Path(path).resolve() for path in allowed_paths]
    if not roots:
        raise ArtifactContractError("ARTIFACT_NOT_MATERIALIZED: no allowed artifact root")
    verified: list[VerifiedArtifact] = []
    for declaration in normalize_artifact_declarations(artifacts):
        raw_path = _artifact_path_value(declaration)
        supplied = Path(raw_path)
        candidates = (
            [supplied.resolve()]
            if supplied.is_absolute()
            else [(root / supplied).resolve() for root in roots]
        )
        artifact = next(
            (
                candidate
                for candidate in candidates
                if any(candidate.is_relative_to(root) for root in roots)
            ),
            None,
        )
        if artifact is None:
            raise ArtifactContractError(
                f"ARTIFACT_NOT_MATERIALIZED: artifact escapes allowed paths: {raw_path}"
            )
        if not artifact.is_file():
            raise ArtifactContractError(
                f"ARTIFACT_NOT_MATERIALIZED: artifact is missing or not a file: {artifact}"
            )
        try:
            data = artifact.read_bytes()
        except OSError as exc:
            raise ArtifactContractError(
                f"ARTIFACT_NOT_MATERIALIZED: artifact is unreadable: {artifact}: {exc}"
            ) from exc
        media_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
        json_value: Any = None
        if artifact.suffix.casefold() == ".json":
            try:
                json_value = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactContractError(
                    f"ARTIFACT_NOT_MATERIALIZED: JSON artifact is invalid: {artifact}"
                ) from exc
            media_type = "application/json"
        declared_media_type = _declared_media_type(declaration)
        if declared_media_type and not _media_types_compatible(
            media_type, declared_media_type
        ):
            raise ArtifactContractError(
                "ARTIFACT_MEDIA_TYPE_MISMATCH: "
                f"artifact {artifact} is {media_type}, not {declared_media_type}"
            )
        declared_sha256 = _declared_sha256(declaration)
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if declared_sha256 is not None and declared_sha256 != actual_sha256:
            raise ArtifactContractError(
                "ARTIFACT_CHECKSUM_MISMATCH: "
                f"artifact {artifact} has sha256 {actual_sha256}, expected {declared_sha256}"
            )
        declared_size_bytes = _declared_size_bytes(declaration)
        if declared_size_bytes is not None and declared_size_bytes != len(data):
            raise ArtifactContractError(
                "ARTIFACT_SIZE_MISMATCH: "
                f"artifact {artifact} has {len(data)} bytes, expected {declared_size_bytes}"
            )
        content_schema = _content_schema(declaration)
        if content_schema is not None:
            if media_type != "application/json":
                raise ArtifactContractError(
                    "ARTIFACT_SCHEMA_UNSUPPORTED: content_schema requires a JSON artifact: "
                    f"{artifact}"
                )
            errors = validate_schema_instance(json_value, content_schema)
            if errors:
                raise ArtifactContractError(
                    "ARTIFACT_SCHEMA_INVALID: " + "; ".join(errors[:3])
                )
        verified.append(
            VerifiedArtifact(
                path=str(artifact),
                media_type=media_type,
                size_bytes=len(data),
                sha256=actual_sha256,
                declared_media_type=declared_media_type,
                content_schema=content_schema,
                declared_sha256=declared_sha256,
                declared_size_bytes=declared_size_bytes,
            )
        )
    return verified


def _artifact_path_value(declaration: dict[str, Any]) -> str:
    raw = declaration.get("path") or declaration.get("uri")
    if not isinstance(raw, str) or not raw.strip():
        raise ArtifactContractError(
            "ARTIFACT_NOT_MATERIALIZED: artifact declaration is missing a path"
        )
    value = raw.strip()
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ArtifactContractError(
                "ARTIFACT_NOT_MATERIALIZED: non-local file URI is not supported"
            )
        return unquote(parsed.path)
    if "://" in value:
        raise ArtifactContractError(
            "ARTIFACT_NOT_MATERIALIZED: artifact URI must resolve to a local file"
        )
    return value


def _declared_media_type(declaration: dict[str, Any]) -> str | None:
    raw = declaration.get("media_type", declaration.get("mime_type"))
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ArtifactContractError("ARTIFACT_MEDIA_TYPE_INVALID: media_type must be a string")
    media_type = raw.split(";", 1)[0].strip().lower()
    if "/" not in media_type or media_type.startswith("/") or media_type.endswith("/"):
        raise ArtifactContractError(
            f"ARTIFACT_MEDIA_TYPE_INVALID: invalid media_type: {raw!r}"
        )
    return media_type


def _declared_sha256(declaration: dict[str, Any]) -> str | None:
    raw = declaration.get("sha256")
    if raw is None:
        raw = declaration.get("checksum")
    if isinstance(raw, dict):
        algorithm = str(raw.get("algorithm", raw.get("alg", "sha256"))).casefold()
        if algorithm != "sha256":
            raise ArtifactContractError(
                f"ARTIFACT_CHECKSUM_UNSUPPORTED: unsupported checksum algorithm: {algorithm}"
            )
        raw = raw.get("value", raw.get("digest"))
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ArtifactContractError("ARTIFACT_CHECKSUM_INVALID: sha256 must be a string")
    value = raw.strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ArtifactContractError(
            "ARTIFACT_CHECKSUM_INVALID: sha256 must be a 64-character hexadecimal digest"
        )
    return value


def _declared_size_bytes(declaration: dict[str, Any]) -> int | None:
    raw = declaration.get("size_bytes", declaration.get("size"))
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ArtifactContractError("ARTIFACT_SIZE_INVALID: size_bytes must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "ARTIFACT_SIZE_INVALID: size_bytes must be an integer"
        ) from exc
    if value < 0:
        raise ArtifactContractError("ARTIFACT_SIZE_INVALID: size_bytes cannot be negative")
    return value


def _media_types_compatible(actual: str, declared: str) -> bool:
    if actual == declared:
        return True
    return actual == "application/json" and declared.endswith("+json")


def _content_schema(declaration: dict[str, Any]) -> dict[str, Any] | None:
    schema = declaration.get("content_schema", declaration.get("schema"))
    if schema is None:
        return None
    if not isinstance(schema, dict):
        raise ArtifactContractError(
            "ARTIFACT_SCHEMA_INVALID: content_schema must be a JSON Schema object"
        )
    return dict(schema)
