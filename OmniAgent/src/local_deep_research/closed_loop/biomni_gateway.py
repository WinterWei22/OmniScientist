from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .biomni_task_protocol import (
    BiomniTaskResolution,
    begin_biomni_submission,
    normalize_biomni_task_payload,
    parse_mcp_payload,
    poll_biomni_task,
    resolve_biomni_submission,
)
from .capability_catalog import CapabilityManifest, catalog_revision
from .qwen_embedding_cache import QwenEmbeddingCache


ToolLoader = Callable[[dict[str, Any]], Awaitable[list[Any]]]


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    canonical_name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    result_kind: str = ""
    provenance_required: bool | None = None
    module: str = ""
    capability_version: str = ""
    effect_contract: dict[str, Any] = field(default_factory=dict)
    result_adapter: str = "generic"
    execution_mode: str = "sync"
    lifecycle: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    timeout_policy: dict[str, Any] = field(default_factory=dict)
    idempotency_policy: dict[str, Any] = field(default_factory=dict)
    provenance_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self, *, score: float | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.canonical_name.rsplit(".", 1)[-1],
            "qualified_name": self.canonical_name,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "result_kind": self.result_kind,
            "module": self.module,
            "capability_version": self.capability_version,
            "effect_contract": self.effect_contract,
            "result_adapter": self.result_adapter,
            "execution_mode": self.execution_mode,
            "lifecycle": self.lifecycle,
            "retry_policy": self.retry_policy,
            "timeout_policy": self.timeout_policy,
            "idempotency_policy": self.idempotency_policy,
            "provenance_policy": self.provenance_policy,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        if score is not None:
            record["score"] = round(float(score), 6)
        if self.provenance_required is not None:
            record["provenance_policy"] = {
                "required": self.provenance_required,
            }
        return record


class BiomniGatewayClient:
    """Single MCP transport and protocol adapter for all Biomni executions."""

    SEARCH_TOOL_NAME = "biomni_search_tools"
    LIST_TOOL_NAME = "biomni_list_tools"
    CAPABILITIES_TOOL_NAME = "biomni_list_capabilities"
    INVOKE_TOOL_NAME = "biomni_invoke_tool"
    A1_TOOL_NAME = "call_biomni"
    STATUS_TOOL_NAME = "get_biomni_task"
    CANCEL_TOOL_NAME = "cancel_biomni_task"
    RETRIEVAL_INSTRUCTION = (
        "Given a multi-step biomedical research task, decide whether this single "
        "Biomni resource directly supports at least one step. The resource does not "
        "need to solve the whole task. Prefer exact database interfaces and specialized "
        "tools over generic analysis methods."
    )

    def __init__(
        self,
        server_config: dict[str, Any],
        *,
        tool_loader: ToolLoader | None = None,
        task_poll_interval_seconds: float = 2.0,
        task_timeout_seconds: float = 900.0,
        submission_timeout_seconds: float = 60.0,
    ) -> None:
        self.server_config = dict(server_config)
        token = os.getenv("BIOMNI_MCP_AUTH_TOKEN", "").strip()
        if token:
            headers = dict(self.server_config.get("headers") or {})
            headers.setdefault("Authorization", f"Bearer {token}")
            self.server_config["headers"] = headers
        if self.server_config.get("transport") == "sse":
            self.server_config.setdefault("timeout", 30.0)
            self.server_config.setdefault("sse_read_timeout", 900.0)
        self._tool_loader = tool_loader
        self._client: Any = None
        self._tools: dict[str, Any] = {}
        self._initialized = False
        self.discovered_tool_names: tuple[str, ...] = ()
        self._catalog: tuple[CapabilityDescriptor, ...] = ()
        self._catalog_by_canonical: dict[str, CapabilityDescriptor] = {}
        self._alias_to_canonical: dict[str, str] = {}
        self.catalog_protocol = "legacy"
        self.catalog_revision = ""
        self.catalog_error = ""
        self._embedding_cache: QwenEmbeddingCache | None = None
        self._a1_capabilities: tuple[dict[str, Any], ...] = ()
        self._inflight_submissions: dict[tuple[str, str], asyncio.Future[Any]] = {}
        self._inflight_polls: dict[str, asyncio.Future[Any]] = {}
        self._search_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.task_poll_interval_seconds = max(0.0, float(task_poll_interval_seconds))
        self.task_timeout_seconds = max(0.0, float(task_timeout_seconds))
        self.submission_timeout_seconds = max(0.0, float(submission_timeout_seconds))
        self.poll_request_timeout_seconds = min(
            30.0, self.submission_timeout_seconds
        )

    @property
    def supports_task_v1(self) -> bool:
        return self.catalog_protocol == "biomni.capability.v1"

    def supports_request_correlation(self, gateway_tool_name: str) -> bool:
        """Return whether a gateway tool explicitly accepts a request ID.

        Catalog and execution protocols are versioned independently during a
        rollout. The MCP input schema is the authoritative contract for an
        invocation argument. Capability-v1 is only a fallback for compatible
        older clients that did not expose an argument schema at all.
        """
        tool = self._tools.get(gateway_tool_name)
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            return self._schema_declares_property(schema, "request_id")
        return self.supports_task_v1

    def supports_argument(self, gateway_tool_name: str, property_name: str) -> bool:
        """Use the advertised MCP argument schema as the invocation contract."""
        tool = self._tools.get(gateway_tool_name)
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            return self._schema_declares_property(schema, property_name)
        if property_name == "request_id":
            return self.supports_task_v1
        return (
            gateway_tool_name == self.A1_TOOL_NAME
            and property_name == "result_contract"
            and self.supports_task_v1
        )

    @staticmethod
    def _schema_declares_property(schema: Any, property_name: str) -> bool:
        if schema is None:
            return False
        export_schema = getattr(schema, "model_json_schema", None)
        if callable(export_schema):
            try:
                schema = export_schema()
            except (TypeError, ValueError):
                return False
        if not isinstance(schema, dict):
            return False
        properties = schema.get("properties")
        return isinstance(properties, dict) and property_name in properties

    async def initialize(self, *, load_catalog: bool = True) -> None:
        if self._initialized:
            if load_catalog and not self._catalog:
                try:
                    await self.refresh_catalog()
                except Exception as exc:
                    self.catalog_error = f"{type(exc).__name__}: {exc}"
            return
        tools = await self._load_tools()
        self._tools = {
            str(getattr(tool, "name", "")): tool
            for tool in tools
            if str(getattr(tool, "name", ""))
        }
        self.discovered_tool_names = tuple(sorted(self._tools))
        self._initialized = True
        if load_catalog and (
            self.CAPABILITIES_TOOL_NAME in self._tools
            or self.LIST_TOOL_NAME in self._tools
        ):
            try:
                await self.refresh_catalog()
            except Exception as exc:
                self.catalog_error = f"{type(exc).__name__}: {exc}"

    async def refresh_catalog(self) -> list[dict[str, Any]]:
        catalog_tool_name = self.CAPABILITIES_TOOL_NAME
        catalog_tool = self._tools.get(catalog_tool_name)
        if catalog_tool is None:
            catalog_tool_name = self.LIST_TOOL_NAME
            catalog_tool = self._tools.get(catalog_tool_name)
        if catalog_tool is None:
            raise RuntimeError(
                "Biomni MCP gateway does not expose biomni_list_capabilities "
                "or biomni_list_tools"
            )
        catalog_arguments = (
            {}
            if catalog_tool_name == self.CAPABILITIES_TOOL_NAME
            else {"include_schema": True}
        )
        try:
            payload = parse_mcp_payload(await catalog_tool.ainvoke(catalog_arguments))
        except Exception:
            # Capability-v1 is authoritative, but a transient failure there
            # should not make a compatible legacy catalog disappear.
            if catalog_tool_name != self.CAPABILITIES_TOOL_NAME:
                raise
            fallback = self._tools.get(self.LIST_TOOL_NAME)
            if fallback is None:
                raise
            catalog_tool_name = self.LIST_TOOL_NAME
            payload = parse_mcp_payload(
                await fallback.ainvoke({"include_schema": True})
            )
        if not isinstance(payload, dict):
            raise RuntimeError(f"{catalog_tool_name} returned a non-object payload")
        body = payload
        if not isinstance(body.get("tools"), list) and isinstance(body.get("result"), dict):
            body = body["result"]
        if not isinstance(body.get("tools"), list) and isinstance(body.get("data"), dict):
            body = body["data"]
        raw_tools = body.get("tools") or body.get("capabilities")
        if not isinstance(raw_tools, list):
            raise RuntimeError(f"{catalog_tool_name} did not contain a tools list")

        descriptors: dict[str, CapabilityDescriptor] = {}
        alias_map: dict[str, str] = {}
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            descriptor = self._descriptor_from_record(raw)
            if descriptor is None:
                continue
            existing = descriptors.get(descriptor.canonical_name)
            if existing is not None:
                aliases = tuple(dict.fromkeys((*existing.aliases, *descriptor.aliases)))
                descriptor = CapabilityDescriptor(
                    canonical_name=descriptor.canonical_name,
                    description=descriptor.description or existing.description,
                    aliases=aliases,
                    input_schema=descriptor.input_schema or existing.input_schema,
                    output_schema=descriptor.output_schema or existing.output_schema,
                    result_kind=descriptor.result_kind or existing.result_kind,
                    provenance_required=(
                        descriptor.provenance_required
                        if descriptor.provenance_required is not None
                        else existing.provenance_required
                    ),
                    module=descriptor.module or existing.module,
                    capability_version=(
                        descriptor.capability_version or existing.capability_version
                    ),
                    effect_contract=descriptor.effect_contract or existing.effect_contract,
                    result_adapter=descriptor.result_adapter or existing.result_adapter,
                    execution_mode=descriptor.execution_mode or existing.execution_mode,
                    lifecycle=descriptor.lifecycle or existing.lifecycle,
                    retry_policy=descriptor.retry_policy or existing.retry_policy,
                    timeout_policy=descriptor.timeout_policy or existing.timeout_policy,
                    idempotency_policy=(
                        descriptor.idempotency_policy or existing.idempotency_policy
                    ),
                    provenance_policy=(
                        descriptor.provenance_policy or existing.provenance_policy
                    ),
                    metadata=descriptor.metadata or existing.metadata,
                )
            descriptors[descriptor.canonical_name] = descriptor
            for alias in (descriptor.canonical_name, *descriptor.aliases):
                alias_map.setdefault(alias, descriptor.canonical_name)

        self._catalog_by_canonical = descriptors
        self._alias_to_canonical = alias_map
        self._catalog = tuple(descriptors[name] for name in sorted(descriptors))
        self.catalog_protocol = str(body.get("protocol") or "legacy")
        declared_revision = str(body.get("catalog_revision") or "").strip()
        self.catalog_revision = declared_revision or catalog_revision(
            [
                CapabilityManifest(
                    canonical_name=item.canonical_name,
                    description=item.description,
                    aliases=item.aliases,
                    capability_version=item.capability_version,
                    input_schema=item.input_schema,
                    output_schema=item.output_schema,
                    effect_contract=item.effect_contract,
                    result_adapter=item.result_adapter,
                    execution_mode=item.execution_mode,
                    lifecycle=item.lifecycle,
                    retry_policy=item.retry_policy,
                    timeout_policy=item.timeout_policy,
                    idempotency_policy=item.idempotency_policy,
                    provenance_policy=item.provenance_policy,
                    result_kind=item.result_kind,
                    module=item.module,
                    metadata=item.metadata,
                )
                for item in self._catalog
            ]
        )
        raw_a1_capabilities = body.get("a1_capabilities", [])
        if isinstance(raw_a1_capabilities, list):
            self._a1_capabilities = tuple(
                {str(key): value for key, value in item.items()}
                for item in raw_a1_capabilities
                if isinstance(item, dict)
            )
        else:
            self._a1_capabilities = ()
        self.catalog_error = ""
        self._search_cache.clear()
        return [descriptor.to_record() for descriptor in self._catalog]

    def get_tool(self, name: str) -> Any:
        if not self._initialized:
            raise RuntimeError("BiomniGatewayClient.initialize() must be called first")
        tool = self._tools.get(name)
        if tool is None:
            raise RuntimeError(f"Biomni MCP gateway does not expose {name}")
        return tool

    def canonicalize_tool_name(self, name: str) -> str:
        normalized = str(name or "").strip()
        return self._alias_to_canonical.get(normalized, normalized)

    def get_capability_descriptor(self, name: str) -> CapabilityDescriptor | None:
        """Return the cached descriptor for an exact catalog capability."""
        return self._catalog_by_canonical.get(self.canonicalize_tool_name(name))

    def request_id_for(
        self, *, run_id: str, step_id: str, attempt: int = 1
    ) -> str:
        """Return a stable ID for one logical execution attempt."""
        normalized_attempt = max(1, int(attempt))
        return (
            f"omniagent:{str(run_id).strip()}:{str(step_id).strip()}"
            f":attempt-{normalized_attempt}"
        )[:256]

    def a1_capability_admission_error(
        self,
        *,
        operation: str,
        requires_method_provenance: bool,
    ) -> str | None:
        """Require an advertised A1 capability for method-sensitive execution."""
        if self.A1_TOOL_NAME not in self.discovered_tool_names:
            return "Biomni MCP does not expose call_biomni."
        if not self._a1_capabilities:
            # Biomni exposes A1 as the coarse-grained call_biomni job. Optional
            # capability metadata is checked on the returned evidence, not used
            # to block a valid submission.
            return None
        for capability in self._a1_capabilities:
            operations = capability.get("operations", [])
            if isinstance(operations, str):
                operations = [operations]
            operation_matches = not operations or operation in {
                str(item).strip().lower() for item in operations
            }
            if not operation_matches:
                continue
            supports_provenance = bool(
                capability.get("method_provenance")
                or capability.get("supports_method_provenance")
            )
            if requires_method_provenance and not supports_provenance:
                continue
            return None
        requirement = " with method provenance" if requires_method_provenance else ""
        return (
            "No advertised Biomni A1 capability supports operation "
            f"{operation!r}{requirement}."
        )

    async def search_capabilities(
        self,
        query: str,
        *,
        max_results: int,
    ) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("BiomniGatewayClient.initialize() must be called first")
        limit = max(1, min(int(max_results), 20))
        normalized_query = " ".join(str(query or "").split()).casefold()
        cache_key = (self.catalog_revision, normalized_query, limit)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return dict(cached) | {"cache_hit": True}
        ranking_error = ""
        if self._catalog:
            try:
                matches = await asyncio.to_thread(self._rank_catalog, query, limit)
                if matches:
                    result = {
                        "query": query,
                        "catalog_size": len(self._catalog),
                        "catalog_protocol": self.catalog_protocol,
                        "catalog_revision": self.catalog_revision,
                        "matches": matches,
                        "retrieval": "qwen3.7-text-embedding-cache",
                        "gateway_tool": self.LIST_TOOL_NAME,
                    }
                    self._cache_search(cache_key, result)
                    return result
                ranking_error = "semantic catalog retrieval returned no matches"
            except Exception as exc:
                ranking_error = f"{type(exc).__name__}: {exc}"

        search_tool = self._tools.get(self.SEARCH_TOOL_NAME)
        if search_tool is None:
            if self._catalog:
                result = {
                    "query": query,
                    "catalog_size": len(self._catalog),
                    "catalog_protocol": self.catalog_protocol,
                    "catalog_revision": self.catalog_revision,
                    "matches": self._lexical_catalog_matches(query, limit),
                    "retrieval": "catalog_lexical_fallback",
                    "gateway_tool": self.LIST_TOOL_NAME,
                    "semantic_error": ranking_error,
                }
                self._cache_search(cache_key, result)
                return result
            raise RuntimeError("Biomni MCP gateway does not expose a tool search endpoint")

        payload = parse_mcp_payload(
            await search_tool.ainvoke({"query": query, "max_results": limit})
        )
        if not isinstance(payload, dict):
            raise RuntimeError("biomni_search_tools returned a non-object payload")
        normalized = dict(payload)
        raw_matches = normalized.get("matches")
        if isinstance(raw_matches, list):
            normalized["matches"] = self._normalize_matches(raw_matches, limit)
        else:
            normalized["matches"] = []
        normalized.setdefault("catalog_size", len(self._catalog) or None)
        normalized.setdefault("catalog_protocol", self.catalog_protocol)
        normalized.setdefault("catalog_revision", self.catalog_revision)
        normalized.setdefault("retrieval", "mcp_lexical_fallback")
        normalized.setdefault("gateway_tool", self.SEARCH_TOOL_NAME)
        if ranking_error:
            normalized["semantic_error"] = ranking_error
        self._cache_search(cache_key, normalized)
        return normalized

    def _cache_search(
        self, key: tuple[str, str, int], value: dict[str, Any]
    ) -> None:
        if len(self._search_cache) >= 128:
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[key] = dict(value)

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        request_id: str,
        wait_for_terminal: bool = False,
    ) -> BiomniTaskResolution:
        canonical_name = self.canonicalize_tool_name(tool_name)
        return await self._submit(
            self.INVOKE_TOOL_NAME,
            {"tool_name": canonical_name, "arguments": arguments},
            request_id=request_id,
            wait_for_terminal=wait_for_terminal,
        )

    async def call_a1(
        self,
        prompt: str,
        *,
        request_id: str,
        result_contract: dict[str, Any] | None = None,
    ) -> BiomniTaskResolution:
        arguments: dict[str, Any] = {"prompt": prompt}
        if result_contract and self.supports_argument(
            self.A1_TOOL_NAME, "result_contract"
        ):
            arguments["result_contract"] = dict(result_contract)
        return await self._submit(
            self.A1_TOOL_NAME,
            arguments,
            request_id=request_id,
        )

    async def poll_task(self, task_metadata: dict[str, Any]) -> BiomniTaskResolution:
        """Poll a previously submitted task without issuing another submission."""
        resolution = await poll_biomni_task(
            task_metadata,
            status_tool=self._tools.get(self.STATUS_TOOL_NAME),
            parse_payload=parse_mcp_payload,
            invoke_status=self._invoke_status_with_timeout,
        )
        metadata = dict(resolution.metadata)
        metadata.setdefault("request_id", task_metadata.get("request_id", ""))
        metadata.setdefault("gateway_tool", task_metadata.get("gateway_tool", ""))
        resolution.metadata = metadata
        if resolution.poll_error.startswith("TimeoutError:"):
            resolution.trace.insert(
                0,
                {
                    "event": "biomni_gateway_poll_timed_out",
                    "gateway_tool": metadata.get("gateway_tool"),
                    "request_id": metadata.get("request_id"),
                    "task_id": metadata.get("task_id"),
                    "timeout_seconds": self.poll_request_timeout_seconds,
                },
            )
        resolution.trace.insert(
            0,
            {
                "event": "biomni_gateway_task_polled",
                "gateway_tool": metadata.get("gateway_tool"),
                "request_id": metadata.get("request_id"),
                "task_id": metadata.get("task_id"),
            },
        )
        return resolution

    def describe(self) -> dict[str, Any]:
        return {
            "transport": self.server_config.get("transport"),
            "url": self.server_config.get("url"),
            "catalog_protocol": self.catalog_protocol,
            "catalog_revision": self.catalog_revision or None,
            "catalog_size": len(self._catalog),
            "catalog_error": self.catalog_error or None,
            "a1_capability_count": len(self._a1_capabilities),
            "submission_timeout_seconds": self.submission_timeout_seconds,
            "poll_request_timeout_seconds": self.poll_request_timeout_seconds,
            "gateway_tools": list(self.discovered_tool_names),
            "request_correlation_gateway_tools": [
                tool_name
                for tool_name in (self.INVOKE_TOOL_NAME, self.A1_TOOL_NAME)
                if self.supports_request_correlation(tool_name)
            ],
            "a1_result_contract_supported": self.supports_argument(
                self.A1_TOOL_NAME, "result_contract"
            ),
        }

    async def _submit(
        self,
        gateway_tool_name: str,
        arguments: dict[str, Any],
        *,
        request_id: str,
        wait_for_terminal: bool = False,
    ) -> BiomniTaskResolution:
        request_id = str(request_id or "").strip()
        if not request_id or len(request_id) > 256:
            return self._submission_failure(
                gateway_tool_name=gateway_tool_name,
                request_id=request_id,
                event="biomni_gateway_invalid_request_id",
                error="request_id must be a non-empty string of at most 256 characters",
            )
        tool = self.get_tool(gateway_tool_name)
        request = dict(arguments)
        request_id_sent = self.supports_request_correlation(gateway_tool_name)
        if request_id_sent:
            request["request_id"] = request_id
        submission_key = (gateway_tool_name, request_id)
        existing_submission = self._inflight_submissions.get(submission_key)
        try:
            if existing_submission is not None and not existing_submission.done():
                # Share the in-flight submission for the same protocol identity.
                # A second caller must not create a competing transport request.
                completed, response = await self._await_rpc(
                    existing_submission,
                    timeout_seconds=self.submission_timeout_seconds,
                )
            else:
                submission_task = asyncio.ensure_future(
                    self._ainvoke_submission(tool, gateway_tool_name, request)
                )
                self._inflight_submissions[submission_key] = submission_task
                submission_task.add_done_callback(
                    lambda completed: self._finish_submission(submission_key, completed)
                )
                completed, response = await self._await_rpc(
                    submission_task,
                    timeout_seconds=self.submission_timeout_seconds,
                )
        except Exception as exc:
            return self._submission_failure(
                gateway_tool_name=gateway_tool_name,
                request_id=request_id,
                event="biomni_gateway_transport_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if not completed:
            timeout_seconds = self.submission_timeout_seconds
            return self._submission_failure(
                gateway_tool_name=gateway_tool_name,
                request_id=request_id,
                event="biomni_gateway_submission_timed_out",
                error=(
                    f"Biomni gateway submission to {gateway_tool_name} timed out after "
                    f"{timeout_seconds:g} seconds before a task ID was returned"
                ),
                timeout_seconds=timeout_seconds,
            )
        submission = normalize_biomni_task_payload(parse_mcp_payload(response))
        returned_request_id = (
            str(submission.get("request_id", "")).strip()
            if isinstance(submission, dict)
            else ""
        )
        if returned_request_id and returned_request_id != request_id:
            return self._submission_failure(
                gateway_tool_name=gateway_tool_name,
                request_id=request_id,
                event="biomni_gateway_request_id_mismatch",
                error=(
                    "Biomni returned a different request_id than the submitted logical "
                    "action; refusing to attach the task"
                ),
            )
        resolution = (
            await resolve_biomni_submission(
                submission,
                status_tool=self._tools.get(self.STATUS_TOOL_NAME),
                parse_payload=parse_mcp_payload,
                poll_interval_seconds=self.task_poll_interval_seconds,
                timeout_seconds=self.task_timeout_seconds,
            )
            if wait_for_terminal
            else begin_biomni_submission(submission)
        )
        metadata = dict(resolution.metadata)
        metadata.setdefault("request_id", request_id)
        metadata["gateway_tool"] = gateway_tool_name
        metadata["request_id_sent"] = request_id_sent
        if gateway_tool_name in {self.INVOKE_TOOL_NAME, self.A1_TOOL_NAME} and not request_id_sent:
            metadata["idempotency_warning"] = (
                "Biomni gateway schema did not advertise request_id; server-side "
                "idempotency could not be verified"
            )
        if self.catalog_protocol != "legacy":
            metadata.setdefault("capability_protocol", self.catalog_protocol)
        resolution.metadata = metadata
        resolution.trace.insert(
            0,
            {
                "event": "biomni_gateway_request_submitted",
                "gateway_tool": gateway_tool_name,
                "request_id": request_id,
                "request_id_sent": request_id_sent,
                "catalog_protocol": self.catalog_protocol,
            },
        )
        return resolution

    async def _ainvoke_submission(
        self,
        tool: Any,
        gateway_tool_name: str,
        request: dict[str, Any],
    ) -> Any:
        if self._tool_loader is not None:
            return await tool.ainvoke(request)

        # Reuse the initialized client/tool. Re-handshaking for every submit or
        # poll adds latency and can create overlapping MCP sessions. The caller
        # already detaches a stalled RPC and records it as UNKNOWN.
        return await tool.ainvoke(request)

    async def cancel_task(self, task_id: str, reason: str) -> dict[str, Any]:
        """Request idempotent cancellation of one Biomni asynchronous task."""
        task_id = str(task_id or "").strip()
        reason = str(reason or "user requested stop").strip()
        if not task_id:
            raise ValueError("Biomni task_id is required")
        await self.initialize(load_catalog=False)
        tool = self._tools.get(self.CANCEL_TOOL_NAME)
        if tool is None:
            raise RuntimeError("Biomni MCP gateway does not expose cancel_biomni_task")
        response = await self._ainvoke_submission(
            tool,
            self.CANCEL_TOOL_NAME,
            {"task_id": task_id, "reason": reason},
        )
        payload = normalize_biomni_task_payload(parse_mcp_payload(response))
        if not isinstance(payload, dict):
            raise RuntimeError("Biomni cancellation returned a non-object payload")
        returned_task_id = str(payload.get("task_id") or "").strip()
        if returned_task_id and returned_task_id != task_id:
            raise RuntimeError("Biomni cancellation returned a different task_id")
        return payload

    async def _invoke_status_with_timeout(self, arguments: dict[str, Any]) -> Any:
        task_id = str(arguments.get("task_id") or "").strip()
        existing_poll = self._inflight_polls.get(task_id)
        if existing_poll is not None and not existing_poll.done():
            raise TimeoutError(
                f"Biomni status request for task {task_id or '<unknown>'} is still pending"
            )
        status_tool = self._tools.get(self.STATUS_TOOL_NAME)
        if status_tool is None:
            raise RuntimeError("Biomni MCP gateway does not expose get_biomni_task")
        poll_task = asyncio.ensure_future(
            self._ainvoke_submission(status_tool, self.STATUS_TOOL_NAME, arguments)
        )
        self._inflight_polls[task_id] = poll_task
        poll_task.add_done_callback(
            lambda completed: self._finish_poll(task_id, completed)
        )
        completed, response = await self._await_rpc(
            poll_task,
            timeout_seconds=self.poll_request_timeout_seconds,
        )
        if not completed:
            if self._inflight_polls.get(task_id) is poll_task:
                self._inflight_polls.pop(task_id, None)
            raise TimeoutError(
                f"Biomni status request for task {task_id or '<unknown>'} timed out after "
                f"{self.poll_request_timeout_seconds:g} seconds"
            )
        return response

    @staticmethod
    async def _await_rpc(
        rpc_task: asyncio.Future[Any],
        *,
        timeout_seconds: float,
    ) -> tuple[bool, Any | None]:
        """Bound one transport RPC and detach it promptly when it becomes stale."""
        try:
            completed, _ = await asyncio.wait(
                {rpc_task},
                timeout=max(0.0, float(timeout_seconds)),
            )
        except BaseException:
            if not rpc_task.done():
                rpc_task.cancel()
            raise
        if not completed:
            # Do not await cancellation here: some streamable HTTP clients can
            # block while unwinding a connection that has already timed out.
            rpc_task.cancel()
            await asyncio.sleep(0)
            return False, None
        return True, rpc_task.result()

    def _finish_submission(
        self,
        submission_key: tuple[str, str],
        completed: asyncio.Future[Any],
    ) -> None:
        if self._inflight_submissions.get(submission_key) is completed:
            self._inflight_submissions.pop(submission_key, None)
        if completed.cancelled():
            return
        try:
            completed.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _finish_poll(
        self,
        task_id: str,
        completed: asyncio.Future[Any],
    ) -> None:
        if self._inflight_polls.get(task_id) is completed:
            self._inflight_polls.pop(task_id, None)
        if completed.cancelled():
            return
        try:
            completed.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _submission_failure(
        self,
        *,
        gateway_tool_name: str,
        request_id: str,
        event: str,
        error: str,
        timeout_seconds: float | None = None,
    ) -> BiomniTaskResolution:
        metadata: dict[str, Any] = {
            "request_id": request_id,
            "gateway_tool": gateway_tool_name,
        }
        if any(
            marker in event
            for marker in (
                "timed_out",
                "timeout",
                "already_pending",
                "transport_error",
            )
        ):
            metadata["submission_unknown"] = True
            metadata["rpc_status"] = "timeout"
        if self.catalog_protocol != "legacy":
            metadata["capability_protocol"] = self.catalog_protocol
        trace: dict[str, Any] = {
            "event": event,
            "gateway_tool": gateway_tool_name,
            "request_id": request_id,
        }
        if timeout_seconds is not None:
            trace["timeout_seconds"] = timeout_seconds
        return BiomniTaskResolution(
            payload={"ok": False, "error": error},
            metadata=metadata,
            trace=[trace],
        )

    async def _load_tools(self) -> list[Any]:
        if self._tool_loader is not None:
            return await self._tool_loader(self.server_config)
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise RuntimeError(
                "langchain-mcp-adapters is required for a live Biomni connection"
            ) from exc
        self._client = MultiServerMCPClient({"biomni": self.server_config})
        return await self._client.get_tools()

    @staticmethod
    def _descriptor_from_record(record: dict[str, Any]) -> CapabilityDescriptor | None:
        manifest = CapabilityManifest.from_record(record)
        if manifest is None:
            return None
        provenance_required = (
            manifest.provenance_policy.get("required")
            if isinstance(manifest.provenance_policy.get("required"), bool)
            else None
        )
        return CapabilityDescriptor(
            canonical_name=manifest.canonical_name,
            description=manifest.description,
            aliases=manifest.aliases,
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            result_kind=manifest.result_kind,
            provenance_required=provenance_required,
            module=manifest.module,
            capability_version=manifest.capability_version,
            effect_contract=manifest.effect_contract,
            result_adapter=manifest.result_adapter,
            execution_mode=manifest.execution_mode,
            lifecycle=manifest.lifecycle,
            retry_policy=manifest.retry_policy,
            timeout_policy=manifest.timeout_policy,
            idempotency_policy=manifest.idempotency_policy,
            provenance_policy=manifest.provenance_policy,
            metadata=manifest.metadata,
        )

    def _normalize_matches(
        self,
        matches: list[Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in matches:
            if not isinstance(raw, dict):
                continue
            descriptor = self._descriptor_from_record(raw)
            if descriptor is None:
                continue
            canonical_name = self.canonicalize_tool_name(descriptor.canonical_name)
            cached = self._catalog_by_canonical.get(canonical_name)
            if cached is not None:
                descriptor = cached
            if canonical_name in seen:
                continue
            seen.add(canonical_name)
            score = raw.get("score")
            normalized.append(
                descriptor.to_record(
                    score=float(score) if isinstance(score, int | float) else None
                )
            )
            if len(normalized) >= limit:
                break
        return normalized

    def _rank_catalog(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self._embedding_cache is None:
            cache_path = os.getenv("OMNIAGENT_MCP_EMBEDDING_CACHE")
            if not cache_path:
                cache_path = str(
                    Path.home() / ".cache" / "omniagent" / "mcp_tool_embeddings.pt"
                )
            self._embedding_cache = QwenEmbeddingCache(cache_path=cache_path)
        documents = [self._resource_document(item) for item in self._catalog]
        query_text = f"Instruct: {self.RETRIEVAL_INSTRUCTION}\nQuery:{query}"
        query_vector = self._embedding_cache.embed_texts([query_text])[0]
        document_vectors = self._embedding_cache.ensure_documents(documents)
        ranked = sorted(
            (
                (sum(left * right for left, right in zip(query_vector, vector)), index)
                for index, vector in enumerate(document_vectors)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            self._catalog[index].to_record(score=score)
            for score, index in ranked[:limit]
        ]

    def _lexical_catalog_matches(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = {
            item for item in query.lower().replace("_", " ").split() if len(item) > 1
        }
        ranked = sorted(
            (
                (
                    len(
                        tokens.intersection(
                            set(
                                f"{descriptor.canonical_name} {descriptor.description}"
                                .lower()
                                .replace("_", " ")
                                .split()
                            )
                        )
                    ),
                    index,
                )
                for index, descriptor in enumerate(self._catalog)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            self._catalog[index].to_record(score=float(score))
            for score, index in ranked[:limit]
            if score > 0
        ]

    @staticmethod
    def _resource_document(descriptor: CapabilityDescriptor) -> str:
        properties = (
            descriptor.input_schema.get("properties", {})
            if isinstance(descriptor.input_schema, dict)
            else {}
        )
        inputs = "; ".join(
            f"{key}: {value.get('description', '') if isinstance(value, dict) else ''}"
            for key, value in properties.items()
        )[:600]
        document = (
            "Resource category: tools.\n"
            f"Resource name: {descriptor.canonical_name} "
            f"({descriptor.canonical_name.replace('_', ' ')}).\n"
            f"Description: {descriptor.description[:600]}"
        )
        return document + (f"\nInputs: {inputs}" if inputs else "")
