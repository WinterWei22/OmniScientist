from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_contract import (
    ArtifactContractError,
    normalize_artifact_declarations,
    verify_artifacts,
)
from .biomni_task_protocol import (
    begin_biomni_submission,
    parse_mcp_payload,
    poll_biomni_task,
    resolve_biomni_submission,
)
from .contracts import A1TaskRequest, A1TaskResult
from .execution_validation import validate_schema_instance
from .qwen_embedding_cache import QwenEmbeddingCache
from .result_adapter import ResultAdapterRegistry
from .roles import _invoke_json

if TYPE_CHECKING:
    from .biomni_gateway import BiomniGatewayClient


ToolLoader = Callable[[dict[str, Any]], Awaitable[list[Any]]]


class BiomniLayeredMCPTool:
    """Search and invoke Biomni tools without routing through Biomni A1."""

    SEARCH_TOOL_NAME = "biomni_search_tools"
    LIST_TOOL_NAME = "biomni_list_tools"
    CAPABILITIES_TOOL_NAME = "biomni_list_capabilities"
    INVOKE_TOOL_NAME = "biomni_invoke_tool"
    STATUS_TOOL_NAME = "get_biomni_task"
    FORBIDDEN_TOOL_NAME = "call_biomni"
    supports_blocking_workflow_tasks = False
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
        selector_model: Any,
        max_results: int = 8,
        tool_loader: ToolLoader | None = None,
        gateway: BiomniGatewayClient | None = None,
        task_poll_interval_seconds: float = 2.0,
        task_timeout_seconds: float = 900.0,
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
        self.selector_model = selector_model
        self.max_results = max(1, min(int(max_results), 20))
        self._tool_loader = tool_loader
        self.gateway = gateway
        self._client: Any = None
        self._search_tool: Any = None
        self._list_tool: Any = None
        self._capabilities_tool: Any = None
        self._invoke_tool: Any = None
        self._status_tool: Any = None
        self._embedding_cache: QwenEmbeddingCache | None = None
        self.result_adapters = ResultAdapterRegistry()
        self.task_poll_interval_seconds = max(0.0, float(task_poll_interval_seconds))
        self.task_timeout_seconds = max(0.0, float(task_timeout_seconds))
        self.discovered_tool_names: tuple[str, ...] = ()

    @property
    def exposed_tool_names(self) -> tuple[str, ...]:
        return (self.SEARCH_TOOL_NAME, self.INVOKE_TOOL_NAME)

    async def initialize(self) -> None:
        if self.gateway is not None:
            await self.gateway.initialize()
            self.discovered_tool_names = self.gateway.discovered_tool_names
            self._search_tool = self.gateway.get_tool(self.SEARCH_TOOL_NAME)
            self._list_tool = None
            self._capabilities_tool = None
            self._invoke_tool = self.gateway.get_tool(self.INVOKE_TOOL_NAME)
            self._status_tool = None
            return
        tools = await self._load_tools()
        tool_map = {getattr(tool, "name", ""): tool for tool in tools}
        self.discovered_tool_names = tuple(sorted(name for name in tool_map if name))
        self._search_tool = tool_map.get(self.SEARCH_TOOL_NAME)
        self._list_tool = tool_map.get(self.LIST_TOOL_NAME)
        self._capabilities_tool = tool_map.get(self.CAPABILITIES_TOOL_NAME)
        self._invoke_tool = tool_map.get(self.INVOKE_TOOL_NAME)
        self._status_tool = tool_map.get(self.STATUS_TOOL_NAME)
        missing = [
            name
            for name, tool in (
                (self.SEARCH_TOOL_NAME, self._search_tool),
                (self.INVOKE_TOOL_NAME, self._invoke_tool),
            )
            if tool is None
        ]
        if missing:
            raise RuntimeError(
                "Biomni layered MCP is missing required tool(s): " + ", ".join(missing)
            )

    async def run(
        self,
        request: A1TaskRequest,
        *,
        wait_for_terminal: bool = False,
    ) -> A1TaskResult:
        if self._search_tool is None or self._invoke_tool is None:
            raise RuntimeError(
                "BiomniLayeredMCPTool.initialize() must be called before run()"
            )

        query = self._build_search_query(request)
        trace: list[dict[str, Any]] = []
        if self.gateway is not None:
            configured_name = request.step.inputs.get("tool_name")
            if isinstance(configured_name, str) and configured_name.strip():
                inputs = dict(request.step.inputs)
                inputs["tool_name"] = self.gateway.canonicalize_tool_name(
                    configured_name
                )
                request = replace(request, step=replace(request.step, inputs=inputs))
        try:
            search_payload = await self._search(query)
            matches = self._extract_matches(search_payload)
            trace.append(
                {
                    "event": "biomni_tool_search",
                    "gateway_tool": search_payload.get(
                        "gateway_tool", self.SEARCH_TOOL_NAME
                    ),
                    "retrieval": search_payload.get("retrieval", "lexical"),
                    "query": query,
                    "catalog_size": (
                        search_payload.get("catalog_size")
                        if isinstance(search_payload, dict)
                        else None
                    ),
                    "candidates": [
                        {
                            "qualified_name": item.get("qualified_name"),
                            "score": item.get("score"),
                        }
                        for item in matches
                    ],
                }
            )
            if not matches:
                return A1TaskResult(
                    success=False,
                    answer="No Biomni internal tool matched the experiment step.",
                    tool_trace=trace,
                    errors=[f"No tool candidates for query: {query}"],
                    raw=self._compact_value(search_payload, 2000),
                )

            selection = await self._select_tool(request, matches)
            tool_name = str(selection.get("tool_name", "")).strip()
            if self.gateway is not None:
                tool_name = self.gateway.canonicalize_tool_name(tool_name)
            arguments = selection.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Resource selector returned non-object arguments")
            allowed_names = {
                str(item.get("qualified_name") or item.get("name")) for item in matches
            }
            if tool_name == self.FORBIDDEN_TOOL_NAME:
                raise ValueError("Agent selector attempted to call forbidden call_biomni")
            if tool_name not in allowed_names:
                raise ValueError(
                    f"Resource selector chose a tool outside retrieved candidates: {tool_name}"
                )
            argument_error = self._validate_arguments(tool_name, arguments, matches)
            if argument_error:
                trace.append(
                    {
                        "event": "biomni_arguments_rejected",
                        "tool_name": tool_name,
                        "error": argument_error,
                    }
                )
                return A1TaskResult(
                    success=False,
                    result_status="invalid_arguments",
                    answer="MCP tool arguments were rejected before invocation.",
                    tool_trace=trace,
                    errors=[f"invalid_arguments: {argument_error}"],
                )
            trace.append(
                {
                    "event": "biomni_tool_selected",
                    "selector": str(selection.get("selector", "omniagent_qwen")),
                    "tool_name": tool_name,
                    "arguments": self._compact_value(arguments, 1200),
                    "rationale": self._compact_text(selection.get("rationale", ""), 1200),
                }
            )

            if self.gateway is not None:
                resolution = await self.gateway.invoke_tool(
                    tool_name,
                    arguments,
                    request_id=(
                        request.request_id
                        or self.gateway.request_id_for(
                            run_id=request.run_id,
                            step_id=request.step.step_id,
                        )
                    ),
                    wait_for_terminal=wait_for_terminal,
                )
            else:
                raw_invoke = await self._invoke_tool.ainvoke(
                    {"tool_name": tool_name, "arguments": arguments}
                )
                invoke_payload = self._parse_payload(raw_invoke)
                resolution = (
                    await resolve_biomni_submission(
                        invoke_payload,
                        status_tool=self._status_tool,
                        parse_payload=self._parse_payload,
                        poll_interval_seconds=self.task_poll_interval_seconds,
                        timeout_seconds=self.task_timeout_seconds,
                    )
                    if wait_for_terminal
                    else begin_biomni_submission(invoke_payload)
                )
            invoke_payload = resolution.payload
            trace.extend(resolution.trace)
            inner_result = (
                invoke_payload.get("result")
                if isinstance(invoke_payload, dict)
                else None
            )
            trace.append(
                {
                    "event": "biomni_tool_invoked",
                    "gateway_tool": self.INVOKE_TOOL_NAME,
                    "tool_name": tool_name,
                    "ok": (
                        invoke_payload.get("ok")
                        if isinstance(invoke_payload, dict)
                        else True
                    ),
                    "tool_result_success": (
                        inner_result.get("success")
                        if isinstance(inner_result, dict)
                        else None
                    ),
                }
            )
            candidate = next(
                (
                    item
                    for item in matches
                    if str(item.get("qualified_name") or item.get("name")) == tool_name
                ),
                None,
            )
            adapter_name = str(candidate.get("result_adapter", "generic")) if candidate else "generic"
            result = self._to_result(
                invoke_payload,
                trace,
                adapter_registry=self.result_adapters,
                adapter_name=adapter_name,
            )
            result.task_metadata = dict(resolution.metadata or {}) | result.task_metadata
            if resolution.pending:
                result.success = False
                result.result_status = "task_pending"
                return result
            self._validate_artifacts(
                result, request.allowed_paths, raise_on_error=False
            )
            output_schema = candidate.get("output_schema", {}) if candidate else {}
            body = result.verification_payload
            output_errors = validate_schema_instance(body, output_schema)
            if output_errors:
                result.success = False
                result.result_status = "invalid_tool_output"
                result.errors.extend(output_errors[:3])
            result.task_metadata.update(dict(resolution.metadata or {}))
            result.task_metadata["output_schema_errors"] = output_errors
            return result
        except Exception as exc:
            return A1TaskResult(
                success=False,
                answer="Biomni layered MCP execution failed.",
                tool_trace=trace,
                errors=[f"{type(exc).__name__}: {exc}"],
            )

    async def invoke_bound_call(
        self,
        request: A1TaskRequest,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        wait_for_terminal: bool = False,
    ) -> A1TaskResult:
        """Invoke a previously catalog-validated call without another semantic search."""
        if self.gateway is None:
            raise RuntimeError("Bound MCP calls require the shared Biomni gateway")
        if self._invoke_tool is None:
            raise RuntimeError(
                "BiomniLayeredMCPTool.initialize() must be called before invoke_bound_call()"
            )
        canonical_name = self.gateway.canonicalize_tool_name(tool_name)
        if canonical_name == self.FORBIDDEN_TOOL_NAME:
            return A1TaskResult(
                success=False,
                result_status="invalid_bound_call",
                errors=["Bound MCP calls must not invoke call_biomni"],
            )
        descriptor = self.gateway.get_capability_descriptor(canonical_name)
        if descriptor is None:
            return A1TaskResult(
                success=False,
                result_status="capability_unavailable",
                errors=[
                    "Bound MCP capability is absent from the cached catalog: "
                    f"{canonical_name}"
                ],
            )
        expected_input_schema = input_schema or descriptor.input_schema
        argument_errors = validate_schema_instance(
            arguments, expected_input_schema, strict_objects=True
        )
        if argument_errors:
            return A1TaskResult(
                success=False,
                result_status="invalid_arguments",
                errors=argument_errors[:3],
                tool_trace=[
                    {
                        "event": "biomni_bound_arguments_rejected",
                        "tool_name": canonical_name,
                    }
                ],
            )

        trace: list[dict[str, Any]] = [
            {
                "event": "biomni_bound_tool_invoked",
                "gateway_tool": self.INVOKE_TOOL_NAME,
                "tool_name": canonical_name,
                "catalog_revision": self.gateway.catalog_revision or None,
            }
        ]
        try:
            resolution = await self.gateway.invoke_tool(
                canonical_name,
                arguments,
                request_id=(
                    request.request_id
                    or self.gateway.request_id_for(
                        run_id=request.run_id,
                        step_id=request.step.step_id,
                    )
                ),
                wait_for_terminal=wait_for_terminal,
            )
            trace.extend(resolution.trace)
            result = self._to_result(
                resolution.payload,
                trace,
                adapter_registry=self.result_adapters,
                adapter_name=descriptor.result_adapter,
            )
            result.task_metadata = dict(resolution.metadata or {}) | result.task_metadata
            if resolution.pending:
                result.success = False
                result.result_status = "task_pending"
                return result
            self._validate_artifacts(
                result, request.allowed_paths, raise_on_error=False
            )
            body = result.verification_payload
            expected_output_schema = output_schema or descriptor.output_schema
            output_errors = validate_schema_instance(body, expected_output_schema)
            if output_errors:
                result.success = False
                result.result_status = "invalid_tool_output"
                result.errors.extend(output_errors[:3])
            result.task_metadata["output_schema_errors"] = output_errors
            return result
        except Exception as exc:
            return A1TaskResult(
                success=False,
                result_status="execution_transport_error",
                errors=[f"{type(exc).__name__}: {exc}"],
                tool_trace=trace,
                task_metadata={
                    "request_id": request.request_id,
                    "submission_unknown": True,
                },
            )

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        """Poll a submitted structured MCP task without invoking it again."""
        if self.gateway is not None:
            resolution = await self.gateway.poll_task(task_metadata)
        else:
            resolution = await poll_biomni_task(
                task_metadata,
                status_tool=self._status_tool,
                parse_payload=self._parse_payload,
            )
        route = task_metadata.get("omniagent_route", {})
        bound_call = route.get("bound_call", {}) if isinstance(route, dict) else {}
        adapter_name = (
            str(bound_call.get("result_adapter", "generic"))
            if isinstance(bound_call, dict)
            else "generic"
        )
        result = self._to_result(
            resolution.payload,
            resolution.trace,
            adapter_registry=self.result_adapters,
            adapter_name=adapter_name,
        )
        result.task_metadata = dict(resolution.metadata or {}) | result.task_metadata
        if resolution.poll_error:
            result.task_metadata["last_poll_error"] = resolution.poll_error
        if resolution.pending:
            result.success = False
            result.result_status = "task_pending"
            return result
        self._validate_artifacts(
            result, request.allowed_paths, raise_on_error=False
        )
        return result

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

    async def _search(self, query: str) -> dict[str, Any]:
        if self.gateway is not None:
            return await self.gateway.search_capabilities(
                query,
                max_results=self.max_results,
            )
        catalog_tool = self._capabilities_tool or self._list_tool
        if catalog_tool is not None:
            try:
                catalog_arguments = (
                    {}
                    if self._capabilities_tool is not None
                    else {"include_schema": True, "module_prefix": None}
                )
                raw_catalog = await catalog_tool.ainvoke(catalog_arguments)
                catalog_payload = self._parse_payload(raw_catalog)
                catalog = (
                    catalog_payload.get("tools", [])
                    if isinstance(catalog_payload, dict)
                    else []
                )
                if catalog:
                    matches = await asyncio.to_thread(
                        self._rank_catalog, query, catalog
                    )
                    if matches:
                        return {
                            "query": query,
                            "catalog_size": len(catalog),
                            "matches": matches,
                            "retrieval": "qwen3.7-text-embedding-cache",
                            "gateway_tool": (
                                self.CAPABILITIES_TOOL_NAME
                                if self._capabilities_tool is not None
                                else self.LIST_TOOL_NAME
                            ),
                        }
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
            else:
                semantic_error = "semantic catalog retrieval returned no matches"
        else:
            semantic_error = "biomni_list_capabilities and biomni_list_tools are unavailable"

        raw_search = await self._search_tool.ainvoke(
            {"query": query, "max_results": self.max_results}
        )
        payload = self._parse_payload(raw_search)
        if isinstance(payload, dict):
            payload.setdefault("retrieval", "mcp_lexical_fallback")
            payload.setdefault("gateway_tool", self.SEARCH_TOOL_NAME)
            payload["semantic_error"] = semantic_error
        return payload

    async def search_capabilities(self, query: str) -> dict[str, Any]:
        return await self._search(query)

    async def parameterize_schema_bound_request(
        self,
        request: A1TaskRequest,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask Qwen to fill parameters only after candidate schemas are disclosed."""
        matches = [item for item in candidates if isinstance(item, dict)]
        if not matches:
            raise ValueError("Schema parameterization requires retrieved candidates")

        # Planner-authored internal tool names and argument objects are not trusted.
        # The parameterizer receives only the discovered candidate set below.
        selector_inputs = dict(request.step.inputs)
        selector_inputs.pop("tool_name", None)
        selector_inputs.pop("arguments", None)
        selector_request = replace(
            request,
            step=replace(request.step, inputs=selector_inputs),
        )
        selection = await self._select_tool(selector_request, matches)
        tool_name = str(selection.get("tool_name", "")).strip()
        if self.gateway is not None:
            tool_name = self.gateway.canonicalize_tool_name(tool_name)
        if not tool_name:
            raise ValueError(
                "No retrieved MCP candidate was selected for the semantic intent"
            )
        arguments = selection.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Resource selector returned non-object arguments")
        candidate = next(
            (
                item
                for item in matches
                if str(item.get("qualified_name") or item.get("name")) == tool_name
            ),
            None,
        )
        if candidate is None:
            raise ValueError(
                f"Resource selector chose a tool outside retrieved candidates: {tool_name}"
            )
        argument_error = self._validate_arguments(tool_name, arguments, matches)
        if argument_error:
            raise ValueError(argument_error)
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "rationale": self._compact_text(selection.get("rationale", ""), 1200),
            "selector": str(selection.get("selector", "omniagent_qwen")),
            "candidate": candidate,
        }

    def _rank_catalog(
        self,
        query: str,
        catalog: list[Any],
    ) -> list[dict[str, Any]]:
        resources = [item for item in catalog if isinstance(item, dict)]
        if not resources:
            return []
        if self._embedding_cache is None:
            cache_path = os.getenv("OMNIAGENT_MCP_EMBEDDING_CACHE")
            if not cache_path:
                cache_path = str(
                    Path.home() / ".cache" / "omniagent" / "mcp_tool_embeddings.pt"
                )
            self._embedding_cache = QwenEmbeddingCache(cache_path=cache_path)

        documents = [
            self._resource_document(resource, index)
            for index, resource in enumerate(resources)
        ]
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
            dict(resources[index]) | {"score": round(float(score), 6)}
            for score, index in ranked[: self.max_results]
        ]

    @staticmethod
    def _resource_document(resource: dict[str, Any], index: int) -> str:
        name = str(resource.get("name", f"Resource {index}"))
        description = str(resource.get("description", ""))[:600]
        schema = resource.get("input_schema", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        inputs = "; ".join(
            f"{key}: {value.get('description', '') if isinstance(value, dict) else ''}"
            for key, value in properties.items()
        )[:600]
        document = (
            "Resource category: tools.\n"
            f"Resource name: {name} ({name.replace('_', ' ')}).\n"
            f"Description: {description}"
        )
        return document + (f"\nInputs: {inputs}" if inputs else "")

    def _build_search_query(self, request: A1TaskRequest) -> str:
        configured = request.step.inputs.get("tool_query")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()[:1200]
        parts = [request.step.objective]
        values = request.step.inputs.get("value")
        if isinstance(values, list):
            parts.extend(str(item) for item in values[:10])
        parts.extend(request.step.expected_outputs[:5])
        return " ".join(part for part in parts if part).strip()[:1200]

    async def _select_tool(
        self,
        request: A1TaskRequest,
        matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        exact_tool_name = str(request.step.inputs.get("tool_name", "")).strip()
        if exact_tool_name:
            arguments = request.step.inputs.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Schema-bound tool arguments must be an object")
            allowed_names = {
                str(item.get("qualified_name") or item.get("name"))
                for item in matches
            }
            if exact_tool_name not in allowed_names:
                raise ValueError(
                    f"Schema-bound tool is not among retrieved candidates: {exact_tool_name}"
                )
            return {
                "tool_name": exact_tool_name,
                "arguments": arguments,
                "rationale": "OmniAgent supplied a schema-bound tool and arguments.",
                "selector": "schema_bound",
            }

        candidates = [
            {
                "name": item.get("name"),
                "qualified_name": item.get("qualified_name"),
                "module": item.get("module"),
                "description": item.get("description"),
                "input_schema": item.get("input_schema", {}),
                "output_schema": item.get("output_schema", {}),
                "effect_contract": item.get("effect_contract", {}),
                "result_adapter": item.get("result_adapter", "generic"),
                "execution_mode": item.get("execution_mode", "sync"),
                "capability_version": item.get("capability_version", ""),
                "score": item.get("score"),
            }
            for item in matches
        ]
        prompt = (
            "You are OmniAgent's Biomni Resource Selector. Select exactly one retrieved "
            "Biomni internal Python tool for the bounded experiment step and construct "
            "only arguments supported by its input schema. You must select a "
            "qualified_name from CANDIDATES. Never select call_biomni. If none directly "
            "supports the requested capability, return an empty tool_name and explain why. "
            "Relative input "
            "files are located under the first allowed path. Do not invent file contents "
            "or unavailable evidence. Return JSON with tool_name, arguments, and "
            "rationale.\n\nREQUEST:\n"
            + json.dumps(
                {
                    "research_goal": request.research_goal,
                    "iteration": request.iteration,
                    "experiment_step": asdict(request.step),
                    "allowed_paths": request.allowed_paths,
                    "prior_observations": request.prior_observations[-8:],
                    "prior_evaluations": request.prior_evaluations[-3:],
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n\nCANDIDATES:\n"
            + json.dumps(candidates, ensure_ascii=False, default=str)
        )
        return await _invoke_json(self.selector_model, prompt)

    @staticmethod
    def _validate_arguments(
        tool_name: str,
        arguments: dict[str, Any],
        matches: list[dict[str, Any]],
    ) -> str | None:
        candidate = next(
            (
                item
                for item in matches
                if str(item.get("qualified_name") or item.get("name")) == tool_name
            ),
            None,
        )
        schema = candidate.get("input_schema", {}) if candidate else {}
        if not isinstance(schema, dict):
            return None
        errors = validate_schema_instance(arguments, schema, strict_objects=True)
        return "; ".join(errors[:3]) if errors else None

    @classmethod
    def _to_result(
        cls,
        payload: Any,
        trace: list[dict[str, Any]],
        _allowed_paths: list[str] | None = None,
        *,
        adapter_registry: ResultAdapterRegistry | None = None,
        adapter_name: str = "generic",
    ) -> A1TaskResult:
        if isinstance(payload, dict) and payload.get("pending") is True:
            return A1TaskResult(
                success=False,
                result_status="task_pending",
                tool_trace=trace,
                raw=cls._compact_value(payload, 2000),
            )
        normalized = (adapter_registry or ResultAdapterRegistry()).adapt(
            payload, adapter_name
        )
        if not normalized.ok:
            error = normalized.error or "Biomni internal tool returned ok=false"
            return A1TaskResult(
                success=False,
                result_status=(
                    "output_contract_failed"
                    if isinstance(payload, str)
                    else normalized.status or "error"
                ),
                answer=cls._compact_text(payload, 4000),
                tool_trace=trace,
                errors=[str(error)],
                raw=cls._compact_value(payload, 2000),
            )
        body = normalized.body
        if isinstance(body, str):
            return A1TaskResult(
                success=False,
                result_status="output_contract_failed",
                answer=cls._compact_text(body, 4000),
                tool_trace=trace,
                errors=[
                    "OUTPUT_CONTRACT_FAILED: MCP returned text without structured data"
                ],
                raw=cls._compact_value(payload, 2000),
            )
        metrics: dict[str, float] = {}
        artifacts: list[str] = cls._artifact_paths(
            normalize_artifact_declarations(normalized.artifacts)
        )
        artifact_declarations: list[dict[str, Any]] = []
        errors: list[str] = []
        inner_success = True
        result_status = "success"
        observations: list[dict[str, Any]] = []
        empty_result = bool(normalized.metadata.get("empty_result"))
        if isinstance(body, dict):
            inner_success = bool(body.get("success", True))
            if not inner_success:
                error = (
                    body.get("error")
                    or body.get("message")
                    or "Biomni tool reported success=false"
                )
                errors.append(str(error))
                result_status = "error"
            nested_result = body.get("result")
            empty_result = empty_result or (
                isinstance(nested_result, dict)
                and nested_result.get("raw_text") == ""
            )
            if isinstance(nested_result, dict):
                result_set = nested_result.get("result_set")
                empty_result = empty_result or (
                    nested_result.get("total_count") == 0
                    or isinstance(result_set, list) and not result_set
                )
            if inner_success and empty_result:
                result_status = "data_empty"
                observations.append(
                    {
                        "status": "data_empty",
                        "result_count": 0,
                        "query_info": cls._compact_value(
                            body.get("query_info", {}), 1200
                        ),
                    }
                )
            metrics = {
                str(key): float(value)
                for key, value in body.get("metrics", {}).items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            if result_status == "data_empty":
                metrics["result_count"] = 0.0
            body_artifacts = body.get("artifacts", [])
            artifact_declarations = normalize_artifact_declarations(
                [*artifacts, *normalize_artifact_declarations(body_artifacts)]
            )
            artifacts = cls._artifact_paths(artifact_declarations)
            material_keys = set(body) - {
                "success",
                "ok",
                "status",
                "message",
                "error",
            }
            if inner_success and result_status == "success" and not (
                material_keys or metrics or artifacts
            ):
                result_status = "output_contract_failed"
                errors.append(
                    "OUTPUT_CONTRACT_FAILED: MCP returned no structured result fields"
                )
        answer = (
            json.dumps(
                {
                    "status": "NO_MATCH",
                    "result_count": 0,
                    "query_info": body.get("query_info", {})
                    if isinstance(body, dict)
                    else {},
                },
                ensure_ascii=False,
            )
            if result_status == "data_empty"
            else cls._compact_text(body, 6000)
        )
        result = A1TaskResult(
            success=inner_success and result_status in {"success", "data_empty"},
            result_status=result_status,
            answer=answer,
            tool_trace=trace,
            observations=observations or [{"tool_result": cls._compact_value(body, 3000)}],
            metrics=metrics,
            artifacts=artifacts,
            errors=errors,
            raw=cls._compact_value(payload, 2000),
            verification_payload=body,
            task_metadata=(
                {
                    "artifact_declarations": artifact_declarations,
                    "result_adapter": adapter_name or "generic",
                    "provenance": list(normalized.provenance),
                    **normalized.metadata,
                }
                if artifact_declarations or normalized.provenance or normalized.metadata
                else {"result_adapter": adapter_name or "generic"}
            ),
        )
        return result

    @staticmethod
    def _extract_matches(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        matches = payload.get("matches", [])
        return [item for item in matches if isinstance(item, dict)]

    @classmethod
    def _parse_payload(cls, response: Any) -> Any:
        return parse_mcp_payload(response)

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str
        )
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated " + str(len(text) - limit) + " chars]"

    @classmethod
    def _compact_value(cls, value: Any, text_limit: int) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._compact_value(item, text_limit)
                for key, item in list(value.items())[:30]
            }
        if isinstance(value, list | tuple):
            items = list(value)
            selected = [cls._compact_value(item, text_limit) for item in items[:30]]
            if len(items) > 30:
                selected.append({"truncated_items": len(items) - 30})
            return selected
        if isinstance(value, str):
            return cls._compact_text(value, text_limit)
        return value

    @staticmethod
    def _artifact_paths(value: Any) -> list[str]:
        items = value if isinstance(value, list | tuple) else [value]
        paths: list[str] = []
        for item in items[:20]:
            if isinstance(item, dict):
                item = item.get("path") or item.get("uri") or ""
            path = str(item).strip()
            if path:
                paths.append(path)
        return paths

    @staticmethod
    def _validate_artifacts(
        result: A1TaskResult,
        allowed_paths: list[str],
        *,
        raise_on_error: bool = True,
    ) -> None:
        if not result.artifacts:
            return
        declarations = result.task_metadata.get(
            "artifact_declarations", result.artifacts
        )
        try:
            verified = verify_artifacts(declarations, allowed_paths)
        except ArtifactContractError as exc:
            if raise_on_error:
                raise ValueError(str(exc)) from exc
            result.success = False
            result.result_status = "artifact_contract_failed"
            result.errors.append(str(exc))
            result.task_metadata["artifact_validation_error"] = str(exc)
            result.artifacts = []
            return
        result.artifacts = [item.path for item in verified]
        result.task_metadata["artifact_manifest"] = [item.to_dict() for item in verified]
