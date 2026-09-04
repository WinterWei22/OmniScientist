from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_contract import (
    ArtifactContractError,
    normalize_artifact_declarations,
    verify_artifacts,
)
from .biomni_task_protocol import (
    BiomniTaskResolution,
    begin_biomni_submission,
    poll_biomni_task,
)
from .contracts import A1TaskRequest, A1TaskResult

if TYPE_CHECKING:
    from .biomni_gateway import BiomniGatewayClient

ToolLoader = Callable[[dict[str, Any]], Awaitable[list[Any]]]


class BiomniA1Tool:
    """Expose exactly one coarse-grained Biomni capability to OmniAgent."""

    TOOL_NAME = "call_biomni"
    STATUS_TOOL_NAME = "get_biomni_task"

    def __init__(
        self,
        server_config: dict[str, Any],
        *,
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
        self._tool_loader = tool_loader
        self.gateway = gateway
        self._client: Any = None
        self._tool: Any = None
        self._status_tool: Any = None
        self.task_poll_interval_seconds = max(0.0, float(task_poll_interval_seconds))
        self.task_timeout_seconds = max(0.0, float(task_timeout_seconds))
        self.discovered_tool_names: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls) -> BiomniA1Tool:
        from local_deep_research.config import settings

        if not settings.biomni_mcp.enabled:
            raise RuntimeError("Biomni MCP is disabled in configuration")
        return cls(
            {
                "transport": settings.biomni_mcp.transport,
                "url": settings.biomni_mcp.url,
            }
        )

    @property
    def exposed_tool_names(self) -> tuple[str, ...]:
        return (self.TOOL_NAME,)

    async def initialize(self, *, load_catalog: bool = True) -> None:
        if self.gateway is not None:
            await self.gateway.initialize(load_catalog=load_catalog)
            self.discovered_tool_names = self.gateway.discovered_tool_names
            self._tool = self.gateway.get_tool(self.TOOL_NAME)
            self._status_tool = None
            return
        tools = await self._load_tools()
        tool_map = {getattr(tool, "name", ""): tool for tool in tools}
        self.discovered_tool_names = tuple(sorted(name for name in tool_map if name))
        self._tool = tool_map.get(self.TOOL_NAME)
        self._status_tool = tool_map.get(self.STATUS_TOOL_NAME)
        if self._tool is None:
            raise RuntimeError(
                "Biomni Host does not expose call_biomni; the Agent layer will not "
                "fall back to atomic MCP tools"
            )

    async def run(self, request: A1TaskRequest) -> A1TaskResult:
        if self._tool is None:
            raise RuntimeError("BiomniA1Tool.initialize() must be called before run()")

        attempts = 3
        base_delay_seconds = 45.0
        last_result: A1TaskResult | None = None
        request_id = request.request_id or str(
            request.step.inputs.get("request_id", "")
        ).strip()
        if not request_id and self.gateway is not None:
            request_id = self.gateway.request_id_for(
                run_id=request.run_id,
                step_id=request.step.step_id,
            )
        result_contract = self._result_contract(request)
        for attempt in range(attempts):
            try:
                if self.gateway is not None:
                    resolution = await self.gateway.call_a1(
                        request.to_prompt(),
                        request_id=request_id,
                        result_contract=result_contract,
                    )
                else:
                    arguments: dict[str, Any] = {"prompt": request.to_prompt()}
                    if result_contract and self._tool_accepts_argument(
                        self._tool, "result_contract"
                    ):
                        arguments["result_contract"] = result_contract
                    response = await self._tool.ainvoke(
                        arguments
                    )
                    payload = self._parse_payload(response)
                    resolution = begin_biomni_submission(payload)
                result = self._result_from_resolution(
                    resolution,
                    request.allowed_paths,
                    result_contract=result_contract,
                )
            except Exception as exc:
                result = A1TaskResult(
                    success=False,
                    result_status="execution_transport_error",
                    errors=[f"{type(exc).__name__}: {exc}"],
                    task_metadata=(
                        {
                            "request_id": request_id,
                            "submission_unknown": True,
                        }
                        if self.gateway is not None
                        else {}
                    ),
                )

            last_result = result
            if result.task_metadata and not result.task_metadata.get("submission_unknown"):
                return result
            if result.task_metadata.get("idempotency_conflict"):
                return result
            if result.task_metadata.get("submission_unknown"):
                if attempt < attempts - 1:
                    await asyncio.sleep(min(5.0, float(attempt + 1)))
                    continue
                return result
            if not self._is_rate_limited_result(result) or attempt >= attempts - 1:
                return result
            await asyncio.sleep(base_delay_seconds * (attempt + 1))

        return last_result or A1TaskResult(success=False, errors=["Biomni A1 call did not return"])

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        """Poll a durable task identity without re-submitting the A1 request."""
        if self.gateway is not None:
            resolution = await self.gateway.poll_task(task_metadata)
        else:
            resolution = await poll_biomni_task(
                task_metadata,
                status_tool=self._status_tool,
                parse_payload=self._parse_payload,
            )
        return self._result_from_resolution(
            resolution,
            request.allowed_paths,
            result_contract=(
                task_metadata.get("biomni_result_contract")
                if isinstance(task_metadata.get("biomni_result_contract"), dict)
                else (
                    # Preserve the contract for tasks created by the previous
                    # release; it is metadata for parsing a pending task, not
                    # a reason to submit another request.
                    task_metadata.get("result_contract")
                    if isinstance(task_metadata.get("result_contract"), dict)
                    else self._result_contract(request)
                )
            ),
        )

    def _result_from_resolution(
        self,
        resolution: BiomniTaskResolution,
        allowed_paths: list[str],
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> A1TaskResult:
        result = self._to_result(
            resolution.payload,
            result_contract=result_contract,
        )
        result.task_metadata = dict(resolution.metadata) | result.task_metadata
        if result_contract:
            result.task_metadata["biomni_result_contract"] = dict(result_contract)
        if resolution.poll_error:
            result.task_metadata["last_poll_error"] = resolution.poll_error
        result.tool_trace = self._compact_trace(resolution.trace + result.tool_trace)
        if result.verification_payload is None and result.output is not None:
            # Downstream verifiers consume this provider-neutral payload rather
            # than having to know the Biomni result.data nesting.
            result.verification_payload = result.output
        if resolution.pending:
            result.success = False
            result.result_status = "task_pending"
            return result
        self._validate_artifacts(result, allowed_paths, raise_on_error=False)
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

    @classmethod
    def _parse_payload(cls, response: Any) -> Any:
        structured = getattr(response, "structuredContent", None)
        if structured:
            return structured
        content = getattr(response, "content", None)
        if content and len(content) == 1 and hasattr(content[0], "text"):
            response = content[0].text
        elif isinstance(response, list) and len(response) == 1:
            item = response[0]
            if hasattr(item, "text"):
                response = item.text
            elif isinstance(item, dict) and item.get("type") == "text":
                response = item.get("text")
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"ok": True, "result": response}
        return response

    @classmethod
    def _to_result(
        cls,
        payload: Any,
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> A1TaskResult:
        compact_raw = cls._compact_raw(payload)
        if isinstance(payload, dict) and payload.get("pending") is True:
            return A1TaskResult(
                success=False,
                result_status="task_pending",
                raw=compact_raw,
            )
        if not isinstance(payload, dict):
            answer = cls._compact_text(payload, 6000)
            return A1TaskResult(
                success=False,
                result_status="output_contract_failed",
                answer=answer,
                errors=["OUTPUT_CONTRACT_FAILED: Biomni A1 returned text without structured output"],
                raw=compact_raw,
            )
        if payload.get("ok") is False:
            error = payload.get("error") or "Biomni A1 returned ok=false"
            return A1TaskResult(
                success=False,
                result_status="error",
                errors=[str(error)],
                tool_trace=cls._compact_trace(
                    cls._normalise_trace(payload.get("trace", []))
                ),
                raw=compact_raw,
            )

        body = payload.get("result", payload)
        if isinstance(body, list | tuple) and len(body) >= 2:
            raw_answer = cls._answer_text(body[-1])
            output = cls._extract_output(raw_answer, result_contract=result_contract)
            answer = cls._compact_text(raw_answer, 6000)
            return A1TaskResult(
                success=output is not None,
                result_status="success" if output is not None else "output_contract_failed",
                answer=answer,
                output=output,
                tool_trace=cls._compact_trace(cls._normalise_trace(body[0])),
                errors=(
                    []
                    if output is not None
                    else ["OUTPUT_CONTRACT_FAILED: Biomni A1 returned text without structured output"]
                ),
                raw=compact_raw,
            )
        if isinstance(body, dict):
            raw_answer = cls._answer_text(
                body.get("answer", body.get("content", ""))
            )
            embedded = cls._answer_result_envelope(raw_answer)
            normalized_body = cls._merge_result_envelope(
                body,
                embedded,
                result_contract=result_contract,
            )
            explicit_success = normalized_body.get("success")
            answer = cls._compact_text(raw_answer, 6000)
            artifact_declarations = normalize_artifact_declarations(
                normalized_body.get("artifacts", [])
            )
            artifacts = cls._artifact_paths(artifact_declarations)
            output = cls._canonical_output(
                normalized_body,
                raw_answer,
                result_contract=result_contract,
            )
            has_structured_result = bool(
                output is not None
                or normalized_body.get("observations")
                or normalized_body.get("metrics")
                or artifacts
            )
            if explicit_success is False:
                result_status = "error"
            elif not has_structured_result:
                result_status = "output_contract_failed"
            else:
                result_status = "success"
            metrics = {
                str(key): float(value)
                for key, value in normalized_body.get("metrics", {}).items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            method_provenance = cls._normalise_records(
                normalized_body.get("method_provenance", [])
            )
            metadata: dict[str, Any] = {}
            if artifact_declarations:
                metadata["artifact_declarations"] = artifact_declarations
            if method_provenance:
                metadata["method_provenance"] = method_provenance
            return A1TaskResult(
                success=explicit_success is not False and result_status == "success",
                result_status=result_status,
                answer=answer,
                output=output,
                tool_trace=cls._compact_trace(
                    cls._normalise_trace(normalized_body.get("tool_trace", []))
                ),
                observations=cls._normalise_records(
                    normalized_body.get("observations", [])
                ),
                metrics=metrics,
                artifacts=artifacts,
                task_metadata=metadata,
                errors=(
                    [str(item) for item in normalized_body.get("errors", [])][:8]
                    or (["OUTPUT_CONTRACT_FAILED: Biomni A1 returned no structured output"]
                        if result_status == "output_contract_failed" else [])
                ),
                raw=compact_raw,
            )
        raw_answer = cls._answer_text(body)
        output = cls._extract_output(raw_answer, result_contract=result_contract)
        answer = cls._compact_text(raw_answer, 6000)
        return A1TaskResult(
            success=output is not None,
            result_status="success" if output is not None else "output_contract_failed",
            answer=answer,
            output=output,
            errors=(
                []
                if output is not None
                else ["OUTPUT_CONTRACT_FAILED: Biomni A1 returned text without structured output"]
            ),
            raw=compact_raw,
        )

    @classmethod
    def _canonical_output(
        cls,
        body: dict[str, Any],
        answer: str,
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> Any:
        """Extract only the result payload from known Biomni A1 envelopes."""
        for key in ("output", "final_output", "submission"):
            if key in body and body[key] is not None:
                return body[key]
        nested = body.get("result")
        if isinstance(nested, dict):
            nested_answer = cls._compact_text(
                nested.get("answer", nested.get("content", answer)), 6000
            )
            nested_output = cls._canonical_output(
                nested,
                nested_answer,
                result_contract=result_contract,
            )
            if nested_output is not None:
                return nested_output
            if cls._looks_like_output(nested, result_contract=result_contract):
                return nested
        if cls._looks_like_output(body, result_contract=result_contract):
            return body
        return cls._extract_output(answer, result_contract=result_contract)

    @classmethod
    def _merge_result_envelope(
        cls,
        outer: dict[str, Any],
        embedded: dict[str, Any] | None,
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge a JSON result envelope transported inside Biomni's answer field."""
        if embedded is None:
            return outer
        merged = dict(embedded)
        structured_fields = (
            "observations",
            "metrics",
            "artifacts",
            "method_provenance",
            "tool_trace",
            "errors",
        )
        for key in structured_fields:
            if outer.get(key) not in (None, "", [], {}):
                merged[key] = outer[key]
        for key in ("success", "status", "execution_status", "contract_status"):
            if key in outer and outer[key] is not None:
                merged[key] = outer[key]

        outer_output = cls._canonical_output(
            outer,
            "",
            result_contract=result_contract,
        )
        embedded_output = cls._canonical_output(
            embedded,
            "",
            result_contract=result_contract,
        )
        if outer_output is not None and not (
            embedded_output is not None and cls._is_text_only_output(outer_output)
        ):
            merged["output"] = outer_output
        elif embedded_output is not None:
            merged["output"] = embedded_output
        return merged

    @classmethod
    def _answer_result_envelope(cls, answer: str) -> dict[str, Any] | None:
        value = cls._decode_json_object(answer)
        if value is None:
            return None
        result_fields = {
            "output",
            "final_output",
            "submission",
            "observations",
            "metrics",
            "artifacts",
            "method_provenance",
            "tool_trace",
        }
        return value if result_fields.intersection(value) else None

    @staticmethod
    def _is_text_only_output(value: Any) -> bool:
        if isinstance(value, str):
            return True
        if not isinstance(value, dict) or not value:
            return False
        return set(value).issubset({"text", "answer", "content"}) and all(
            isinstance(item, str) for item in value.values()
        )

    @classmethod
    def _looks_like_output(
        cls,
        value: dict[str, Any],
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> bool:
        required = (
            result_contract.get("required_output_fields", [])
            if isinstance(result_contract, dict)
            else []
        )
        if required:
            return all(cls._path_exists(value, str(path)) for path in required)
        if not value:
            return False
        control_fields = {
            "ok",
            "protocol",
            "execution_status",
            "contract_status",
            "success",
            "task_id",
            "request_id",
            "task_type",
            "tool_name",
            "status",
            "error",
            "errors",
            "warnings",
            "pending",
            "worker_id",
            "attempt_count",
            "max_attempts",
            "next_attempt_at",
            "task_dir",
            "log_path",
            "answer",
            "content",
            "output",
            "final_output",
            "submission",
            "result",
            "observations",
            "metrics",
            "artifacts",
            "method_provenance",
            "tool_trace",
        }
        return bool(set(value) - control_fields)

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated " + str(len(text) - limit) + " chars]"

    @classmethod
    def _extract_output(
        cls,
        answer: str,
        *,
        result_contract: dict[str, Any] | None = None,
    ) -> Any:
        """Recover an explicit JSON output when the A1 transport returns text only."""
        value = cls._decode_json_object(answer)
        if value is None:
            return None
        if cls._looks_like_output(value, result_contract=result_contract):
            return value
        return None

    @staticmethod
    def _decode_json_object(answer: str, *, max_chars: int = 262144) -> dict[str, Any] | None:
        text = str(answer or "").strip()
        if not text or len(text) > max_chars:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _path_exists(value: dict[str, Any], path: str) -> bool:
        current: Any = value
        for part in (item for item in path.split(".") if item):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return bool(path.strip())

    @staticmethod
    def _result_contract(request: A1TaskRequest) -> dict[str, Any]:
        # Keep the provider-owned option separate from the local
        # ``response_contract`` and domain workflow evidence requirements.
        value = request.step.inputs.get("biomni_result_contract")
        if not isinstance(value, dict):
            return {}
        fields = value.get("required_output_fields", [])
        if not isinstance(fields, (list, tuple)):
            return {}
        normalized = list(
            dict.fromkeys(str(field).strip() for field in fields if str(field).strip())
        )
        return {"required_output_fields": normalized} if normalized else {}

    @staticmethod
    def _tool_accepts_argument(tool: Any, property_name: str) -> bool:
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            return False
        export = getattr(schema, "model_json_schema", None)
        if callable(export):
            try:
                schema = export()
            except (TypeError, ValueError):
                return False
        return isinstance(schema, dict) and property_name in schema.get(
            "properties", {}
        )

    @classmethod
    def _compact_trace(cls, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_events = 128
        selected = trace[-max_events:] if len(trace) > max_events else trace
        compact = []
        for item in selected:
            compact.append({str(key): cls._compact_value(value, 1500) for key, value in item.items()})
        omitted = len(trace) - len(selected)
        if omitted > 0:
            compact.insert(0, {"event": "trace_truncated", "omitted_events": omitted})
        return compact

    @classmethod
    def _normalise_records(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        records = []
        for item in items[:20]:
            if isinstance(item, dict):
                records.append({str(key): cls._compact_value(val, 1500) for key, val in item.items()})
            else:
                records.append({"value": cls._compact_text(item, 1500)})
        return records

    @classmethod
    def _compact_raw(cls, value: Any) -> Any:
        return cls._compact_value(value, 2000)

    @classmethod
    def _compact_value(cls, value: Any, text_limit: int) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._compact_value(item, text_limit) for key, item in list(value.items())[:20]}
        if isinstance(value, list | tuple):
            items = list(value)
            selected = [cls._compact_value(item, text_limit) for item in items[:20]]
            if len(items) > 20:
                selected.append({"truncated_items": len(items) - 20})
            return selected
        if isinstance(value, str):
            return cls._compact_text(value, text_limit)
        return value

    @staticmethod
    def _is_rate_limited_result(result: A1TaskResult) -> bool:
        text = " ".join(result.errors).lower()
        return any(marker in text for marker in ("ratelimit", "rate limit", "tpm limit", "429"))

    @staticmethod
    def _normalise_trace(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [item if isinstance(item, dict) else {"event": str(item)} for item in items]

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
    def _answer_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

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
