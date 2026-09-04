from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .a1_tool import BiomniA1Tool
from .biomni_gateway import BiomniGatewayClient, ToolLoader
from .contracts import A1TaskRequest, A1TaskResult
from .execution_router import (
    A1ExecutionBackend,
    HybridExecutionRouter,
    LayeredMCPExecutionBackend,
)
from .execution_models import RouteDecision
from .layered_mcp_tool import BiomniLayeredMCPTool
from .routing_policy import RoutingMode, RoutingPolicy


EventSink = Callable[[str, dict[str, Any]], None]


class BiomniExecutionGateway:
    """The Harness-facing execution boundary for all Biomni routes.

    The public surface is one ``initialize``/``run`` pair.  Internally, the
    existing deterministic router remains responsible for route policy while a
    single ``BiomniGatewayClient`` owns MCP transport, catalog compatibility,
    task polling, and request correlation.
    """

    def __init__(
        self,
        server_config: dict[str, Any],
        *,
        selector_model: Any,
        tool_loader: ToolLoader | None = None,
        max_results: int = 8,
        task_poll_interval_seconds: float = 2.0,
        task_timeout_seconds: float = 900.0,
        routing_policy: RoutingPolicy | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.event_sink = event_sink or (lambda _event, _payload: None)
        self.task_poll_interval_seconds = max(0.0, float(task_poll_interval_seconds))
        self.task_timeout_seconds = max(0.0, float(task_timeout_seconds))
        self.routing_policy = routing_policy or RoutingPolicy.from_environment()
        self.client = BiomniGatewayClient(
            server_config,
            tool_loader=tool_loader,
            task_poll_interval_seconds=task_poll_interval_seconds,
            task_timeout_seconds=task_timeout_seconds,
        )
        self._a1_tool = BiomniA1Tool(
            server_config,
            gateway=self.client,
            task_poll_interval_seconds=task_poll_interval_seconds,
            task_timeout_seconds=task_timeout_seconds,
        )
        self._mcp_tool = BiomniLayeredMCPTool(
            server_config,
            selector_model=selector_model,
            max_results=max_results,
            gateway=self.client,
            task_poll_interval_seconds=task_poll_interval_seconds,
            task_timeout_seconds=task_timeout_seconds,
        )
        self._router = HybridExecutionRouter(
            a1_backend=A1ExecutionBackend(self._a1_tool),
            mcp_backend=LayeredMCPExecutionBackend(self._mcp_tool),
            routing_policy=self.routing_policy,
            event_sink=self.event_sink,
        )

    @property
    def exposed_tool_names(self) -> tuple[str, ...]:
        if self.routing_policy.mode is RoutingMode.A1_ONLY:
            available = set(self.client.discovered_tool_names)
            return tuple(
                name
                for name in ("call_biomni", "get_biomni_task")
                if not available or name in available
            )
        return self.client.discovered_tool_names

    @property
    def route_history(self) -> list[dict[str, Any]]:
        return self._router.route_history

    @property
    def poll_request_timeout_seconds(self) -> float:
        """Expose the transport deadline so Runtime can bound a poll end-to-end."""
        return self.client.poll_request_timeout_seconds

    async def initialize(self) -> None:
        await self._router.initialize()
        self.event_sink("biomni_execution_gateway_initialized", self.describe())

    async def run(self, request: A1TaskRequest) -> A1TaskResult:
        return await self._router.run(request)

    async def bind(self, request: A1TaskRequest) -> RouteDecision:
        """Prepare a verified route before the Runtime dispatches external work."""
        return await self._router.prepare(request)

    async def dispatch(
        self,
        request: A1TaskRequest,
        decision: RouteDecision,
    ) -> A1TaskResult:
        return await self._router.dispatch(request, decision)

    async def poll(
        self,
        request: A1TaskRequest,
        task_metadata: dict[str, Any],
    ) -> A1TaskResult:
        return await self._router.poll(request, task_metadata)

    async def refresh_capabilities(self) -> list[dict[str, Any]]:
        return await self.client.refresh_catalog()

    def describe(self) -> dict[str, Any]:
        return {
            "execution_interface": "biomni_execution_gateway",
            **self.client.describe(),
        }
