from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    context: Any


class ExecutionStateGraph:
    """Compile the Harness phases into an explicit async state graph.

    RunPersistence is the durable checkpoint source of truth. LangGraph is used
    only for phase transitions; external side effects remain guarded by the
    ActionLedger and are resumed from the persisted task identity.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        builder = StateGraph(GraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("bind", self._bind)
        builder.add_node("dispatch", self._dispatch)
        builder.add_node("wait_external", self._wait_external)
        builder.add_node("verify", self._verify)
        builder.add_node("reduce", self._reduce)
        builder.add_node("materialize", self._materialize)
        builder.add_node("finalize", self._finalize)

        builder.add_conditional_edges(
            START,
            self._entry,
            {
                "plan": "plan",
                "bind": "bind",
                "wait_external": "wait_external",
                "finalize": "finalize",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "plan",
            self._after_plan,
            {"bind": "bind", "end": END},
        )
        builder.add_conditional_edges(
            "bind",
            self._after_bind,
            {
                "dispatch": "dispatch",
                "wait_external": "wait_external",
                "verify": "verify",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "dispatch",
            self._after_dispatch,
            {"wait_external": "wait_external", "verify": "verify", "end": END},
        )
        builder.add_conditional_edges(
            "wait_external",
            self._after_wait_external,
            {"verify": "verify", "end": END},
        )
        builder.add_edge("verify", "reduce")
        builder.add_edge("reduce", "materialize")
        builder.add_conditional_edges(
            "materialize",
            self._after_materialize,
            {"bind": "bind", "finalize": "finalize", "end": END},
        )
        builder.add_edge("finalize", END)
        self._compiled = builder.compile()

    async def run(self, context: Any) -> None:
        await self._compiled.ainvoke({"context": context})

    @staticmethod
    def _context(data: GraphState) -> Any:
        return data["context"]

    def _entry(self, data: GraphState) -> str:
        context = self._context(data)
        state = context.state
        if context.terminal:
            return "end"
        if state.pending_execution is not None:
            return "wait_external"
        if context.plan is None:
            return "plan"
        if context.cursor >= len(context.plan.steps):
            return "finalize"
        return "bind"

    async def _plan(self, data: GraphState) -> GraphState:
        context = self._context(data)
        await self.runtime._graph_plan(context)
        return data

    def _after_plan(self, data: GraphState) -> str:
        context = self._context(data)
        if context.terminal:
            return "end"
        if context.state.status.value != "running":
            return "end"
        if context.plan is None:
            return "end"
        return "bind"

    async def _bind(self, data: GraphState) -> GraphState:
        context = self._context(data)
        await self.runtime._graph_bind(context)
        return data

    def _after_bind(self, data: GraphState) -> str:
        context = self._context(data)
        if context.terminal:
            return "end"
        if context.pending:
            return "wait_external"
        if context.skip_dispatch:
            return "verify"
        return "dispatch"

    async def _dispatch(self, data: GraphState) -> GraphState:
        context = self._context(data)
        await self.runtime._graph_dispatch(context)
        return data

    def _after_dispatch(self, data: GraphState) -> str:
        context = self._context(data)
        if context.terminal:
            return "end"
        return "wait_external" if context.pending else "verify"

    async def _wait_external(self, data: GraphState) -> GraphState:
        context = self._context(data)
        await self.runtime._graph_wait_external(context)
        return data

    def _after_wait_external(self, data: GraphState) -> str:
        context = self._context(data)
        if context.terminal or context.state.status.value != "running":
            return "end"
        if context.pending:
            return "wait_external"
        return "verify"

    async def _verify(self, data: GraphState) -> GraphState:
        context = self._context(data)
        self.runtime._graph_verify(context)
        return data

    async def _reduce(self, data: GraphState) -> GraphState:
        context = self._context(data)
        self.runtime._graph_reduce(context)
        return data

    async def _materialize(self, data: GraphState) -> GraphState:
        context = self._context(data)
        self.runtime._graph_materialize(context)
        return data

    def _after_materialize(self, data: GraphState) -> str:
        context = self._context(data)
        if context.terminal or context.plan is None:
            return "end"
        if (
            context.cursor < len(context.plan.steps)
            and context.state.a1_call_count < self.runtime.policy.max_a1_calls
        ):
            return "bind"
        return "finalize"

    async def _finalize(self, data: GraphState) -> GraphState:
        context = self._context(data)
        await self.runtime._graph_finalize(context)
        return data
