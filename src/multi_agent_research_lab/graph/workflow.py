"""LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any, Literal

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

Route = Literal["researcher", "analyst", "writer", "critic", "done"]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        return self.supervisor.run(state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        return self.researcher.run(state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        return self.analyst.run(state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        return self.writer.run(state)

    def _critic_node(self, state: ResearchState) -> ResearchState:
        return self.critic.run(state)

    @staticmethod
    def _route_from_supervisor(state: ResearchState) -> Route:
        if not state.route_history:
            return "done"
        route = state.route_history[-1]
        if route in {"researcher", "analyst", "writer", "critic", "done"}:
            return route  # type: ignore[return-value]
        return "done"

    def build(self) -> Any:
        """Create a compiled LangGraph graph."""

        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            msg = "langgraph not installed. Run: pip install -e '.[llm]'"
            raise ImportError(msg) from exc

        graph: StateGraph[ResearchState] = StateGraph(ResearchState)

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("critic", self._critic_node)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )

        for worker in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            graph = self.build()
            result = graph.invoke(state)
            if isinstance(result, ResearchState):
                final_state = result
            else:
                final_state = ResearchState.model_validate(result)
            span["attributes"]["iterations"] = final_state.iteration
            span["attributes"]["routes"] = final_state.route_history
            logger.info(
                "Workflow finished iterations=%s routes=%s",
                final_state.iteration,
                final_state.route_history,
            )
            return final_state
