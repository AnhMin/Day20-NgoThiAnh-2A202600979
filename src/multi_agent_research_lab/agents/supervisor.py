"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


def _critic_ran(state: ResearchState) -> bool:
    return any(result.agent == AgentName.CRITIC for result in state.agent_results)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        settings = get_settings()

        with trace_span("supervisor", {"iteration": state.iteration}) as span:
            if state.iteration >= settings.max_iterations:
                state.record_route("done")
                if not state.final_answer:
                    state.errors.append("Stopped: max iterations reached without final answer")
                span["attributes"]["route"] = "done"
                span["attributes"]["reason"] = "max_iterations"
                state.add_trace_event(
                    "supervisor_route",
                    {"next": "done", "reason": "max_iterations"},
                )
                return state

            if state.final_answer and _critic_ran(state):
                state.record_route("done")
                span["attributes"]["route"] = "done"
                state.add_trace_event("supervisor_route", {"next": "done", "reason": "complete"})
                return state

            if state.final_answer:
                next_route = "critic"
            elif not state.research_notes:
                next_route = "researcher"
            elif not state.analysis_notes:
                next_route = "analyst"
            elif not state.final_answer:
                next_route = "writer"
            else:
                next_route = "done"

            state.record_route(next_route)
            span["attributes"]["route"] = next_route
            state.add_trace_event("supervisor_route", {"next": next_route})
            return state
