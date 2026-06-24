"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_ANALYST_SYSTEM = (
    "You are an analyst. Extract key claims from research notes, compare viewpoints, "
    "flag weak or unsupported evidence, and produce structured analysis with sections: "
    "Key Claims, Agreements, Conflicts, Gaps, and Confidence."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        with trace_span("analyst") as span:
            notes = state.research_notes or ""
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Research notes:\n{notes}\n\n"
                "Produce structured analysis."
            )
            response = self.llm.complete(_ANALYST_SYSTEM, user_prompt)
            state.analysis_notes = response.content
            span["attributes"]["notes_length"] = len(response.content)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst_done", {"length": len(response.content)})
            return state
