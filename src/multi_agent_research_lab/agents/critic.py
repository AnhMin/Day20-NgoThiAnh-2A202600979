"""Optional critic agent for fact-checking and safety review."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_CRITIC_SYSTEM = (
    "You are a critic and fact-checker. Review the final answer against research notes "
    "and sources. Report: (1) unsupported claims, (2) missing citations, "
    "(3) hallucination risk, (4) overall quality score 0-10 with brief justification."
)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient(temperature=0.0)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        with trace_span("critic") as span:
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or 'n/a'}\n\n"
                f"Final answer:\n{state.final_answer or 'n/a'}\n\n"
                "Provide a concise critique."
            )
            response = self.llm.complete(_CRITIC_SYSTEM, user_prompt)
            span["attributes"]["critique_length"] = len(response.content)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("critic_done", {"length": len(response.content)})
            return state
