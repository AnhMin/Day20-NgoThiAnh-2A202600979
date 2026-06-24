"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_WRITER_SYSTEM = (
    "You are a technical writer. Synthesize research and analysis into a clear final "
    "answer for the target audience. Include citations or source references where "
    "appropriate. Be accurate and avoid unsupported claims."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        with trace_span("writer") as span:
            sources_ref = "\n".join(
                f"- {doc.title} ({doc.url or 'no url'})" for doc in state.sources
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or 'n/a'}\n\n"
                f"Analysis:\n{state.analysis_notes or 'n/a'}\n\n"
                f"Available sources:\n{sources_ref}\n\n"
                "Write the final answer."
            )
            response = self.llm.complete(_WRITER_SYSTEM, user_prompt)
            state.final_answer = response.content
            span["attributes"]["answer_length"] = len(response.content)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer_done", {"length": len(response.content)})
            return state
