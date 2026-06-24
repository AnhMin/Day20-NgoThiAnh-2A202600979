"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_RESEARCHER_SYSTEM = (
    "You are a research assistant. Collect facts from the provided sources and write "
    "concise research notes. Include bullet points, name key concepts, and note which "
    "source each claim comes from when possible."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self.llm = llm or LLMClient(temperature=0.2)
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        with trace_span("researcher", {"query": state.request.query}) as span:
            sources = self.search.search(state.request.query, state.request.max_sources)
            state.sources = sources
            span["attributes"]["source_count"] = len(sources)

            sources_text = "\n".join(
                f"[{idx + 1}] {doc.title}\nURL: {doc.url or 'n/a'}\n{doc.snippet}"
                for idx, doc in enumerate(sources)
            )
            user_prompt = (
                f"Research query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Sources:\n{sources_text}\n\n"
                "Write structured research notes."
            )
            response = self.llm.complete(_RESEARCHER_SYSTEM, user_prompt)
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "sources": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("researcher_done", {"sources": len(sources)})
            return state
