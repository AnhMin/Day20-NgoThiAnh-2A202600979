from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    updated = SupervisorAgent().run(state)
    assert updated.route_history[-1] == "researcher"
    assert updated.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Some notes",
    )
    updated = SupervisorAgent().run(state)
    assert updated.route_history[-1] == "analyst"


def test_supervisor_routes_to_done_when_complete() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="answer",
        agent_results=[],
    )
    # First call routes to critic
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "critic"
