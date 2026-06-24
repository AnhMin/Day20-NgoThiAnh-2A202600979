"""Integration tests for workflow and services."""

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

langgraph = pytest.importorskip("langgraph")


def test_llm_client_mock_fallback() -> None:
    client = LLMClient()
    response = client.complete("system", "Explain multi-agent systems in one paragraph.")
    assert response.content
    assert response.input_tokens is not None


def test_search_client_mock_fallback() -> None:
    results = SearchClient().search("GraphRAG", max_results=2)
    assert len(results) == 2
    assert results[0].snippet


def test_multi_agent_workflow_runs_end_to_end() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = MultiAgentWorkflow().run(state)
    assert result.final_answer
    assert result.research_notes
    assert result.analysis_notes
    assert "researcher" in result.route_history
    assert "writer" in result.route_history
