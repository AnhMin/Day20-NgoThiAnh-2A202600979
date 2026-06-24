"""Search client abstraction for ResearcherAgent."""

import json
import logging
import urllib.error
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily backend and mock fallback."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()
        if settings.tavily_api_key:
            return self._tavily_search(query, max_results, settings.tavily_api_key)
        return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        logger.info("Using mock search (no TAVILY_API_KEY)")
        mock_docs = [
            SourceDocument(
                title=f"Overview: {query[:60]}",
                url="https://example.com/overview",
                snippet=(
                    f"Key concepts related to '{query}': multi-agent orchestration, "
                    "shared state, routing policies, and production guardrails."
                ),
            ),
            SourceDocument(
                title=f"Recent advances in {query[:40]}",
                url="https://example.com/advances",
                snippet=(
                    "Recent work highlights improved retrieval-augmented generation, "
                    "graph-based knowledge integration, and benchmark-driven evaluation."
                ),
            ),
            SourceDocument(
                title="Production patterns for LLM agents",
                url="https://example.com/patterns",
                snippet=(
                    "Effective agent systems use explicit roles, timeouts, max iterations, "
                    "tracing, and fallback strategies when tools or models fail."
                ),
            ),
        ]
        return mock_docs[: min(max_results, len(mock_docs))]

    def _tavily_search(self, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        payload = json.dumps(
            {"api_key": api_key, "query": query, "max_results": max_results}
        ).encode()
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results: list[SourceDocument] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", item.get("snippet", ""))[:500],
                )
            )
        if not results:
            logger.warning("Tavily returned no results; falling back to mock search")
            return self._mock_search(query, max_results)
        return results
