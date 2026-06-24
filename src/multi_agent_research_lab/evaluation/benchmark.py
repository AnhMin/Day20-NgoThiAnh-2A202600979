"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _estimate_cost(state: ResearchState) -> float | None:
    costs: list[float] = []
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            costs.append(float(cost))
    if not costs:
        return 0.0 if state.final_answer else None
    return sum(costs)


def _estimate_quality(state: ResearchState) -> float | None:
    if not state.final_answer:
        return None

    score = 5.0
    if state.sources:
        score += min(1.5, len(state.sources) * 0.3)
    if state.research_notes and len(state.research_notes) > 200:
        score += 1.0
    if state.analysis_notes and len(state.analysis_notes) > 150:
        score += 1.0
    if len(state.final_answer) > 300:
        score += 1.0
    if state.errors:
        score -= min(2.0, len(state.errors))
    return max(0.0, min(10.0, score))


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, and heuristic quality for a workflow run."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    citation_coverage = None
    if state.final_answer and state.sources:
        cited = sum(1 for src in state.sources if src.title.lower() in state.final_answer.lower())
        citation_coverage = cited / len(state.sources)

    notes_parts = [
        f"sources={len(state.sources)}",
        f"errors={len(state.errors)}",
    ]
    if citation_coverage is not None:
        notes_parts.append(f"citation_coverage={citation_coverage:.0%}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost(state),
        quality_score=_estimate_quality(state),
        notes=", ".join(notes_parts),
    )
    return state, metrics
