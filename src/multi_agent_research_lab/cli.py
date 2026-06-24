"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM = (
    "You are a research assistant. Answer the user's query with accurate, structured "
    "information. If you are uncertain, say so clearly."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline_state(query: str) -> ResearchState:
    """Execute single-agent baseline and return final state."""

    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient(temperature=0.3)
    response = llm.complete(_BASELINE_SYSTEM, query)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "mode": "baseline",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    state.add_trace_event("baseline_done", {"length": len(response.content)})
    return state


def run_multi_agent_state(query: str) -> ResearchState:
    """Execute multi-agent workflow and return final state."""

    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def _load_benchmark_queries() -> list[str]:
    config_path = "configs/lab_default.yaml"
    try:
        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        queries = data.get("benchmark", {}).get("queries", [])
        if queries:
            return list(queries)
    except OSError:
        pass
    return ["Research GraphRAG state-of-the-art and write a 500-word summary"]


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    state = run_baseline_state(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    try:
        result = run_multi_agent_state(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(Panel.fit(result.final_answer or "", title="Multi-Agent Result"))
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Single query; omit to use config defaults"),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Report filename under reports/"),
    ] = "benchmark_report.md",
) -> None:
    """Benchmark single-agent vs multi-agent and write a markdown report."""

    _init()
    queries = [query] if query else _load_benchmark_queries()
    all_metrics = []

    for item in queries:
        console.print(f"[bold]Benchmarking:[/bold] {item}")
        _, baseline_metrics = run_benchmark("single-agent", item, run_baseline_state)
        _, multi_metrics = run_benchmark("multi-agent", item, run_multi_agent_state)
        all_metrics.extend([baseline_metrics, multi_metrics])

    summary = (
        "Comparison of single-agent baseline vs multi-agent workflow "
        f"across {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}."
    )
    report = render_markdown_report(all_metrics, summary=summary)
    store = LocalArtifactStore()
    path = store.write_text(output, report)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark Complete"))


if __name__ == "__main__":
    app()
