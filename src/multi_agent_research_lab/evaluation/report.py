"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    *,
    title: str = "Benchmark Report",
    summary: str | None = None,
) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        f"# {title}",
        "",
    ]
    if summary:
        lines.extend([summary, ""])

    lines.extend(
        [
            "## Comparison",
            "",
            "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    if len(metrics) >= 2:
        baseline = metrics[0]
        multi = metrics[1]
        lines.extend(["", "## Analysis", ""])
        if baseline.latency_seconds and multi.latency_seconds:
            delta = multi.latency_seconds - baseline.latency_seconds
            lines.append(f"- Latency delta (multi - baseline): {delta:+.2f}s")
        if baseline.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
            cost_delta = multi.estimated_cost_usd - baseline.estimated_cost_usd
            lines.append(f"- Cost delta (multi - baseline): ${cost_delta:+.4f}")
        if baseline.quality_score is not None and multi.quality_score is not None:
            quality_delta = multi.quality_score - baseline.quality_score
            lines.append(f"- Quality delta (multi - baseline): {quality_delta:+.1f}")

    lines.append("")
    return "\n".join(lines)
