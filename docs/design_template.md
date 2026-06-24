# Design Template

## Problem

Build a research assistant that accepts open-ended technical queries, gathers evidence,
analyzes findings, and produces a cited final answer. The system must support comparison
between a single-agent baseline and a supervised multi-agent workflow.

## Why multi-agent?

A single agent must search, analyze, and write in one pass, which increases prompt
complexity and makes failures harder to debug. Splitting responsibilities across
Researcher, Analyst, and Writer improves traceability, allows targeted retries, and
mirrors how human research teams collaborate.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route to next worker or stop | Shared state | `route_history` | Max iterations exceeded |
| Researcher | Collect sources and notes | Query | `sources`, `research_notes` | Search/API failure |
| Analyst | Structure insights from notes | Research notes | `analysis_notes` | Empty or weak notes |
| Writer | Synthesize final answer | Notes + analysis | `final_answer` | Missing upstream context |
| Critic | Fact-check final answer | Final answer + notes | critique in `agent_results` | Over-blocking valid answers |

## Shared state

- `request`: original user query and constraints
- `sources`, `research_notes`, `analysis_notes`, `final_answer`: pipeline outputs
- `route_history`, `iteration`: routing and guardrail tracking
- `agent_results`: per-agent metadata including token cost
- `trace`, `errors`: observability and failure capture

## Routing policy

```text
supervisor -> researcher -> supervisor -> analyst -> supervisor -> writer
          -> supervisor -> critic -> supervisor -> done
```

Stop when `final_answer` exists and critic has run, or when `max_iterations` is reached.

## Guardrails

- Max iterations: 6 (configurable via `MAX_ITERATIONS`)
- Timeout: 60s per LLM call (configurable via `TIMEOUT_SECONDS`)
- Retry: 3 attempts with exponential backoff on LLM calls
- Fallback: mock LLM/search when API keys are absent
- Validation: Pydantic schemas for all core inputs/outputs

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| GraphRAG summary | Latency, cost, quality | Multi-agent slower but richer structure |
| Customer support workflows | Citation coverage | Multi-agent cites more sources |
| Production guardrails | Failure rate | Both complete; multi-agent traceable |
