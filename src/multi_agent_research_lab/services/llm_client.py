"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Approximate USD per 1M tokens for gpt-4o-mini (input / output).
_COST_PER_1M_INPUT = 0.15
_COST_PER_1M_OUTPUT = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    inp = input_tokens or 0
    out = output_tokens or 0
    return (inp * _COST_PER_1M_INPUT + out * _COST_PER_1M_OUTPUT) / 1_000_000


class LLMClient:
    """Provider-agnostic LLM client with OpenAI backend and mock fallback."""

    def __init__(self, model: str | None = None, temperature: float = 0.2) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature
        self._timeout = settings.timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        settings = get_settings()
        if not settings.openai_api_key:
            return self._mock_complete(user_prompt)
        return self._openai_complete(system_prompt, user_prompt)

    def _mock_complete(self, user_prompt: str) -> LLMResponse:
        preview = user_prompt[:300].replace("\n", " ")
        content = (
            f"[Mock LLM] Synthesized response based on the prompt: {preview}"
            if len(user_prompt) > 50
            else f"[Mock LLM] Response to: {user_prompt}"
        )
        return LLMResponse(content=content, input_tokens=80, output_tokens=120, cost_usd=0.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _openai_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            msg = "openai package not installed. Run: pip install -e '.[llm]'"
            raise AgentExecutionError(msg) from exc

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key, timeout=self._timeout)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = _estimate_cost(input_tokens, output_tokens)
        logger.info(
            "LLM call model=%s in_tokens=%s out_tokens=%s cost_usd=%s",
            self.model,
            input_tokens,
            output_tokens,
            cost,
        )
        return LLMResponse(
            content=choice,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
