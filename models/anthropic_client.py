import os
import time

from anthropic import Anthropic, APIConnectionError, InternalServerError, RateLimitError

# ``output_config.effort`` + adaptive thinking — Opus 4.6 & Sonnet 4.6 only (not Opus 4.5; that uses budget).
_OUTPUT_EFFORT_PREFIXES = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)

# Used when ``thinking`` is adaptive (non-minimal effort models).
_REASONING_LEVEL_TO_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# Everyone else: extended thinking via ``thinking.budget_tokens`` (tokens).
_REASONING_LEVEL_TO_BUDGET_TOKENS = {
    "minimal": 0,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


def _uses_output_effort(model: str) -> bool:
    m = (model or "").strip().lower()
    return any(m.startswith(p) for p in _OUTPUT_EFFORT_PREFIXES)


class ClaudeModel:
    """Anthropic Claude inference via API with retry and exponential backoff."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self._retryable = (RateLimitError, APIConnectionError, InternalServerError)
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set (env or api_key=)")
        self._client = Anthropic(api_key=self._api_key)

    def create_config(self, prompt: str,max_new_tokens: int,temperature: float, **kwargs) -> dict:
        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        rl = (kwargs.get("reasoning_level") or "minimal").strip().lower()

        if _uses_output_effort(self.model):
            if rl == "minimal":
                create_kwargs["thinking"] = {"type": "disabled"}
                create_kwargs["output_config"] = {"effort": "low"}
            else:
                create_kwargs["thinking"] = {"type": "adaptive"}
                create_kwargs["output_config"] = {"effort": _REASONING_LEVEL_TO_EFFORT[rl]}
        else:
            budget = _REASONING_LEVEL_TO_BUDGET_TOKENS[rl]
            if budget <= 0:
                create_kwargs["thinking"] = {"type": "disabled"}
            else:
                create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        # Extended thinking: API allows temperature != 1 only when thinking is disabled (see Anthropic docs).
        th = create_kwargs.get("thinking")
        ttype = th.get("type") if isinstance(th, dict) else None
        if ttype in ("adaptive", "enabled"):
            create_kwargs["temperature"] = 1.0
        elif kwargs.get("temperature") is not None:
            create_kwargs["temperature"] = temperature

        return create_kwargs

    def infer_with_usage(self,prompt: str, max_new_tokens: int,temperature: float, **kwargs) -> tuple[str, int, int]:
        create_kwargs = self.create_config(prompt,max_new_tokens,temperature, **kwargs)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.messages.create(**create_kwargs)
                text = resp.content[-1].text
                usage = getattr(resp, "usage", None)
                in_tok = int(usage.input_tokens) if usage and getattr(usage, "input_tokens", None) is not None else 0
                out_tok = int(usage.output_tokens) if usage and getattr(usage, "output_tokens", None) is not None else 0
                return (text.strip(), in_tok, out_tok)
            except self._retryable:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(10.0 * (2**attempt), 60.0))
