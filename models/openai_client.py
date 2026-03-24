import os
import re
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


def _is_gpt_5_or_up(model: str) -> bool:
    m = (model or "").strip().lower()
    mo = re.match(r"gpt-(\d+)", m)
    return bool(mo and int(mo.group(1)) >= 5)


class GPTModel:
    """OpenAI GPT inference via API with retry and exponential backoff."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_retries = max_retries
        self._retryable = (RateLimitError, APITimeoutError, APIConnectionError)
        self._reasoning_level_to_reasoning_effort = {
            "minimal": "none",
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY must be set (env or api_key=)")
        self._client = OpenAI(api_key=self._api_key)

    def create_config(self, prompt: str, max_new_tokens:int ,temperature: float, **kwargs) -> dict:
        create_kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_new_tokens,
        }
        rl = (kwargs.get("reasoning_level") or "minimal").strip().lower()
        if _is_gpt_5_or_up(self.model):
            create_kwargs["reasoning_effort"] = self._reasoning_level_to_reasoning_effort[rl]
        else:
            create_kwargs["temperature"] = temperature
        return create_kwargs

    def infer_with_usage(self, prompt: str,max_new_tokens: int,temperature: float, **kwargs) -> tuple[str, int, int]:
        create_kwargs = self.create_config(prompt,max_new_tokens,temperature, **kwargs)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**create_kwargs)
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None) or getattr(resp, "usage_metadata", None)
                in_tok = int(usage.prompt_tokens) if usage and getattr(usage, "prompt_tokens", None) is not None else 0
                out_tok = int(usage.completion_tokens) if usage and getattr(usage, "completion_tokens", None) is not None else 0
                return (text, in_tok, out_tok)
            except self._retryable:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(10.0 * (2**attempt), 60.0))
