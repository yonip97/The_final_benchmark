"""
Google Gemini inference via the new unified SDK (google-genai).
Supports Gemini 2.5 and up. Use model names like gemini-2.5-flash, gemini-2.5-pro.
"""
import os
import time

from google import genai
from google.genai import types

try:
    from google.api_core import exceptions as google_exc

    _GOOGLE_RETRYABLE = (
        google_exc.ResourceExhausted,
        google_exc.ServiceUnavailable,
        google_exc.DeadlineExceeded,
        google_exc.InternalServerError,
        ConnectionError,
    )
except ImportError:
    _GOOGLE_RETRYABLE = (ConnectionError, OSError)

# Gemini 2.5: ``thinking_budget`` (tokens; 0 = off).
_REASONING_LEVEL_TO_THINKING_BUDGET = {
    "minimal": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

# Gemini 3+: ``thinking_level``.
_REASONING_LEVEL_TO_THINKING_LEVEL = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


class GeminiModel:
    """Google Gemini inference via the new unified SDK (google-genai). Gemini 2.5+."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY must be set (env or api_key=)")

        self._retryable = _GOOGLE_RETRYABLE
        self._client = genai.Client(api_key=self._api_key)

    def create_config(self,max_new_tokens: int,temperature: float, **kwargs) -> types.GenerateContentConfig:
        rl = (kwargs.get("reasoning_level") or "minimal").strip().lower()
        config_kwargs: dict = {"max_output_tokens": max_new_tokens}
        config_kwargs["temperature"] = temperature

        m = self.model.strip().lower()
        if m.startswith("gemini-2.5"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=_REASONING_LEVEL_TO_THINKING_BUDGET[rl]
            )
        elif m.startswith("gemini-3") or m.startswith("gemini-4"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=_REASONING_LEVEL_TO_THINKING_LEVEL[rl]
            )
        elif rl != "minimal":
            raise ValueError(
                "reasoning_level only applies to gemini-2.5* (budget) or gemini-3+* (level); "
                "otherwise use reasoning_level=minimal."
            )

        return types.GenerateContentConfig(**config_kwargs)

    def infer_with_usage(self, prompt: str,max_new_tokens: int,temperature: float, **kwargs) -> tuple[str, int, int]:
        config = self.create_config(max_new_tokens,temperature,**kwargs)

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                text = (resp.text or "").strip()
                usage = getattr(resp, "usage_metadata", None)
                in_tok = int(usage.prompt_token_count) if usage and getattr(usage, "prompt_token_count", None) is not None else 0
                cand = int(usage.candidates_token_count) if usage and getattr(usage, "candidates_token_count", None) is not None else 0
                thoughts = int(usage.thoughts_token_count) if usage and getattr(usage, "thoughts_token_count", None) is not None else 0
                out_tok = cand + thoughts
                return (text, in_tok, out_tok)
            except self._retryable:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(10.0 * (2**attempt), 60.0))
