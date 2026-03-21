_PRICE_PER_1M = {

    "gpt-5.4": (2.50, 15.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),

    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-5-20251101": (5.00, 25.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),

    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


def compute_price(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Return estimated cost in USD for the given token counts and model.
    Uses known per-1M-token (input, output) rates; unknown models return 0.0.

    ``output_tokens`` should include all billed output-side tokens for the provider
    (e.g. OpenAI ``completion_tokens`` including reasoning; Gemini candidates plus
    thoughts; Anthropic output including extended thinking when applicable).
    """
    if input_tokens < 0 or output_tokens < 0:
        return 0.0
    model_key = (model_name or "").strip().lower()
    if not model_key:
        return 0.0
    if model_key in _PRICE_PER_1M:
        in_p, out_p = _PRICE_PER_1M[model_key]
    else:
        pair = None
        # Prefer longer / more specific keys first (e.g. gemini-3-flash-preview before gemini-3-flash).
        for key, p in sorted(_PRICE_PER_1M.items(), key=lambda kv: -len(kv[0])):
            if model_key.startswith(key) or key in model_key:
                pair = p
                break
        if pair is None:
            return 0.0
        in_p, out_p = pair
    return (input_tokens / 1_000_000) * in_p + (output_tokens / 1_000_000) * out_p
