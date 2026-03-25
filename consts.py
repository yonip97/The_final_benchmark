
# Per-1M-token (input USD, output USD) for ``utils.compute_price``.
PRICE_PER_1M: dict[str, tuple[float, float]] = {
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

# Longer prefixes first if two entries could both match the same id (rare).
REASONING_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-5",
    "gemini-2.5",
    "gemini-3",
    "claude-4",
)
