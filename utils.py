import json
import os
from pathlib import Path

import pandas as pd

from consts import PRICE_PER_1M


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
    if model_key in PRICE_PER_1M:
        in_p, out_p = PRICE_PER_1M[model_key]
    else:
        pair = None
        # Prefer longer / more specific keys first (e.g. gemini-3-flash-preview before gemini-3-flash).
        for key, p in sorted(PRICE_PER_1M.items(), key=lambda kv: -len(kv[0])):
            if model_key.startswith(key) or key in model_key:
                pair = p
                break
        if pair is None:
            return 0.0
        in_p, out_p = pair
    return (input_tokens / 1_000_000) * in_p + (output_tokens / 1_000_000) * out_p


def load_credentials(path: str | Path | None = None) -> None:
    """
    Load key=value pairs from a credentials file and set them in os.environ.
    Lines starting with # and empty lines are ignored.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "credentials.env"
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value


def load_data(split: str | None = None):
    """
    Load data from data/data.jsonl. If split is given (e.g. 'dev', 'test'),
    return only rows with that split; otherwise return all.
    """
    all_data = []
    with open(os.path.join("./data/data.jsonl"), "rb") as f:
        for line in f:
            all_data.append(json.loads(line))
    df = pd.DataFrame.from_records(all_data)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df
