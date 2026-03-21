import json
from pathlib import Path
from typing import Literal

PromptKind = Literal["inference", "judgement"]

_PROMPTS_BASE = "data/prompts"
_INFERENCE_DIR = "inference_prompts"
_JUDGEMENT_DIR = "judgement_prompts"


def _prompts_root(kind: PromptKind) -> Path:
    base = Path(__file__).resolve().parent
    sub = _INFERENCE_DIR if kind == "inference" else _JUDGEMENT_DIR
    return base / _PROMPTS_BASE / sub




def load_prompt_for_run(kind: PromptKind, prompt_name: str, **kwargs: str) -> str:
    """
    Load prompt from data/prompts/inference_prompts/{prompt_name}/ or data/prompts/judgement_prompts/{prompt_name}/.
    Reads prompt.txt and optional past_text_prompt.txt (appended after main). Format with kwargs.
    """
    root = _prompts_root(kind) / prompt_name
    if not root.is_dir():
        raise FileNotFoundError(f"Prompt dir not found: {root}")
    prompt_path = root / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    main = prompt_path.read_text().format(**kwargs)
    past_path = root / "past_text_prompt.txt"
    if past_path.exists():
        past = past_path.read_text().strip()
        return f"{main}\n\n{past}"
    return main


def extract_cot_final_output(raw_output: str | None, delimiter: str | None) -> str | None:
    """
    Split raw_output on delimiter and return the part after the last occurrence.
    Used for judge model output when delimiter is specified.
    """
    if not delimiter or raw_output is None:
        return raw_output
    if delimiter not in raw_output:
        return raw_output
    parts = raw_output.split(delimiter)
    return parts[-1].strip() if parts else raw_output


def extract_judge_json_object(text: str | None) -> str | None:
    """Last `{` … first `}` after it (text.split('{')[-1].split('}')[0])."""
    if text is None or not (s := text.strip()):
        return None
    if "{" not in s:
        return s
    return "{" + s.split("{")[-1].split("}")[0] + "}"


def parse_judge_output_to_dict(text: str | None) -> dict | None:
    """Delimiter output → slice JSON text → `json.loads` to dict. None if empty or not a JSON object."""
    if text is None or not str(text).strip():
        return None
    s = extract_judge_json_object(text)
    if s is None or not str(s).strip():
        return None
    try:
        obj = json.loads(s.strip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
