from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from prompts import load_prompt_for_run


def _format_gold(gold_descriptions: List[str]) -> str:
    if not gold_descriptions:
        return ""
    lines = []
    for i, desc in enumerate(gold_descriptions):
        if i < 26:
            letter = chr(65 + i)
        else:
            letter = chr(65 + (i - 26) // 26) + chr(65 + (i - 26) % 26)
        s = (desc or "").strip()
        if s and not s.endswith("."):
            s += "."
        lines.append(f"{letter}. {s}")
    return "\n".join(lines)


def build_judge_model_inputs(
    judged_outputs: List[str],
    gold: List[List[str]],
    summaries: List[str],
    judgement_prompt_name: str
) -> List[str]:
    """Build one model input per sample for the judge. Structure: prompt (text), summary, predicted outputs, gold; past text prompt from template if present. Gold is formatted as A. first. B. second. ..."""
    inputs = []
    for judged_out, g, summary in zip(judged_outputs, gold, summaries):
        kwargs = {"summary": summary, "model_output": judged_out or "", "gold": _format_gold(g)}
        inputs.append(load_prompt_for_run("judgement", judgement_prompt_name, **kwargs))
    return inputs


def _parse_judge_json_to_tp_fp_fn(judge_dict: dict, gold: List[str]) -> tuple[int, int, int]:
    n_gold, n_pred = len(gold), len(judge_dict)
    if n_gold == 0 and n_pred == 0:
        return 0, 0, 0
    non_null_values = [v for v in judge_dict.values() if v is not None]
    unique_values = set(non_null_values)
    tp = len(unique_values)
    return tp, n_pred - tp, n_gold - tp


def results_from_judge_outputs(
    judge_outputs: list[dict | None],
    gold: List[List[str]],
    texts: List[str],
    summaries: List[str],
    judge_inputs: List[str],
    save_path: Path | None = None,
) -> List[dict]:
    """
    Each item in ``judge_outputs`` is either a dict from the judge or unusable (then we
    discard that row).

    Assumption 1: None of the keys or values in the judgment dict is made up. Keys are real
    model predictions; non-null values are real gold labels.

    Assumption 2: Each key is one predicted output. Its value is either the gold label it
    matches, or null when that prediction does not match any gold.

    Assumption 3: Each predicted key is matched to at most one gold item.

    **tp** — Number of unique gold matches.

    **fp** — Number of predictions - tp.

    **fn** — Number of gold labels - tp.

    With ``save_path``, we write a parquet with text, summary, human_descriptions, inputs,
    tp, fp, fn, gold, discarded, and error if present.
    """
    results = []
    for judge_out, g in tqdm(
        zip(judge_outputs, gold),
        total=len(judge_outputs),
        desc="Judgment (metrics)",
    ):
        if judge_out is None or not isinstance(judge_out, dict):
            results.append({
                "tp": None,
                "fp": None,
                "fn": None,
                "discarded": True,
            })
        else:
            tp, fp, fn = _parse_judge_json_to_tp_fp_fn(judge_out, g)
            results.append({
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "discarded": False,
            })
    if save_path is not None:
        save_path = Path(save_path)
        df = pd.DataFrame({
            "text": texts,
            "summary": summaries,
            "human_descriptions": gold,
            "inputs": judge_inputs,
            "tp": [r["tp"] for r in results],
            "fp": [r["fp"] for r in results],
            "fn": [r["fn"] for r in results],
            "discarded": [r["discarded"] for r in results],
        })
        df.to_parquet(save_path, index=False)
    return results
