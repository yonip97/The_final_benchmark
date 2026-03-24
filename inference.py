import json
from pathlib import Path
from typing import Any, Callable, List

from tqdm import tqdm

from judgment import build_judge_model_inputs
from parallel_inference import infer_parallel_api_with_usage, infer_parallel_local
from prompts import extract_cot_final_output, load_prompt_for_run, parse_judge_output_to_dict
from utils import compute_price


def _infer_kwargs(
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
) -> dict:
    d: dict = {}
    if temperature is not None:
        d["temperature"] = temperature
    if max_new_tokens is not None:
        d["max_new_tokens"] = max_new_tokens
    if reasoning_level is not None:
        d["reasoning_level"] = reasoning_level
    return d


def _apply_delimiter_and_save(
    raw_outputs: List[str | None],
    prices: List[float],
    errors: List[str | None],
    model_inputs: List[str],
    output_delimiter: str | None,
    save_path: Path | None,
    texts: List[str],
    summaries: List[str],
    human_descriptions: List[List[str]],
    postprocess: Callable[[str | None], Any] | None = None,
) -> List[Any]:
    """Apply delimiter, optional postprocess (e.g. judge → dict); if save_path set, save columns: text, summary, human_descriptions, inputs, raw_outputs, processed_output, prices, errors. Return processed output per sample."""
    processed: List[Any] = []
    for raw in raw_outputs:
        out = extract_cot_final_output(raw, output_delimiter) if output_delimiter else raw
        out = out.strip() if isinstance(out, str) else out
        if postprocess is not None:
            out = postprocess(out)
        processed.append(out)
    if save_path is not None:
        save_path = Path(save_path)
        payload = {
            "text": texts,
            "summary": summaries,
            "human_descriptions": human_descriptions,
            "inputs": model_inputs,
            "raw_outputs": raw_outputs,
            "processed_output": processed,
            "prices": prices,
            "errors": errors,
        }
        save_path.write_text(json.dumps(payload, indent=2))
    return processed


def run_model_inference(
    model: Any,
    model_inputs: List[str],
    backend: str,
    model_id: str | None = None,
    parallel: bool = False,
    workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
    local_device: str = "cuda",
) -> tuple[List[str | None], List[float], List[str | None]]:
    """Run model on inputs; returns (raw_outputs, prices, errors). Dispatch to API or local internally."""
    if backend == "api":
        return run_inference_api(
            model, model_inputs,
            parallel=parallel, max_workers=workers,
            temperature=temperature, max_new_tokens=max_new_tokens,
            reasoning_level=reasoning_level,
        )
    if backend == "local" and model_id is None:
        raise ValueError("model_id is required for local inference")
    return run_inference_local(
        model, model_inputs, model_id=model_id,
        parallel=parallel, workers=workers,
        temperature=temperature, max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
        local_device=local_device,
    )


def run_inference_api(
    model: Any,
    model_inputs: List[str],
    *,
    parallel: bool = False,
    max_workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
) -> tuple[List[str | None], List[float], List[str | None]]:
    """API backend: single worker or multiple (thread pool). Returns (raw_outputs, prices, errors)."""
    infer_kwargs = _infer_kwargs(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
    )
    model_name = getattr(model, "model", None) or getattr(model, "model_id", "") or ""
    if parallel:
        raw_tuples = infer_parallel_api_with_usage(model, model_inputs, max_workers=max_workers, **infer_kwargs)
        raw_outputs = [text for text, _, _, _ in raw_tuples]
        prices = [0.0 if err else compute_price(in_tok, out_tok, model_name) for _, in_tok, out_tok, err in raw_tuples]
        errors = [err for _, _, _, err in raw_tuples]
        return (raw_outputs, prices, errors)
    raw_outputs = []
    prices = []
    errors = []
    for inp in tqdm(model_inputs, desc="Inference (API)"):
        try:
            text, in_tok, out_tok = model.infer_with_usage(inp, **infer_kwargs)
            raw_outputs.append(text)
            prices.append(compute_price(in_tok, out_tok, model_name))
            errors.append(None)
        except Exception as e:
            raw_outputs.append(None)
            prices.append(0.0)
            errors.append(str(e))
    return (raw_outputs, prices, errors)


def run_inference_local(
    model: Any,
    model_inputs: List[str],
    *,
    model_id: str | None = None,
    parallel: bool = False,
    workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
    local_device: str = "cuda",
) -> tuple[List[str | None], List[float], List[str | None]]:
    """Local backend: sequential in-process, or multiprocess (one model copy per worker, one generate per prompt). price=0."""
    infer_kwargs = _infer_kwargs(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
    )
    if parallel and model_id is not None and workers > 1:
        raw = infer_parallel_local(
            model_id, model_inputs, num_processes=workers, local_device=local_device, **infer_kwargs
        )
        raw_outputs: List[str | None] = []
        prices: List[float] = []
        errors: List[str | None] = []
        for s in raw:
            if s.startswith("[ERROR:") and "]" in s:
                raw_outputs.append(None)
                prices.append(0.0)
                errors.append(s.strip())
            else:
                raw_outputs.append(s)
                prices.append(0.0)
                errors.append(None)
        return (raw_outputs, prices, errors)
    raw_outputs = []
    prices = []
    errors = []
    for inp in tqdm(model_inputs, desc="Inference (local)"):
        try:
            text, _, _ = model.infer_with_usage(inp, **infer_kwargs)
            raw_outputs.append(text)
            prices.append(0.0)
            errors.append(None)
        except Exception as e:
            raw_outputs.append(None)
            prices.append(0.0)
            errors.append(str(e))
    return (raw_outputs, prices, errors)


def run_judged_model_inference(
    model: Any,
    backend: str,
    inference_prompt_name: str,
    texts: List[str],
    summaries: List[str],
    human_descriptions: List[List[str]],
    model_id: str | None = None,
    parallel: bool = False,
    workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
    local_device: str = "cuda",
    output_delimiter: str | None = None,
    save_path: Path | None = None,
) -> List[str | None]:
    """Build model_inputs from texts/summaries, run inference, save to save_path. Returns processed output per sample."""
    model_inputs = [
        load_prompt_for_run("inference", inference_prompt_name, text=t, summary=s)
        for t, s in zip(texts, summaries)
    ]
    raw_outputs, prices, errors = run_model_inference(
        model, model_inputs, backend=backend, model_id=model_id,
        parallel=parallel, workers=workers,
        temperature=temperature, max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
        local_device=local_device,
    )
    return _apply_delimiter_and_save(
        raw_outputs, prices, errors, model_inputs, output_delimiter, save_path,
        texts, summaries, human_descriptions,
    )


def run_judge_model_inference(
    model: Any,
    backend: str,
    judgement_prompt_name: str,
    judged_outputs: List[str],
    texts: List[str],
    summaries: List[str],
    human_descriptions: List[List[str]],
    model_id: str | None = None,
    parallel: bool = False,
    workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
    local_device: str = "cuda",
    output_delimiter: str | None = None,
    save_path: Path | None = None,
) -> tuple[List[dict | None], List[str]]:
    """Build model_inputs from judged_outputs and gold (human_descriptions), run inference, save to save_path. Returns (parsed judge dicts per sample, judge model inputs)."""
    model_inputs = build_judge_model_inputs(judged_outputs, human_descriptions, summaries, judgement_prompt_name)
    raw_outputs, prices, errors = run_model_inference(
        model, model_inputs, backend=backend, model_id=model_id,
        parallel=parallel, workers=workers,
        temperature=temperature, max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
        local_device=local_device,
    )
    processed = _apply_delimiter_and_save(
        raw_outputs, prices, errors, model_inputs, output_delimiter, save_path,
        texts, summaries, human_descriptions,
        postprocess=parse_judge_output_to_dict,
    )
    return (processed, model_inputs)
