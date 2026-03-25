import argparse
import json
from pathlib import Path
from typing import Any

from evaluation import (
    make_results_dir,
    run_evaluation,
)
from models import ClaudeModel, GeminiModel, Gemma3LocalModel, GPTModel, Ministral3LocalModel
from utils import load_credentials


def _torch_local_device(cli: str) -> str:
    """CLI uses ``cpu`` / ``gpu``; PyTorch loaders use ``cpu`` / ``cuda``."""
    return "cuda" if cli == "gpu" else "cpu"


def get_model(model_id: str, *, api_retries: int = 5, local_device: str = "cuda") -> Any:
    """
    Resolve model_id to a model instance. Supports:
    - openai:* or gpt-* -> GPTModel
    - anthropic:* or claude-* -> ClaudeModel
    - google:* or gemini-* -> GeminiModel
    - Local: mistralai/Ministral-3-14B-Instruct-2512 or google/gemma-3-12b-it (hf: prefix optional).
      ``local_device`` is ``\"cpu\"`` or ``\"cuda\"`` for the local model classes (CLI: use ``--local-device cpu|gpu``).
    Temperature and max_new_tokens are passed at call time (infer_with_usage).
    """
    id_lower = model_id.strip().lower()
    if id_lower.startswith("openai:") or id_lower.startswith("gpt-"):
        name = model_id.split(":", 1)[-1].strip() if ":" in model_id else model_id
        return GPTModel(model=name, max_retries=api_retries)
    if id_lower.startswith("anthropic:") or id_lower.startswith("claude-"):
        name = model_id.split(":", 1)[-1].strip() if ":" in model_id else model_id
        return ClaudeModel(model=name, max_retries=api_retries)
    if id_lower.startswith("google:") or id_lower.startswith("gemini-"):
        name = model_id.split(":", 1)[-1].strip() if ":" in model_id else model_id
        return GeminiModel(model=name, max_retries=api_retries)
    if id_lower.startswith("hf:") or "/" in model_id:
        name = model_id.split(":", 1)[-1].strip() if id_lower.startswith("hf:") else model_id.strip()
        if name == Ministral3LocalModel.MODEL_ID:
            return Ministral3LocalModel(local_device=local_device)
        if name == Gemma3LocalModel.MODEL_ID:
            return Gemma3LocalModel(local_device=local_device)
        raise ValueError(
            f"Unsupported local model {name!r}. Use {Ministral3LocalModel.MODEL_ID!r} or {Gemma3LocalModel.MODEL_ID!r}."
        )
    if id_lower.startswith("gpt"):
        return GPTModel(model=model_id, max_retries=api_retries)
    if id_lower.startswith("claude"):
        return ClaudeModel(model=model_id, max_retries=api_retries)
    if id_lower.startswith("gemini"):
        return GeminiModel(model=model_id, max_retries=api_retries)
    raise ValueError(f"Unknown model id: {model_id!r}. Use openai:..., anthropic:..., google:..., or hf:... prefix.")


def main() -> None:
    load_credentials()
    parser = argparse.ArgumentParser(
        description="Run evaluation. Both --inference_prompt and --judgement_prompt are required (dir names under data/prompts/)."
    )
    parser.add_argument("--judged_model", required=True, help="Model being evaluated (API id or hf:mistralai/Ministral-3-14B-Instruct-2512 / google/gemma-3-12b-it)")
    parser.add_argument("--judge_model", required=True, help="Judge model (e.g. gpt-4o-mini); used in run dir name and for future LLM judge.")
    parser.add_argument("--inference_prompt", required=True, help="Prompt dir name under data/prompts/inference_prompts/ (e.g. zero_shot1, cot2).")
    parser.add_argument("--judgement_prompt", required=True, help="Prompt dir name under data/prompts/judgement_prompts/")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (optional).")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max new tokens (default 2000).")
    parser.add_argument("--reasoning-level",type=str,default="minimal",choices=("high", "low", "medium", "minimal"),dest="reasoning_level",help="OpenAI: reasoning_effort (gpt-5+). Gemini: 2.5 budget / 3+ level. Claude: effort (Opus 4.6, Sonnet 4.6) else thinking budget (e.g. Opus 4.5). Default: minimal.")
    parser.add_argument("--split", default="dev", choices=("dev", "test"), help="Data split (default: dev).")
    parser.add_argument("--api-retries",type=int,default=5,help="Max retries per API request with exponential backoff (OpenAI/Anthropic/Google only; default 5).")
    parser.add_argument("--inference_workers", type=int, default=4, help="Concurrency for inference (thread pool for api or model processes, default 4).")
    parser.add_argument("--judgment_workers", type=int, default=4, help="Concurrency for judge model  (thread pool for api or model processes, default 4).")
    parser.add_argument("--results_dir", type=Path, default="./results", help="Root dir for results (default: ./results). Only this tree is written; prompts and data are read-only.")
    parser.add_argument("--inference_delimiter", default=None, help="Delimiter to cut the judged model's final answer from its CoT (e.g. 'Final answer:' or '---').")
    parser.add_argument("--judgment_delimiter", default=None, help="Delimiter to cut the final JSON/dict from the judge's CoT (e.g. 'FINAL ANSWER:' or '```json'). Required for LLM judge.")
    parser.add_argument("--allow_duplicates", action="store_true", help="Allow execution if results for these models and prompts already")
    parser.add_argument("--local-device",choices=("cpu", "gpu"),default="cpu",dest="local_device",help="Local Ministral/Gemma only: cpu or gpu (default: gpu). Ignored for API models.")
    args = parser.parse_args()
    local_torch = _torch_local_device(args.local_device)

    inference_prompt_name = args.inference_prompt.strip()
    judgement_prompt_name = args.judgement_prompt.strip()

    results_dir = make_results_dir(
        args.judged_model,
        args.judge_model,
        inference_prompt_name,
        judgement_prompt_name,
        results_root=args.results_dir,
        data_split = args.split,
        allow_duplicates=args.allow_duplicates
    )
    run_config = {
        "judged_model": args.judged_model,
        "judge_model": args.judge_model,
        "inference_prompt": inference_prompt_name,
        "judgement_prompt": judgement_prompt_name,
        "inference_delimiter": args.inference_delimiter,
        "judgment_delimiter": args.judgment_delimiter,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_level": args.reasoning_level,
        "split": args.split,
        "inference_workers": args.inference_workers,
        "judgment_workers": args.judgment_workers,
        "api_retries": args.api_retries,
        "local_device": args.local_device,
    }

    judged_model = get_model(args.judged_model, api_retries=args.api_retries, local_device=local_torch)
    judge_model = get_model(args.judge_model, api_retries=args.api_retries, local_device=local_torch)
    metrics = run_evaluation(
        judged_model,
        judge_model,
        results_dir,
        judged_model_id=args.judged_model,
        judge_model_id=args.judge_model,
        split=args.split,
        inference_prompt_name=inference_prompt_name,
        judgement_prompt_name=judgement_prompt_name,
        inference_delimiter=args.inference_delimiter,
        judgment_delimiter=args.judgment_delimiter,
        inference_workers=args.inference_workers,
        judgment_workers=args.judgment_workers,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reasoning_level=args.reasoning_level,
        local_device=local_torch,
        run_config=run_config,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Results dir: {results_dir}")


if __name__ == "__main__":
    main()
