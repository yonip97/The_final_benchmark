import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils import load_data
from inference import run_judged_model_inference, run_judge_model_inference
from judgment import results_from_judge_outputs,compute_metrics_from_judgment_results

def _model_id_for_results_dir(model_id: str) -> str:
    """HF org/name ids would split path segments; only the results dir name uses this."""
    return model_id.replace("/", "_")


def make_results_dir(judged_model, judge_model, inference_prompt_name, judgement_prompt_name, results_root, data_split, allow_duplicates):
    experiment_name = (
        f"{_model_id_for_results_dir(judged_model)}__{_model_id_for_results_dir(judge_model)}__"
        f"{inference_prompt_name}__{judgement_prompt_name}"
    )
    parent_path = Path(results_root) / experiment_name
    if not parent_path.exists():
        parent_path.mkdir()
    parent_path = parent_path / data_split
    if not parent_path.exists():
        parent_path.mkdir()
    if not allow_duplicates:
        if any(parent_path.iterdir()):
            raise ValueError(f"Experiment {experiment_name} already has existing runs. "
                             "Please allow duplicates or check your arguments.")

    # 3. Create the unique timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = parent_path / f"run_{timestamp}"
    
    run_path.mkdir()
    
    return run_path

def _is_hf_model_id(model_id: str) -> bool:
    s = model_id.strip().lower()
    return s.startswith("hf:") or "/" in model_id


def _hf_model_id_for_gpu(model_id: str) -> str:
    return model_id.split(":", 1)[-1].strip() if model_id.strip().lower().startswith("hf:") else model_id.strip()


def run_evaluation(
    model: Any,
    judge_model: Any,
    results_dir: Path,
    judged_model_id: str,
    judge_model_id: str,
    split: str = "dev",
    inference_prompt_name: str = "detect_inconsistencies",
    judgement_prompt_name: str = "exact_match",
    inference_delimiter: str | None = None,
    judgment_delimiter: str | None = None,
    inference_workers: int = 4,
    judgment_workers: int = 4,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    reasoning_level: str | None = None,
    local_device: str = "cuda",
    run_config: dict | None = None,
) -> dict:
    """
    Full pipeline: inference (judged model) -> inference (judge model) -> compute_metrics.
    Both judged and judge can be api or local; backend inferred from model id (hf: or org/name → local) if not set.
    """
    df = load_data(split=split)[:12]
    if run_config is not None:
        (results_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    texts = [row["text"] for _, row in df.iterrows()]
    summaries = [row["summary"] for _, row in df.iterrows()]
    human_descriptions = [
        row["human_descriptions"] if isinstance(row["human_descriptions"], list) else []
        for _, row in df.iterrows()
    ]

    judged_backend = "local" if _is_hf_model_id(judged_model_id) else "api"
    judged_hf_id = _hf_model_id_for_gpu(judged_model_id) if judged_backend == "local" else None

    judged_outputs = run_judged_model_inference(
        model,
        backend=judged_backend,
        inference_prompt_name=inference_prompt_name,
        texts=texts,
        summaries=summaries,
        human_descriptions=human_descriptions,
        model_id=judged_hf_id,
        parallel=inference_workers > 0,
        workers=inference_workers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
        local_device=local_device,
        output_delimiter=inference_delimiter,
        save_path=results_dir / "judged_outputs.json",
    )

    judge_backend = "local" if _is_hf_model_id(judge_model_id) else "api"
    judge_hf_id = _hf_model_id_for_gpu(judge_model_id) if judge_backend == "local" else None

    judge_outputs, judge_inputs = run_judge_model_inference(
        judge_model,
        backend=judge_backend,
        judgement_prompt_name=judgement_prompt_name,
        judged_outputs=judged_outputs,
        texts=texts,
        summaries=summaries,
        human_descriptions=human_descriptions,
        model_id=judge_hf_id,
        parallel=judgment_workers > 0,
        workers=judgment_workers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        reasoning_level=reasoning_level,
        local_device=local_device,
        output_delimiter=judgment_delimiter,
        save_path=results_dir / "judge_outputs.json",
    )
    judgment_results = results_from_judge_outputs(
        judge_outputs, human_descriptions,
        texts=texts, summaries=summaries, judge_inputs=judge_inputs,
        save_path=results_dir / "judgment_results.parquet",
    )
    aggregated = compute_metrics_from_judgment_results(judgment_results, run_dir=results_dir)
    return aggregated

