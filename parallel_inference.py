import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from tqdm import tqdm

from models import HuggingFaceModel


def infer_parallel_api_with_usage(
    model: Any,
    model_inputs: List[str],
    max_workers: int = 4,
    **infer_kwargs,
) -> List[tuple[str, int, int, str | None]]:
    """
    Like infer_parallel_api but returns (text, input_tokens, output_tokens, error) per item.
    error is None on success, else the exception string.
    """
    n = len(model_inputs)
    results: List[tuple[str, int, int, str | None]] = [("", 0, 0, None)] * n

    def _one(i: int, inp: str) -> tuple[int, str, int, int, str | None]:
        try:
            text, in_tok, out_tok = model.infer_with_usage(inp, **infer_kwargs)
            return (i, text, in_tok, out_tok, None)
        except Exception as e:
            return (i, "", 0, 0, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_one, i, inp): i for i, inp in enumerate(model_inputs)}
        for future in tqdm(
            as_completed(future_to_idx),
            total=n,
            desc="Inference (API, parallel)",
        ):
            i, text, in_tok, out_tok, err = future.result()
            results[i] = (text, in_tok, out_tok, err)
    return results


def _run_batch(
    model_id: str,
    device: int | str | None,
    model_input_batch: List[tuple[int, str]],
    token: str | None,
    pipeline_kwargs: dict,
) -> List[tuple[int, str]]:
    """Worker process: load model once, run inference on assigned batch. Used for GPU models."""
    model = HuggingFaceModel(
        model_id=model_id,
        device=device,
        token=token,
        **pipeline_kwargs,
    )
    out: List[tuple[int, str]] = []
    for idx, model_input in model_input_batch:
        try:
            text, _, _ = model.infer_with_usage(model_input)
            out.append((idx, text))
        except Exception as e:
            out.append((idx, f"[ERROR: {e}]"))
    return out


def _run_batch_tuple(args: tuple) -> List[tuple[int, str]]:
    """Picklable single-arg wrapper for ``Pool.imap`` (spawn-safe)."""
    return _run_batch(*args)


def infer_parallel_gpu(
    model_id: str,
    model_inputs: List[str],
    num_processes: int | None = None,
    device_per_process: int | str | None = None,
    token: str | None = None,
    **pipeline_kwargs,
) -> List[str]:
    """
    Run inference in parallel for a Hugging Face model on GPU(s).
    Splits model_inputs across num_processes; each process loads its own model copy.
    device_per_process: e.g. 0, 1 for multi-GPU, or None for default.
    """
    n = num_processes or min(mp.cpu_count(), len(model_inputs), 4)
    n = max(1, min(n, len(model_inputs)))
    # Build batches: list of (index, model_input)
    batch_size = (len(model_inputs) + n - 1) // n
    batches: List[List[tuple[int, str]]] = []
    for i in range(n):
        start = i * batch_size
        end = min(start + batch_size, len(model_inputs))
        batches.append([(j, model_inputs[j]) for j in range(start, end)])
    # Assign device per process if multi-GPU
    devices = []
    if isinstance(device_per_process, int):
        devices = [device_per_process + i for i in range(n)]
    elif device_per_process is not None:
        devices = [device_per_process] * n
    else:
        devices = [None] * n
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    # Use spawn to avoid CUDA/fork issues when each process loads its own model
    ctx = mp.get_context("spawn")
    args_list = [
        (model_id, devices[i], batches[i], token, pipeline_kwargs)
        for i in range(n)
    ]
    with ctx.Pool(n) as pool:
        results_per_proc = list(
            tqdm(
                pool.imap(_run_batch_tuple, args_list),
                total=len(args_list),
                desc="Inference (GPU parallel)",
            )
        )
    # Flatten and sort by index
    flat = [item for sub in results_per_proc for item in sub]
    flat.sort(key=lambda x: x[0])
    return [text for _, text in flat]
