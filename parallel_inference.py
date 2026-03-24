import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

import torch
from tqdm import tqdm


def infer_parallel_api_with_usage(
    model: Any,
    model_inputs: List[str],
    max_workers: int = 4,
    **infer_kwargs,
) -> List[tuple[str, int, int, str | None]]:
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
        for future in tqdm(as_completed(future_to_idx), total=n, desc="Inference (API, parallel)"):
            i, text, in_tok, out_tok, err = future.result()
            results[i] = (text, in_tok, out_tok, err)
    return results


def _local_worker_run(
    args: tuple[str, str | None, list[tuple[int, str]], str | None, dict[str, Any], str],
) -> list[tuple[int, str]]:
    """One process: optional ``CUDA_VISIBLE_DEVICES``, one model load, prompts one-by-one."""
    model_id, cuda_visible, batch, token, infer_kwargs, local_device = args
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    from models.gemma3_local import Gemma3LocalModel
    from models.ministral3_local import Ministral3LocalModel

    if model_id == Ministral3LocalModel.MODEL_ID:
        model = Ministral3LocalModel(token=token, local_device=local_device)
    elif model_id == Gemma3LocalModel.MODEL_ID:
        model = Gemma3LocalModel(token=token, local_device=local_device)
    else:
        raise ValueError(f"Unknown local model_id {model_id!r}")
    out: list[tuple[int, str]] = []
    for idx, text in batch:
        try:
            t, _, _ = model.infer_with_usage(text, **infer_kwargs)
            out.append((idx, t))
        except Exception as e:
            out.append((idx, f"[ERROR: {e}]"))
    return out


def _local_worker_tuple(args: tuple) -> list[tuple[int, str]]:
    return _local_worker_run(args)


def infer_parallel_local(
    model_id: str,
    model_inputs: List[str],
    num_processes: int | None = None,
    token: str | None = None,
    local_device: str = "cuda",
    **infer_kwargs: Any,
) -> List[str]:
    """
    Multiprocess local inference: each worker loads its own model (spawn), assigns GPUs round-robin.
    Each prompt is still one ``generate`` call (no tensor batching).
    """
    n = num_processes or min(mp.cpu_count(), len(model_inputs), 4)
    n = max(1, min(n, len(model_inputs)))
    batch_size = (len(model_inputs) + n - 1) // n
    batches: list[list[tuple[int, str]]] = []
    for i in range(n):
        start = i * batch_size
        end = min(start + batch_size, len(model_inputs))
        batches.append([(j, model_inputs[j]) for j in range(start, end)])

    use_cpu = local_device == "cpu" or not torch.cuda.is_available()
    gpus = 0 if use_cpu else torch.cuda.device_count()
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    args_list = [
        (model_id, str(i % gpus) if gpus else None, batches[i], token, dict(infer_kwargs), local_device)
        for i in range(n)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(n) as pool:
        chunks = list(
            tqdm(pool.imap(_local_worker_tuple, args_list), total=len(args_list), desc="Inference (local parallel)")
        )
    flat = [item for sub in chunks for item in sub]
    flat.sort(key=lambda x: x[0])
    return [text for _, text in flat]
