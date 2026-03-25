import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple

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
    args: tuple[str, tuple[int, ...] | None, list[tuple[int, str]], str | None, dict[str, Any], str],
) -> list[tuple[int, str, int, int, str | None]]:
    """One process: load with device_map=auto + max_memory on physical cuda_device_ids (no env vars)."""
    model_id, cuda_device_ids, batch, token, infer_kwargs, local_device = args
    from models.gemma3_local import Gemma3LocalModel
    from models.ministral3_local import Ministral3LocalModel

    if model_id == Ministral3LocalModel.MODEL_ID:
        model = Ministral3LocalModel(
            token=token, local_device=local_device, cuda_device_ids=cuda_device_ids
        )
    elif model_id == Gemma3LocalModel.MODEL_ID:
        model = Gemma3LocalModel(
            token=token, local_device=local_device, cuda_device_ids=cuda_device_ids
        )
    else:
        raise ValueError(f"Unknown local model_id {model_id!r}")
    out: list[tuple[int, str, int, int, str | None]] = []
    for idx, text in tqdm(batch):
        try:
            t, in_tok, out_tok = model.infer_with_usage(text, **infer_kwargs)
            out.append((idx, t, in_tok, out_tok, None))
        except Exception as e:
            out.append((idx, "", 0, 0, str(e)))
    return out


def _local_worker_tuple(args: tuple) -> list[tuple[int, str]]:
    return _local_worker_run(args)


def infer_parallel_local(
    model_id: str,
    model_inputs: List[str],
    num_processes: int,
    token: str | None = None,
    local_device: str = "cuda",
    **infer_kwargs: Any,
) -> List[Tuple[str, int, int, str | None]]:
    """
    Multiprocess local inference: each worker loads its own model (spawn), assigns GPUs round-robin.
    Each prompt is still one ``generate`` call (no tensor batching).
    Returns the same shape as ``infer_parallel_api_with_usage``: ``(text, in_tok, out_tok, err)`` per row.
    """
    import torch

    n = min(num_processes, len(model_inputs))
    batch_size = (len(model_inputs) + n - 1) // n
    batches: list[list[tuple[int, str]]] = []
    for i in range(n):
        start = i * batch_size
        end = min(start + batch_size, len(model_inputs))
        batches.append([(j, model_inputs[j]) for j in range(start, end)])

    use_cpu = local_device == "cpu" or not torch.cuda.is_available()
    gpus = 0 if use_cpu else torch.cuda.device_count()
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not use_cpu and gpus > 0 and n > 0:
        assert gpus % n == 0, (
            f"CUDA device count ({gpus}) must be divisible by worker count ({n}) "
            f"so each process gets the same number of GPUs."
        )
    per = gpus // n if (not use_cpu and gpus and n) else 0

    def _device_ids_for_worker(i: int) -> tuple[int, ...] | None:
        if not per:
            return None
        lo = i * per
        return tuple(range(lo, lo + per))

    args_list = [
        (model_id, _device_ids_for_worker(i), batches[i], token, dict(infer_kwargs), local_device)
        for i in range(n)
    ]
    ctx = mp.get_context("spawn")
    chunks: list[list[tuple[int, str, int, int, str | None]]] = []
    with ctx.Pool(n) as pool:
        for chunk in pool.imap_unordered(_local_worker_tuple, args_list):
            chunks.append(chunk)
    flat = [item for sub in chunks for item in sub]
    flat.sort(key=lambda x: x[0])
    return [(t, in_tok, out_tok, err) for _, t, in_tok, out_tok, err in flat]
