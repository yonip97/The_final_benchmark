def max_memory_for_device_ids(allowed_device_ids: tuple[int, ...]) -> dict[int, str]:
    import torch

    if not allowed_device_ids:
        return {}
    n = torch.cuda.device_count()
    allow = frozenset(allowed_device_ids)
    return {d: ("80GiB" if d in allow else "0GiB") for d in range(n)}
