import json
from pathlib import Path
from typing import List


def compute_metrics_from_judgment_results(
    judgment_results: List[dict],
    run_dir: Path | None = None,
) -> dict:
    """
    Per-sample tp/fp/fn are None when discarded. Aggregates only sum over non-discarded rows;
    discarded rows are excluded from precision/recall/F1.
    """
    valid = [r for r in judgment_results if not r.get("discarded")]
    sum_tp = sum(r["tp"] for r in valid)
    sum_fp = sum(r["fp"] for r in valid)
    sum_fn = sum(r["fn"] for r in valid)
    denom_p = sum_tp + sum_fp
    denom_r = sum_tp + sum_fn
    precision = sum_tp / denom_p if denom_p else None
    recall = sum_tp / denom_r if denom_r else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    aggregated = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(judgment_results),
        "n_discarded": len(judgment_results) - len(valid),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(aggregated, indent=2))
    return aggregated
