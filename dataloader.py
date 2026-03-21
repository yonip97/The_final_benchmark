import os
import pandas as pd
import json


def load_data(split: str | None = None):
    """
    Load data from data/data.jsonl. If split is given (e.g. 'dev', 'test'),
    return only rows with that split; otherwise return all.
    """
    all_data = []
    with open(os.path.join("./data/data.jsonl"), "rb") as f:
        for line in f:
            all_data.append(json.loads(line))
    df = pd.DataFrame.from_records(all_data)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df

