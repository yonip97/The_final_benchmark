import os
import pandas as pd
import json


def load_data():
    all_data = []
    with open(os.path.join("./data/data.jsonl"), "rb") as f:
        for line in f:
            all_data.append(json.loads(line))
    df = pd.DataFrame.from_records(all_data)
    return df


if __name__ == "__main__":
    data = load_data()
