"""
Load credentials from a file and inject them into os.environ.
Call load_credentials() at startup before using any model APIs.
"""
import os
from pathlib import Path


def load_credentials(path: str | Path | None = None) -> None:
    """
    Load key=value pairs from a credentials file and set them in os.environ.
    Lines starting with # and empty lines are ignored.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "credentials.env"
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value
