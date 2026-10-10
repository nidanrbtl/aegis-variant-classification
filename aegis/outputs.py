"""Small helpers for writing result tables and console output."""

from __future__ import annotations

import pandas as pd

from . import config


def save_table(df: pd.DataFrame, filename: str) -> None:
    config.ensure_output_dirs()
    df.to_csv(config.TABLE_DIR / filename, index=False)


def load_table(filename: str) -> pd.DataFrame:
    path = config.TABLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run the step that produces it first.")
    return pd.read_csv(path)


def banner(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def show(df: pd.DataFrame, digits: int = 4) -> None:
    print(df.round(digits).to_string(index=False))
