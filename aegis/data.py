"""Loading and basic validation of the panel CSV files."""

from __future__ import annotations

import pandas as pd

from . import config


def load_panel(panel: str) -> pd.DataFrame:
    """Read one panel and check that the id and label columns are present."""
    if panel not in config.DATA_FILES:
        raise KeyError(f"Unknown panel '{panel}'. Expected one of {config.PANELS}.")

    path = config.DATA_DIR / config.DATA_FILES[panel]
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "The competition data is not part of this repository; "
            "see data/README.md for where to place it."
        )

    df = pd.read_csv(path)

    missing = {config.ID_COL, config.TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{panel}: required columns are missing: {sorted(missing)}")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column except the variant id and the label."""
    return [c for c in df.columns if c not in (config.ID_COL, config.TARGET_COL)]


def target_and_groups(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the label vector and the Variant_ID grouping used for the CV split."""
    return df[config.TARGET_COL].astype(int), df[config.ID_COL]
