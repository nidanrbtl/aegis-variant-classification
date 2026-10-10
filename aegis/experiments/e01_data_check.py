"""Step 01 - sanity checks on the raw data before any modelling.

Reports size, class balance and missingness per panel, and verifies that a
variant occurring in several panels always carries the same label.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import feature_columns, load_panel
from ..outputs import banner, save_table, show


def summarize_panel(panel: str, df: pd.DataFrame) -> dict:
    features = df[feature_columns(df)]
    label = df[config.TARGET_COL]
    n_numeric = features.select_dtypes(include="number").shape[1]

    return {
        "panel": panel,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_features": features.shape[1],
        "label_0": int((label == 0).sum()),
        "label_1": int((label == 1).sum()),
        "label_0_percent": round((label == 0).mean() * 100, 2),
        "label_1_percent": round((label == 1).mean() * 100, 2),
        "unique_variant_id": df[config.ID_COL].nunique(),
        "panel_duplicate_variant_id": int(df[config.ID_COL].duplicated().sum()),
        "missing_cell_percent": round(features.isna().mean().mean() * 100, 2),
        "numeric_feature_count": n_numeric,
        "non_numeric_feature_count": features.shape[1] - n_numeric,
    }


def run() -> None:
    banner("STEP 01 | DATA CHECK")

    summaries, ids = [], []
    for panel in config.PANELS:
        df = load_panel(panel)
        summaries.append(summarize_panel(panel, df))
        ids.append(df[[config.ID_COL, config.TARGET_COL]].assign(Panel=panel))

    panel_summary = pd.DataFrame(summaries)
    all_ids = pd.concat(ids, ignore_index=True)
    n_unique = all_ids[config.ID_COL].nunique()

    global_summary = pd.DataFrame([{
        "total_rows": len(all_ids),
        "unique_variant_id": n_unique,
        "cross_panel_duplicate_rows": len(all_ids) - n_unique,
        # variants that are labelled differently in different panels
        "label_conflict_count": int(all_ids.groupby(config.ID_COL)[config.TARGET_COL].nunique().gt(1).sum()),
    }])

    save_table(panel_summary, "aegis_01_panel_data_check.csv")
    save_table(global_summary, "aegis_01_global_data_check.csv")

    show(panel_summary)
    print()
    show(global_summary)
