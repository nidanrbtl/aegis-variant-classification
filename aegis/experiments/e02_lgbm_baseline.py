"""Step 02 - LightGBM baseline on the untouched numeric columns, threshold 0.5."""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import cv_folds, fit_predict, threshold_metrics
from ..features import raw_numeric_features
from ..models import make_lgbm
from ..outputs import banner, save_table, show


def run() -> None:
    banner("STEP 02 | LIGHTGBM BASELINE (raw numeric features)")

    rows = []
    for panel in config.PANELS:
        X, y, groups = raw_numeric_features(load_panel(panel))

        for fold in cv_folds(X, y, groups):
            _, proba = fit_predict(make_lgbm, fold)
            metrics = threshold_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
            rows.append({
                "panel": panel,
                "fold": fold.number,
                "n_train": len(fold.y_train),
                "n_val": len(fold.y_val),
                "val_label_0": int((fold.y_val == 0).sum()),
                "val_label_1": int((fold.y_val == 1).sum()),
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mcc": metrics["mcc"],
                "variant_id_overlap": fold.group_overlap,
            })

    results = pd.DataFrame(rows)
    summary = results.groupby("panel").agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_mcc=("mcc", "mean"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()

    save_table(results, "aegis_02_lgbm_baseline_fold_results.csv")
    save_table(summary, "aegis_02_lgbm_baseline_panel_summary.csv")

    show(summary)
    print(f"\nAverage panel F1: {summary['mean_f1'].mean():.4f}")
