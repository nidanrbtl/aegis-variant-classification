"""Step 04 - LightGBM on the engineered feature set (``aegis.features``)."""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import cv_folds, fit_predict, fixed_vs_best
from ..features import build_features
from ..models import make_lgbm
from ..outputs import banner, save_table, show


def run() -> None:
    banner("STEP 04 | ENGINEERED FEATURES (LightGBM)")

    rows = []
    for panel in config.PANELS:
        X, y, groups = build_features(load_panel(panel))
        print(f"{panel}: {X.shape[1]} features, {X.isna().mean().mean() * 100:.1f}% missing cells")

        for fold in cv_folds(X, y, groups):
            _, proba = fit_predict(make_lgbm, fold)
            rows.append({
                "panel": panel,
                "fold": fold.number,
                "n_features": X.shape[1],
                **fixed_vs_best(fold.y_val, proba),
                "variant_id_overlap": fold.group_overlap,
            })

    results = pd.DataFrame(rows)
    summary = results.groupby("panel").agg(
        mean_f1_0_5=("f1_0_5", "mean"),
        mean_best_f1=("best_f1", "mean"),
        mean_best_threshold=("best_threshold", "mean"),
        mean_balanced_accuracy_0_5=("balanced_accuracy_0_5", "mean"),
        mean_best_balanced_accuracy=("best_balanced_accuracy", "mean"),
        mean_mcc_0_5=("mcc_0_5", "mean"),
        mean_best_mcc=("best_mcc", "mean"),
        n_features=("n_features", "mean"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()

    save_table(results, "aegis_04_decode_features_fold_results.csv")
    save_table(summary, "aegis_04_decode_features_panel_summary.csv")

    show(summary)
    print(f"\nAverage panel F1 @0.5     : {summary['mean_f1_0_5'].mean():.4f}")
    print(f"Average panel F1 best thr : {summary['mean_best_f1'].mean():.4f}  (optimistic)")
