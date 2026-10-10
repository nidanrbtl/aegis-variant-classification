"""Step 06 - threshold selection without looking at the validation fold.

Step 03 picked the threshold on the fold it was then evaluated on. Here the
threshold comes from an inner cross-validation on the training part only
(``aegis.evaluation.nested_threshold``), which gives an honest estimate of
what threshold tuning is worth.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import cv_folds, fit_predict, nested_threshold, threshold_metrics
from ..features import build_features
from ..models import make_lgbm
from ..outputs import banner, save_table, show


def run() -> None:
    banner("STEP 06 | NESTED THRESHOLD SELECTION (LightGBM, engineered features)")

    rows = []
    for panel in config.PANELS:
        X, y, groups = build_features(load_panel(panel))

        for fold in cv_folds(X, y, groups):
            threshold = nested_threshold(fold.X_train, fold.y_train, fold.groups_train, make_lgbm)
            _, proba = fit_predict(make_lgbm, fold)

            fixed = threshold_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
            nested = threshold_metrics(fold.y_val, proba, threshold)
            rows.append({
                "panel": panel,
                "fold": fold.number,
                "n_features": X.shape[1],
                "nested_threshold": threshold,
                "f1_0_5": fixed["f1"],
                "f1_nested_threshold": nested["f1"],
                "balanced_accuracy_0_5": fixed["balanced_accuracy"],
                "balanced_accuracy_nested": nested["balanced_accuracy"],
                "mcc_0_5": fixed["mcc"],
                "mcc_nested": nested["mcc"],
                "variant_id_overlap": fold.group_overlap,
            })
            print(f"{panel} fold {fold.number}: threshold {threshold:.2f}, "
                  f"F1 {fixed['f1']:.4f} -> {nested['f1']:.4f}")

    results = pd.DataFrame(rows)
    summary = results.groupby("panel").agg(
        mean_f1_0_5=("f1_0_5", "mean"),
        mean_f1_nested=("f1_nested_threshold", "mean"),
        mean_nested_threshold=("nested_threshold", "mean"),
        std_nested_threshold=("nested_threshold", "std"),
        mean_balanced_accuracy_0_5=("balanced_accuracy_0_5", "mean"),
        mean_balanced_accuracy_nested=("balanced_accuracy_nested", "mean"),
        mean_mcc_0_5=("mcc_0_5", "mean"),
        mean_mcc_nested=("mcc_nested", "mean"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()

    save_table(results, "aegis_06_nested_threshold_fold_results.csv")
    save_table(summary, "aegis_06_nested_threshold_panel_summary.csv")

    show(summary)
    fixed_f1, nested_f1 = summary["mean_f1_0_5"].mean(), summary["mean_f1_nested"].mean()
    print(f"\nAverage panel F1 @0.5             : {fixed_f1:.4f}")
    print(f"Average panel F1 nested threshold : {nested_f1:.4f}  (gain {nested_f1 - fixed_f1:+.4f})")
