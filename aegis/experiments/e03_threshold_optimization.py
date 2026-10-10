"""Step 03 - how much does the decision threshold matter for the baseline?

For every fold the F1-optimal threshold is located on the validation fold
itself. That makes ``best_*`` an optimistic upper bound (see
``aegis.evaluation``); the unbiased version of this experiment is step 06.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import cv_folds, fit_predict, fixed_vs_best, threshold_curve
from ..features import raw_numeric_features
from ..models import make_lgbm
from ..outputs import banner, save_table, show

THRESHOLD_SUMMARY = dict(
    mean_f1_0_5=("f1_0_5", "mean"),
    mean_best_f1=("best_f1", "mean"),
    mean_best_threshold=("best_threshold", "mean"),
    std_best_threshold=("best_threshold", "std"),
    mean_balanced_accuracy_0_5=("balanced_accuracy_0_5", "mean"),
    mean_best_balanced_accuracy=("best_balanced_accuracy", "mean"),
    mean_mcc_0_5=("mcc_0_5", "mean"),
    mean_best_mcc=("best_mcc", "mean"),
    max_variant_id_overlap=("variant_id_overlap", "max"),
)


def run() -> None:
    banner("STEP 03 | THRESHOLD OPTIMISATION (LightGBM, raw numeric features)")

    rows, curves = [], []
    for panel in config.PANELS:
        X, y, groups = raw_numeric_features(load_panel(panel))

        for fold in cv_folds(X, y, groups):
            _, proba = fit_predict(make_lgbm, fold)
            rows.append({
                "panel": panel,
                "fold": fold.number,
                **fixed_vs_best(fold.y_val, proba),
                "variant_id_overlap": fold.group_overlap,
            })
            curves.append(threshold_curve(fold.y_val, proba).assign(panel=panel, fold=fold.number))

    results = pd.DataFrame(rows)
    summary = results.groupby("panel").agg(**THRESHOLD_SUMMARY).reset_index()

    save_table(results, "aegis_03_threshold_fold_results.csv")
    save_table(summary, "aegis_03_threshold_panel_summary.csv")
    save_table(pd.concat(curves, ignore_index=True), "aegis_03_threshold_curves.csv")

    show(summary)
    gain = summary["mean_best_f1"].mean() - summary["mean_f1_0_5"].mean()
    print(f"\nAverage panel F1 @0.5      : {summary['mean_f1_0_5'].mean():.4f}")
    print(f"Average panel F1 best thr  : {summary['mean_best_f1'].mean():.4f}  (optimistic, gain {gain:+.4f})")
