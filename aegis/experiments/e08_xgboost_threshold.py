"""Step 08 - threshold study for the untuned XGBoost model.

Same idea as step 03, with the full metric set. ``best_*`` columns use the
validation fold's own optimal threshold and are therefore optimistic.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import (
    best_f1_threshold,
    classification_metrics,
    cv_folds,
    fit_predict,
    ranking_metrics,
    threshold_curve,
)
from ..features import build_features
from ..models import make_xgb
from ..outputs import banner, save_table, show

REPORTED = ("f1", "precision", "recall", "balanced_accuracy", "mcc")
CONFUSION = ("tn", "fp", "fn", "tp")


def run() -> None:
    banner("STEP 08 | XGBOOST THRESHOLD STUDY (untuned)")

    rows, curves = [], []
    for panel in config.PANELS:
        X, y, groups = build_features(load_panel(panel))

        for fold in cv_folds(X, y, groups):
            _, proba = fit_predict(make_xgb, fold)

            threshold = best_f1_threshold(fold.y_val, proba)
            fixed = classification_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
            best = classification_metrics(fold.y_val, proba, threshold)

            rows.append({
                "panel": panel,
                "fold": fold.number,
                "n_train": len(fold.y_train),
                "n_val": len(fold.y_val),
                "val_label_0": int((fold.y_val == 0).sum()),
                "val_label_1": int((fold.y_val == 1).sum()),
                "variant_id_overlap": fold.group_overlap,
                **{f"{m}_0_5": fixed[m] for m in REPORTED},
                "pr_auc": fixed["pr_auc"],
                "roc_auc": fixed["roc_auc"],
                **{f"{m}_0_5": fixed[m] for m in CONFUSION},
                "best_threshold": threshold,
                **{f"best_{m}": best[m] for m in REPORTED + CONFUSION},
            })
            curves.append(
                threshold_curve(fold.y_val, proba)
                .assign(**ranking_metrics(fold.y_val, proba), panel=panel, fold=fold.number)
            )

    results = pd.DataFrame(rows)
    panel_summary = results.groupby("panel").agg(
        mean_f1_0_5=("f1_0_5", "mean"),
        mean_best_f1=("best_f1", "mean"),
        mean_best_threshold=("best_threshold", "mean"),
        std_best_threshold=("best_threshold", "std"),
        mean_precision_best=("best_precision", "mean"),
        mean_recall_best=("best_recall", "mean"),
        mean_balanced_accuracy_best=("best_balanced_accuracy", "mean"),
        mean_mcc_best=("best_mcc", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        total_tn_best=("best_tn", "sum"),
        total_fp_best=("best_fp", "sum"),
        total_fn_best=("best_fn", "sum"),
        total_tp_best=("best_tp", "sum"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()

    global_summary = pd.DataFrame([{
        "average_panel_f1_0_5": panel_summary["mean_f1_0_5"].mean(),
        "average_panel_best_f1": panel_summary["mean_best_f1"].mean(),
        "average_panel_best_balanced_accuracy": panel_summary["mean_balanced_accuracy_best"].mean(),
        "average_panel_best_mcc": panel_summary["mean_mcc_best"].mean(),
        "average_panel_pr_auc": panel_summary["mean_pr_auc"].mean(),
        "average_panel_roc_auc": panel_summary["mean_roc_auc"].mean(),
        "max_variant_id_overlap": panel_summary["max_variant_id_overlap"].max(),
    }])

    save_table(results, "aegis_08_xgboost_panel_threshold_fold_results.csv")
    save_table(pd.concat(curves, ignore_index=True), "aegis_08_xgboost_panel_threshold_curves.csv")
    save_table(panel_summary, "aegis_08_xgboost_panel_threshold_panel_summary.csv")
    save_table(global_summary, "aegis_08_xgboost_panel_threshold_global_summary.csv")

    show(panel_summary)
    print()
    show(global_summary)
