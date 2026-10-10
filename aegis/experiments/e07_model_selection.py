"""Step 07 - LightGBM vs XGBoost vs Random Forest vs Logistic Regression.

All candidates see the same engineered features and the same folds and are
compared at the fixed 0.5 threshold.
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .. import config
from ..data import load_panel
from ..evaluation import cv_folds, fit_predict, threshold_metrics
from ..features import build_features
from ..models import CANDIDATE_MODELS
from ..outputs import banner, save_table, show


def run() -> None:
    banner("STEP 07 | MODEL SELECTION")

    rows = []
    for panel in config.PANELS:
        X, y, groups = build_features(load_panel(panel))

        for name, make_model in CANDIDATE_MODELS.items():
            print(f"{panel} | {name}")
            for fold in cv_folds(X, y, groups):
                with warnings.catch_warnings():
                    # Two known warnings are silenced here: the median imputer skips
                    # columns that are empty in a fold, and logistic regression does
                    # not converge because the features are not scaled (a known
                    # limitation of this comparison, see README).
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                    _, proba = fit_predict(make_model, fold)
                metrics = threshold_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
                rows.append({
                    "panel": panel,
                    "model": name,
                    "fold": fold.number,
                    "f1": metrics["f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "mcc": metrics["mcc"],
                    "variant_id_overlap": fold.group_overlap,
                })

    results = pd.DataFrame(rows)
    panel_summary = results.groupby(["model", "panel"]).agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_mcc=("mcc", "mean"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()

    global_summary = panel_summary.groupby("model").agg(
        average_panel_f1=("mean_f1", "mean"),
        average_panel_balanced_accuracy=("mean_balanced_accuracy", "mean"),
        average_panel_mcc=("mean_mcc", "mean"),
        max_variant_id_overlap=("max_variant_id_overlap", "max"),
    ).reset_index().sort_values("average_panel_f1", ascending=False)

    save_table(results, "aegis_07_model_selection_fold_results.csv")
    save_table(panel_summary, "aegis_07_model_selection_panel_summary.csv")
    save_table(global_summary, "aegis_07_model_selection_global_summary.csv")

    show(global_summary)
