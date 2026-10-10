"""Step 05 - which feature family carries the signal?

LightGBM is retrained with a single family only (``ONLY_x``) and with one
family removed (``WITHOUT_x``) and compared against the full feature set.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel, target_and_groups
from ..evaluation import cv_folds, fit_predict, fixed_vs_best
from ..features import FAMILIES, build_feature_families, combine_families
from ..models import make_lgbm
from ..outputs import banner, save_table, show

MODES = ["ALL"] + [f"ONLY_{f}" for f in FAMILIES] + [f"WITHOUT_{f}" for f in FAMILIES]


def families_for(mode: str) -> tuple[str, ...]:
    if mode == "ALL":
        return FAMILIES
    kind, family = mode.split("_", 1)
    if kind == "ONLY":
        return (family,)
    return tuple(f for f in FAMILIES if f != family)


def run() -> None:
    banner("STEP 05 | FEATURE FAMILY ABLATION (LightGBM)")

    rows = []
    for panel in config.PANELS:
        df = load_panel(panel)
        families = build_feature_families(df)
        y, groups = target_and_groups(df)

        for mode in MODES:
            print(f"{panel} | {mode}")
            X = combine_families(families, families_for(mode))

            for fold in cv_folds(X, y, groups):
                _, proba = fit_predict(make_lgbm, fold)
                rows.append({
                    "panel": panel,
                    "mode": mode,
                    "fold": fold.number,
                    "n_features": X.shape[1],
                    **fixed_vs_best(fold.y_val, proba),
                    "variant_id_overlap": fold.group_overlap,
                })

    results = pd.DataFrame(rows)
    panel_summary = results.groupby(["panel", "mode"]).agg(
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

    global_summary = panel_summary.groupby("mode").agg(
        average_panel_f1_0_5=("mean_f1_0_5", "mean"),
        average_panel_best_f1=("mean_best_f1", "mean"),
        average_panel_balanced_accuracy=("mean_best_balanced_accuracy", "mean"),
        average_panel_mcc=("mean_best_mcc", "mean"),
        avg_n_features=("n_features", "mean"),
        max_variant_id_overlap=("max_variant_id_overlap", "max"),
    ).reset_index().sort_values("average_panel_best_f1", ascending=False)

    save_table(results, "aegis_05_family_ablation_fold_results.csv")
    save_table(panel_summary, "aegis_05_family_ablation_panel_summary.csv")
    save_table(global_summary, "aegis_05_family_ablation_global_summary.csv")

    show(global_summary)
