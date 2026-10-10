"""Step 10 - tuned per-panel XGBoost on the engineered features.

Uses the frozen search winners (``config.FINAL_XGB_PARAMS``), i.e. the final
model *without* the ACMG-like evidence features, so comparing it with step 11
shows what those features add. Feature importances are aggregated per family.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import best_f1_threshold, classification_metrics, cv_folds, fit_predict
from ..features import build_features, feature_family
from ..models import make_xgb
from ..outputs import banner, save_table, show


def summarize_importance(importance: pd.DataFrame):
    """Per-family share and top-20 features from the per-fold importances."""
    family = importance.groupby(["panel", "family"]).agg(
        mean_importance=("importance", "mean"),
        total_importance=("importance", "sum"),
    ).reset_index()
    family["importance_share"] = family.groupby("panel")["total_importance"].transform(lambda x: x / x.sum())

    top20 = (
        importance.groupby(["panel", "feature", "family"])["importance"]
        .mean()
        .reset_index()
        .sort_values(["panel", "importance"], ascending=[True, False])
        .groupby("panel")
        .head(20)
    )
    return family, top20


def run() -> None:
    banner("STEP 10 | TUNED XGBOOST PER PANEL (engineered features)")

    rows, importance_rows = [], []
    for panel in config.PANELS:
        params = config.FINAL_XGB_PARAMS[panel]
        print(f"{panel}: {params}")
        X, y, groups = build_features(load_panel(panel))

        for fold in cv_folds(X, y, groups):
            model, proba = fit_predict(lambda: make_xgb(params), fold)

            threshold = best_f1_threshold(fold.y_val, proba)
            fixed = classification_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
            best = classification_metrics(fold.y_val, proba, threshold)

            rows.append({
                "panel": panel,
                "fold": fold.number,
                "n_features": X.shape[1],
                "best_threshold": threshold,
                "f1_0_5": fixed["f1"],
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_balanced_accuracy": best["balanced_accuracy"],
                "best_mcc": best["mcc"],
                "pr_auc": best["pr_auc"],
                "roc_auc": best["roc_auc"],
                "tn": best["tn"],
                "fp": best["fp"],
                "fn": best["fn"],
                "tp": best["tp"],
                "variant_id_overlap": fold.group_overlap,
            })
            importance_rows.extend(
                {"panel": panel, "fold": fold.number, "feature": name,
                 "family": feature_family(name), "importance": float(value)}
                for name, value in zip(X.columns, model.feature_importances_)
            )

    results = pd.DataFrame(rows)
    importance = pd.DataFrame(importance_rows)

    panel_summary = results.groupby("panel").agg(
        mean_f1=("best_f1", "mean"),
        mean_precision=("best_precision", "mean"),
        mean_recall=("best_recall", "mean"),
        mean_mcc=("best_mcc", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_threshold=("best_threshold", "mean"),
        total_tn=("tn", "sum"),
        total_fp=("fp", "sum"),
        total_fn=("fn", "sum"),
        total_tp=("tp", "sum"),
        max_variant_id_overlap=("variant_id_overlap", "max"),
    ).reset_index()
    family_summary, top20 = summarize_importance(importance)

    save_table(results, "aegis_10_modeling_suite_fold_results.csv")
    save_table(importance, "aegis_10_modeling_suite_feature_importance_raw.csv")
    save_table(panel_summary, "aegis_10_modeling_suite_panel_summary.csv")
    save_table(family_summary, "aegis_10_modeling_suite_family_importance.csv")
    save_table(top20, "aegis_10_modeling_suite_top20_features.csv")

    show(panel_summary)
    print(f"\nAverage panel F1 @0.5            : {results.groupby('panel')['f1_0_5'].mean().mean():.4f}")
    print(f"Average panel F1 per-fold best   : {panel_summary['mean_f1'].mean():.4f}  (optimistic)")
