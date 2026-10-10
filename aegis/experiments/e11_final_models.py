"""Step 11 - the final AEGIS v2 models: one tuned XGBoost per panel on the
engineered features plus the ACMG-like evidence scores.

Every fold is evaluated under three threshold strategies (see
``aegis.evaluation``): ``fixed_0.5``, ``nested`` and ``per_fold_best``. The
last one is what the competition-time result tables used; because it tunes the
threshold on the validation fold it is an optimistic upper bound. ``nested``
shows what threshold tuning is worth without that leak.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data import load_panel
from ..evaluation import (
    all_positive_f1,
    best_f1_threshold,
    classification_metrics,
    cv_folds,
    fit_predict,
    nested_threshold,
    ranking_metrics,
    threshold_curve,
)
from ..evidence import add_evidence_features
from ..features import build_features, feature_family
from ..models import make_xgb
from ..outputs import banner, save_table, show

STRATEGIES = ("fixed_0.5", "nested", "per_fold_best")
MEAN_METRICS = ("threshold", "f1", "precision", "recall", "specificity",
                "balanced_accuracy", "mcc", "pr_auc", "roc_auc")
CONFUSION = ("tn", "fp", "fn", "tp")


def evaluate_panel(panel: str):
    """Cross-validate the final model of one panel."""
    df = load_panel(panel)
    X, y, groups = build_features(df)
    X = add_evidence_features(X, panel)

    def make_model():
        return make_xgb(config.FINAL_XGB_PARAMS[panel])

    rows, importance, curves, predictions = [], [], [], []
    for fold in cv_folds(X, y, groups, drop_constant=True):
        model, proba = fit_predict(make_model, fold)

        thresholds = {
            "fixed_0.5": config.DEFAULT_THRESHOLD,
            "nested": nested_threshold(fold.X_train, fold.y_train, fold.groups_train, make_model),
            "per_fold_best": best_f1_threshold(fold.y_val, proba),
        }
        for strategy, threshold in thresholds.items():
            rows.append({
                "panel": panel,
                "fold": fold.number,
                "strategy": strategy,
                "threshold": threshold,
                **classification_metrics(fold.y_val, proba, threshold),
                "all_positive_f1": all_positive_f1(fold.y_val),
                "n_features_after_constant_drop": fold.X_train.shape[1],
                "variant_id_overlap": fold.group_overlap,
            })

        importance.extend(
            {"panel": panel, "fold": fold.number, "feature": name, "importance": float(value)}
            for name, value in zip(fold.X_train.columns, model.feature_importances_)
        )
        curves.append(
            threshold_curve(fold.y_val, proba)
            .assign(**ranking_metrics(fold.y_val, proba), panel=panel, fold=fold.number)
        )
        predictions.append(pd.DataFrame({
            "panel": panel,
            "fold": fold.number,
            config.ID_COL: groups.iloc[fold.val_index].to_numpy(),
            "y_true": fold.y_val.to_numpy(),
            "proba": proba,
        }))

        f1 = {r["strategy"]: r["f1"] for r in rows[-3:]}
        print(f"{panel} fold {fold.number}: F1 @0.5 {f1['fixed_0.5']:.4f} | "
              f"nested {f1['nested']:.4f} | per-fold best {f1['per_fold_best']:.4f}")

    return (pd.DataFrame(rows), pd.DataFrame(importance),
            pd.concat(curves, ignore_index=True), pd.concat(predictions, ignore_index=True))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics and pooled confusion counts per panel and strategy."""
    aggregations = {m: (m, "mean") for m in MEAN_METRICS}
    aggregations.update({f"total_{c}": (c, "sum") for c in CONFUSION})
    aggregations["all_positive_f1"] = ("all_positive_f1", "mean")
    aggregations["variant_id_overlap"] = ("variant_id_overlap", "max")
    return results.groupby(["panel", "strategy"], sort=False).agg(**aggregations).reset_index()


def average_over_panels(panel_summary: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean over the four panels (the competition's headline metric)."""
    columns = [m for m in MEAN_METRICS if m != "threshold"] + ["all_positive_f1"]
    return panel_summary.groupby("strategy", sort=False)[columns].mean().reset_index()


def summarize_importance(importance: pd.DataFrame):
    importance = importance.assign(family=importance["feature"].map(feature_family))

    family = importance.groupby(["panel", "family"]).agg(
        total_importance=("importance", "sum"),
        mean_importance=("importance", "mean"),
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
    banner("STEP 11 | FINAL MODELS (AEGIS v2)")

    parts = [evaluate_panel(panel) for panel in config.PANELS]
    results, importance, curves, predictions = (
        pd.concat(frames, ignore_index=True) for frames in zip(*parts)
    )

    panel_summary = summarize(results)
    overall = average_over_panels(panel_summary)
    family, top20 = summarize_importance(importance)

    save_table(results, "aegis_v2_fold_results.csv")
    save_table(importance, "aegis_v2_feature_importance.csv")
    save_table(curves, "aegis_v2_threshold_curves.csv")
    save_table(panel_summary, "aegis_v2_panel_summary.csv")
    save_table(overall, "aegis_v2_overall_summary.csv")
    save_table(family, "aegis_v2_family_importance.csv")
    save_table(top20, "aegis_v2_top20_features.csv")
    predictions.to_csv(config.PREDICTION_DIR / "aegis_v2_oof_predictions.csv", index=False)

    banner("AEGIS v2 | PANEL SUMMARY")
    show(panel_summary[["panel", "strategy", "threshold", "f1", "precision", "recall",
                        "specificity", "mcc", "pr_auc", "roc_auc"]])
    banner("AEGIS v2 | AVERAGE OVER PANELS")
    show(overall)
