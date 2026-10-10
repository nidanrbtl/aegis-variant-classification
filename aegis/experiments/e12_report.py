"""Step 12 - figures and a markdown summary of the final models.

Reads the tables written by the earlier steps; trains nothing.
"""

from __future__ import annotations

import pandas as pd

from .. import config, plots
from ..outputs import banner, load_table

STRATEGY_LABELS = {
    "fixed_0.5": "Sabit eşik (0,5)",
    "nested": "İç içe CV eşiği",
    "per_fold_best": "Doğrulama katında seçilen eşik (iyimser)",
}


def confusion_summary(panel_summary: pd.DataFrame) -> pd.DataFrame:
    """TP/TN/FP/FN with sensitivity, specificity, precision and NPV (pooled over folds)."""
    tp, tn = panel_summary["total_tp"], panel_summary["total_tn"]
    fp, fn = panel_summary["total_fp"], panel_summary["total_fn"]
    return pd.DataFrame({
        "panel": panel_summary["panel"],
        "strategy": panel_summary["strategy"],
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensitivity": (tp / (tp + fn)).round(4),
        "specificity": (tn / (tn + fp)).round(4),
        "precision": (tp / (tp + fp)).round(4),
        "npv": (tn / (tn + fn)).round(4),
    })


def markdown_report(panel_summary, overall, confusion, family, top20) -> str:
    def table(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, floatfmt=".4f")

    overall = overall.assign(strategy=overall["strategy"].map(STRATEGY_LABELS))
    panel_view = panel_summary[["panel", "strategy", "threshold", "f1", "precision", "recall",
                                "specificity", "mcc", "pr_auc", "roc_auc", "all_positive_f1"]]

    return f"""# AEGIS v2 - sonuç özeti

Bu dosya `python -m aegis report` ile üretilir; elle düzenlemeyin.

## Yapılandırma

| Alan | Değer |
|---|---|
| Model | Panel başına ayarlanmış XGBoost |
| Özellikler | Mühendislik özellikleri + ACMG benzeri kanıt skorları |
| Doğrulama | StratifiedGroupKFold ({config.N_SPLITS} kat), grup = {config.ID_COL} |
| Variant_ID sızıntısı | {int(panel_summary["variant_id_overlap"].max())} |

## Dört panelin ortalaması

`all_positive_f1`, her varyanta patojenik diyen modelin F1 değeridir; karşılaştırma tabanıdır.

{table(overall)}

## Panel bazında

{table(panel_view)}

## Karmaşıklık matrisi özeti (5 katın toplamı)

{table(confusion)}

## Özellik ailelerinin önemi

{table(family)}

## Panel başına en önemli 20 özellik

{table(top20)}
"""


def run() -> None:
    banner("STEP 12 | REPORT")

    panel_summary = load_table("aegis_v2_panel_summary.csv")
    overall = load_table("aegis_v2_overall_summary.csv")
    family = load_table("aegis_v2_family_importance.csv")
    top20 = load_table("aegis_v2_top20_features.csv")
    curves = load_table("aegis_v2_threshold_curves.csv")
    models = load_table("aegis_07_model_selection_global_summary.csv")

    predictions_path = config.PREDICTION_DIR / "aegis_v2_oof_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} is missing - run `python -m aegis final` first.")
    predictions = pd.read_csv(predictions_path)

    confusion = confusion_summary(panel_summary)
    confusion.to_csv(config.TABLE_DIR / "aegis_v2_confusion_summary.csv", index=False)

    plots.algorithm_comparison(models, baseline_f1=overall["all_positive_f1"].iloc[0])
    plots.threshold_strategies(panel_summary)
    plots.confusion_matrices(panel_summary, strategy="fixed_0.5")
    plots.threshold_curves(curves)
    plots.precision_recall_curves(predictions)
    plots.feature_importance(top20)

    report_path = config.REPORT_DIR / "aegis_v2_report.md"
    report_path.write_text(markdown_report(panel_summary, overall, confusion, family, top20), encoding="utf-8")

    print(f"Figures : {config.FIGURE_DIR}")
    print(f"Report  : {report_path}")
