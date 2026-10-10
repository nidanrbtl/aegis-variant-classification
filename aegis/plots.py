"""Figures for the final report. All text on the figures is Turkish because
they go straight into the project report and the README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

from . import config

# One small, fixed palette: three series colours, a neutral for context marks
# and a single-hue ramp for magnitudes.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#898781"
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#eef4fd", "#86b6ef", "#2a78d6", "#0d366b"])

STRATEGY_STYLE = {
    "fixed_0.5": ("Sabit eşik (0,5)", BLUE, "o"),
    "nested": ("İç içe CV ile seçilen eşik", ORANGE, "s"),
    "per_fold_best": ("Doğrulama katında seçilen eşik (iyimser)", AQUA, "D"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
})


def _save(fig, filename: str) -> None:
    config.ensure_output_dirs()
    fig.savefig(config.FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def algorithm_comparison(model_summary: pd.DataFrame, baseline_f1: float) -> None:
    """Average panel F1 of the candidate algorithms at threshold 0.5."""
    data = model_summary.sort_values("average_panel_f1")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(data["model"], data["average_panel_f1"], height=0.5, color=BLUE)
    for row, value in enumerate(data["average_panel_f1"]):
        ax.text(value - 0.012, row, _decimal(value), ha="right", va="center", color="white", fontweight="bold")
    ax.axvline(baseline_f1, color=NEUTRAL, linewidth=1.2)
    ax.text(baseline_f1 - 0.012, 0.5, f"taban (hepsine patojenik demek): {_decimal(baseline_f1)}",
            ha="right", va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Ortalama panel F1 (eşik 0,5)")
    ax.set_title("Algoritma karşılaştırması", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, "aegis_v2_algorithm_comparison_f1.png")


def threshold_strategies(panel_summary: pd.DataFrame) -> None:
    """F1 of the final models per panel under the three threshold strategies."""
    panels = [p for p in config.PANELS if p in set(panel_summary["panel"])]
    rows = {panel: i for i, panel in enumerate(reversed(panels))}

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    offsets = dict(zip(STRATEGY_STYLE, (0.2, 0.0, -0.2)))  # keeps near-identical values apart
    for strategy, (label, color, marker) in STRATEGY_STYLE.items():
        part = panel_summary[panel_summary["strategy"] == strategy]
        ax.scatter(part["f1"], part["panel"].map(rows) + offsets[strategy], s=70, color=color, marker=marker,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)

    baseline = panel_summary.drop_duplicates("panel").set_index("panel")["all_positive_f1"]
    centers = np.array([rows[p] for p in panels])
    ax.vlines(baseline[panels], centers - 0.32, centers + 0.32, color=NEUTRAL, linewidth=2, zorder=2,
              label="Taban: hepsine patojenik demek")

    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.set_ylim(-0.6, len(panels) - 0.4)
    ax.set_xlim(0.80, 0.96)
    ax.set_xlabel("F1 (5 kat ortalaması)")
    ax.set_title("Nihai modeller: eşik stratejisine göre F1", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _save(fig, "aegis_v2_threshold_strategies.png")


def confusion_matrices(panel_summary: pd.DataFrame, strategy: str = "fixed_0.5") -> None:
    """Pooled confusion matrix per panel, shaded by the share within each true class."""
    data = panel_summary[panel_summary["strategy"] == strategy].set_index("panel")
    panels = [p for p in config.PANELS if p in data.index]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
    for ax, panel in zip(np.atleast_1d(axes), panels):
        row = data.loc[panel]
        counts = np.array([[row["total_tn"], row["total_fp"]], [row["total_fn"], row["total_tp"]]], dtype=float)
        shares = counts / counts.sum(axis=1, keepdims=True)

        ax.imshow(shares, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(counts[i, j])}\n%{shares[i, j] * 100:.0f}", ha="center", va="center",
                        color="white" if shares[i, j] > 0.55 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["Benign", "Patojenik"])
        ax.set_yticks([0, 1], ["Benign", "Patojenik"], rotation=90, va="center")
        ax.set_xlabel("Tahmin")
        ax.set_title(panel)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    np.atleast_1d(axes)[0].set_ylabel("Gerçek")

    fig.suptitle(f"Karmaşıklık matrisleri - {STRATEGY_STYLE[strategy][0]} (5 katın toplamı; yüzde = gerçek sınıf içindeki pay)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "aegis_v2_confusion_matrices.png")


def threshold_curves(curves: pd.DataFrame) -> None:
    """Mean F1, precision and recall over the folds as the threshold moves."""
    panels = [p for p in config.PANELS if p in set(curves["panel"])]
    series = [("f1", "F1", BLUE), ("precision", "Precision", ORANGE), ("recall", "Recall", AQUA)]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.0), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        mean = curves[curves["panel"] == panel].groupby("threshold")[[s[0] for s in series]].mean()
        for column, label, color in series:
            ax.plot(mean.index, mean[column], color=color, linewidth=2, label=label)
        ax.set_title(panel)
        ax.set_xlabel("Karar eşiği")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("Skor")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Karar eşiğinin etkisi (nihai modeller)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "aegis_v2_threshold_curves.png")


def precision_recall_curves(predictions: pd.DataFrame) -> None:
    """Precision-recall curve per panel from the pooled out-of-fold predictions."""
    panels = [p for p in config.PANELS if p in set(predictions["panel"])]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.0), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        part = predictions[predictions["panel"] == panel]
        precision, recall, _ = precision_recall_curve(part["y_true"], part["proba"])
        prevalence = part["y_true"].mean()

        ax.plot(recall, precision, color=BLUE, linewidth=2)
        ax.axhline(prevalence, color=NEUTRAL, linewidth=1.2)
        ax.text(0.02, prevalence - 0.008, f"rastgele: {_decimal(prevalence, 2)}",
                va="top", color=INK_SECONDARY, fontsize=9)
        ax.set_title(panel)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.6, 1.005)
    np.atleast_1d(axes)[0].set_ylabel("Precision")
    fig.suptitle("Precision-recall eğrileri (kat dışı tahminler)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "aegis_v2_pr_curves.png")


def feature_importance(top_features: pd.DataFrame, n: int = 12) -> None:
    """Most important features per panel (XGBoost gain share, mean over folds)."""
    panels = [p for p in config.PANELS if p in set(top_features["panel"])]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, panel in zip(axes.ravel(), panels):
        part = top_features[top_features["panel"] == panel].nlargest(n, "importance").iloc[::-1]
        ax.barh(part["feature"], part["importance"], height=0.6, color=BLUE)
        ax.set_title(panel)
        ax.set_xlabel("Önem (kat ortalaması)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("En önemli özellikler", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "aegis_v2_feature_importance.png")
