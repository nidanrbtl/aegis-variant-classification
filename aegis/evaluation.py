"""Cross-validation protocol, metrics and decision-threshold selection.

All experiments share the same protocol:

* ``StratifiedGroupKFold`` grouped by ``Variant_ID``, so a variant that occurs
  in several rows can never be in the training and the validation fold at the
  same time (checked explicitly for every fold).
* Metrics are computed per fold and averaged afterwards.

Three ways of choosing the decision threshold are supported, and it matters
which one a reported number comes from:

``fixed``          0.5 for every panel. No tuning, no optimism.
``nested``         chosen on out-of-fold predictions *inside the training
                   part* of each fold (:func:`nested_threshold`). Unbiased.
``per-fold best``  the F1-optimal threshold of the validation fold itself
                   (:func:`best_f1_threshold` on the validation labels).
                   This uses the labels it is evaluated on, so it is an
                   optimistic upper bound, not an estimate of real performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from . import config

ModelFactory = Callable[[], object]


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
@dataclass
class Fold:
    """One train/validation split of the grouped cross-validation."""

    number: int
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    val_index: np.ndarray
    group_overlap: int = 0  # always 0; kept in the result tables as evidence


def make_splitter(n_splits: int = config.N_SPLITS) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def assert_no_group_overlap(groups: pd.Series, train_idx, val_idx) -> None:
    """Fail loudly if a Variant_ID appears on both sides of a split."""
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
    if overlap:
        raise RuntimeError(f"Variant_ID leakage detected: {len(overlap)} shared ids.")


def drop_constant_columns(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """Remove columns that carry a single value in the training fold."""
    n_unique = X_train.nunique(dropna=False)
    keep = n_unique[n_unique > 1].index.tolist()
    return X_train[keep], X_val[keep]


def cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = config.N_SPLITS,
    drop_constant: bool = False,
) -> Iterator[Fold]:
    """Yield the leakage-checked folds of the shared validation protocol."""
    splitter = make_splitter(n_splits)
    for number, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        assert_no_group_overlap(groups, train_idx, val_idx)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        if drop_constant:
            X_train, X_val = drop_constant_columns(X_train, X_val)

        yield Fold(
            number=number,
            X_train=X_train,
            y_train=y.iloc[train_idx],
            groups_train=groups.iloc[train_idx],
            X_val=X_val,
            y_val=y.iloc[val_idx],
            val_index=val_idx,
        )


def fit_predict(make_model: ModelFactory, fold: Fold):
    """Fit a fresh model on the training part and score the validation part."""
    model = make_model()
    model.fit(fold.X_train, fold.y_train)
    return model, model.predict_proba(fold.X_val)[:, 1]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def threshold_metrics(y_true, proba, threshold: float) -> dict:
    """Metrics that depend on the decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def ranking_metrics(y_true, proba) -> dict:
    """Threshold-free metrics; NaN when a fold contains a single class."""
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": np.nan, "roc_auc": np.nan}
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def classification_metrics(y_true, proba, threshold: float = config.DEFAULT_THRESHOLD) -> dict:
    return {**threshold_metrics(y_true, proba, threshold), **ranking_metrics(y_true, proba)}


def all_positive_f1(y_true) -> float:
    """F1 of the trivial classifier that calls every variant pathogenic.

    The panels are 69-83 % pathogenic, so this baseline is already high and is
    the number every model has to be compared against.
    """
    return f1_score(y_true, np.ones(len(y_true), dtype=int), zero_division=0)


# --------------------------------------------------------------------------
# Decision thresholds
# --------------------------------------------------------------------------
def best_f1_threshold(y_true, proba) -> float:
    """Threshold on the grid with the highest F1 (the lowest one on ties)."""
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in config.THRESHOLD_GRID]
    return config.THRESHOLD_GRID[int(np.argmax(scores))]


def threshold_curve(y_true, proba) -> pd.DataFrame:
    """Threshold-dependent metrics for every threshold on the grid."""
    rows = [{"threshold": t, **threshold_metrics(y_true, proba, t)} for t in config.THRESHOLD_GRID]
    return pd.DataFrame(rows)


def nested_threshold(X_train, y_train, groups_train, make_model: ModelFactory) -> float:
    """Pick a threshold using only the training part of a fold.

    The training rows are split again (grouped by Variant_ID), every row gets
    an out-of-fold probability from a model that has not seen it, and the
    F1-optimal threshold on those probabilities is returned. The validation
    fold plays no part, so metrics computed with this threshold are unbiased.
    """
    n_splits = min(config.INNER_SPLITS, int(y_train.value_counts().min()))
    if n_splits < 2:
        return config.DEFAULT_THRESHOLD

    oof = np.zeros(len(y_train))
    for train_idx, val_idx in make_splitter(n_splits).split(X_train, y_train, groups_train):
        model = make_model()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

    return best_f1_threshold(y_train, oof)


def fixed_vs_best(y_true, proba) -> dict:
    """F1, balanced accuracy and MCC at 0.5 and at the fold's own best-F1 threshold."""
    fixed = threshold_metrics(y_true, proba, config.DEFAULT_THRESHOLD)
    threshold = best_f1_threshold(y_true, proba)
    best = threshold_metrics(y_true, proba, threshold)
    return {
        "f1_0_5": fixed["f1"],
        "balanced_accuracy_0_5": fixed["balanced_accuracy"],
        "mcc_0_5": fixed["mcc"],
        "best_threshold": threshold,
        "best_f1": best["f1"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "best_mcc": best["mcc"],
    }
