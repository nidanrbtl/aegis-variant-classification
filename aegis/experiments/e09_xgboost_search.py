"""Step 09 - randomised hyperparameter search for XGBoost, one winner per panel.

Each sampled configuration is scored with a 3-fold grouped CV at threshold
0.5; the configuration with the highest mean F1 wins. The winners are frozen
in ``config.FINAL_XGB_PARAMS`` for the final models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from .. import config
from ..data import load_panel
from ..evaluation import classification_metrics, cv_folds, fit_predict
from ..features import build_features
from ..models import make_xgb
from ..outputs import banner, save_table, show


def score_params(X, y, groups, params: dict) -> dict:
    f1, mcc, pr_auc = [], [], []
    for fold in cv_folds(X, y, groups, n_splits=config.SEARCH_SPLITS):
        _, proba = fit_predict(lambda: make_xgb(params), fold)
        metrics = classification_metrics(fold.y_val, proba, config.DEFAULT_THRESHOLD)
        f1.append(metrics["f1"])
        mcc.append(metrics["mcc"])
        pr_auc.append(metrics["pr_auc"])

    return {
        "mean_f1": float(np.mean(f1)),
        "std_f1": float(np.std(f1)),
        "mean_mcc": float(np.mean(mcc)),
        "mean_pr_auc": float(np.nanmean(pr_auc)),
    }


def run() -> None:
    banner("STEP 09 | XGBOOST RANDOMISED HYPERPARAMETER SEARCH")

    candidates = list(ParameterSampler(
        config.XGB_SEARCH_SPACE,
        n_iter=config.N_SEARCH_TRIALS,
        random_state=config.RANDOM_STATE,
    ))

    rows = []
    for panel in config.PANELS:
        X, y, groups = build_features(load_panel(panel))

        for trial, params in enumerate(candidates, start=1):
            scores = score_params(X, y, groups, params)
            rows.append({"panel": panel, "trial": trial, **params, **scores, "variant_id_overlap": 0})
            print(f"{panel} trial {trial:02d}: F1 {scores['mean_f1']:.4f}, "
                  f"MCC {scores['mean_mcc']:.4f}, PR-AUC {scores['mean_pr_auc']:.4f}")

    trials = pd.DataFrame(rows)
    winners = (
        trials.sort_values(["panel", "mean_f1"], ascending=[True, False])
        .groupby("panel")
        .head(1)
        .reset_index(drop=True)
    )

    save_table(trials, "aegis_09_xgboost_hyperparameter_search_all_trials.csv")
    save_table(winners, "aegis_09_xgboost_best_params_by_panel.csv")

    show(winners)
