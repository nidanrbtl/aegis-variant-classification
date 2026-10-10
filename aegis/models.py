"""Model factories. Every call returns a fresh, unfitted estimator."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from . import config


def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **config.LGBM_PARAMS,
    )


def make_xgb(params: dict | None = None) -> XGBClassifier:
    """XGBoost classifier; ``params`` defaults to the untuned base configuration."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        **(config.XGB_BASE_PARAMS if params is None else params),
    )


def make_random_forest():
    # Random forests and linear models cannot take NaN, hence the imputer.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    )


def make_logistic_regression():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
    )


# Candidates compared in the model-selection experiment.
CANDIDATE_MODELS = {
    "LightGBM": make_lgbm,
    "XGBoost": make_xgb,
    "RandomForest": make_random_forest,
    "LogisticRegression": make_logistic_regression,
}
