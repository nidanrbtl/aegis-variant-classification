"""Project-wide settings: paths, panels, random seeds and model hyperparameters.

Everything that an experiment might want to change lives here, so the
experiment modules contain no magic numbers of their own.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Both locations can be redirected with environment variables, which keeps the
# competition data outside the repository if needed.
DATA_DIR = Path(os.environ.get("AEGIS_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("AEGIS_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
# Row-level out-of-fold predictions. Never committed (see .gitignore).
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
PANELS = ("MASTER", "KANSER", "PAH", "CFTR")

DATA_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

ID_COL = "Variant_ID"
TARGET_COL = "Label"  # 1 = pathogenic, 0 = benign

# Column-name prefixes of the four anonymised feature families.
FAMILY_PREFIXES = {
    "AL": "AL_",    # population / allele statistics
    "EK": "EK_",    # in-silico prediction scores
    "CAT": "CAT_",  # categorical genomic context
    "AA": "AA_",    # reference and alternate amino acid
}

# --------------------------------------------------------------------------
# Validation protocol
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5          # outer StratifiedGroupKFold, grouped by Variant_ID
INNER_SPLITS = 3      # inner folds used to pick a threshold without leakage
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.01, 1.00, 0.01)

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 25,
}

# Untuned XGBoost used for model selection (step 07) and the first
# threshold study (step 08).
XGB_BASE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
}

# Randomised search space of step 09.
XGB_SEARCH_SPACE = {
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [2, 3, 4, 5],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.75, 0.85, 1.0],
    "colsample_bytree": [0.75, 0.85, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 2, 5],
}
N_SEARCH_TRIALS = 12
SEARCH_SPLITS = 3

# Final per-panel hyperparameters (AEGIS v2). These are the winners of the
# randomised search in step 09 at the time of the competition, frozen here so
# the final models do not depend on a file produced by an earlier run. The
# best candidates lie within the fold-to-fold noise of each other (the top two
# for MASTER were 0.0001 F1 apart), so re-running the search with a different
# XGBoost version can crown a different winner.
FINAL_XGB_PARAMS = {
    "MASTER": {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    "KANSER": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.75,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
    },
    "PAH": {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 1.0,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
    "CFTR": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
}


def ensure_output_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (TABLE_DIR, FIGURE_DIR, REPORT_DIR, PREDICTION_DIR):
        folder.mkdir(parents=True, exist_ok=True)
