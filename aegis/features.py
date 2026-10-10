"""Feature engineering shared by every experiment.

The raw competition columns are anonymised and come in four families
(``AL_``, ``EK_``, ``CAT_``, ``AA_``). Each ``*_features`` function below turns
one family into model-ready numeric columns; :func:`build_features` concatenates
them. Missing values are left as NaN on purpose: the tree models handle them
natively and "not reported" is itself informative for genomic annotations.

Note on pandas: string columns keep missing values as NaN under the pandas 3
string dtype, which is what the category and amino-acid logic below relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import feature_columns, target_and_groups

FAMILIES = ("AL", "EK", "AA", "CAT", "MISS")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale.
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino-acid side-chain volumes (cubic angstroms).
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Only the diagonal of BLOSUM62 (self-substitution scores).
# KNOWN LIMITATION: the data contains no synonymous changes, so this lookup
# never matches and ``AA_blosum_simple`` is empty. See README, "Bilinen
# sınırlamalar".
BLOSUM_DIAGONAL = {
    ("A", "A"): 4, ("R", "R"): 5, ("N", "N"): 6, ("D", "D"): 6,
    ("C", "C"): 9, ("Q", "Q"): 5, ("E", "E"): 5, ("G", "G"): 6,
    ("H", "H"): 8, ("I", "I"): 4, ("L", "L"): 4, ("K", "K"): 5,
    ("M", "M"): 5, ("F", "F"): 6, ("P", "P"): 7, ("S", "S"): 4,
    ("T", "T"): 5, ("W", "W"): 11, ("Y", "Y"): 7, ("V", "V"): 4,
}

# In-silico scores that live on a 0-1 scale and are thresholded like PP3 evidence.
PROBABILITY_SCORE_COLS = ("EK_4", "EK_5", "EK_6")
TOP_CATEGORIES = 10


def _family_columns(df: pd.DataFrame, family: str) -> list[str]:
    prefix = config.FAMILY_PREFIXES[family]
    return [c for c in df.columns if c.startswith(prefix)]


def _to_frame(columns: dict, index: pd.Index) -> pd.DataFrame:
    """Build a frame in one go (avoids the column-by-column fragmentation warning)."""
    return pd.DataFrame(columns, index=index)


# --------------------------------------------------------------------------
# One function per feature family
# --------------------------------------------------------------------------
def al_features(df: pd.DataFrame) -> pd.DataFrame:
    """Population statistics: binary flags stay as they are, everything else
    becomes ``-log10(value)`` so that rarer variants get larger values."""
    out = {}
    for col in _family_columns(df, "AL"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col]
        observed = values.dropna()
        if observed.empty:
            continue
        if set(observed.unique()).issubset({0, 1}):
            out[col] = values
        else:
            out[f"log_{col}"] = -np.log10(values.clip(lower=1e-50))
    return _to_frame(out, df.index)


def ek_features(df: pd.DataFrame) -> pd.DataFrame:
    """In-silico scores plus simple agreement statistics between predictors."""
    out = {col: df[col] for col in _family_columns(df, "EK")}

    score_cols = [c for c in PROBABILITY_SCORE_COLS if c in df.columns]
    if score_cols:
        scores = df[score_cols]
        out["EK_score_mean"] = scores.mean(axis=1)
        out["EK_score_max"] = scores.max(axis=1)
        out["EK_score_min"] = scores.min(axis=1)
        out["EK_score_std"] = scores.std(axis=1)
        out["EK_PP3_count_05"] = (scores > 0.5).sum(axis=1)
        out["EK_PP3_count_07"] = (scores > 0.7).sum(axis=1)

    if "EK_7" in df.columns:
        out["EK_7_high_20"] = (df["EK_7"] > 20).astype(int)
        out["EK_7_high_30"] = (df["EK_7"] > 30).astype(int)

    return _to_frame(out, df.index)


def aa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physico-chemical consequences of the amino-acid substitution."""
    if "AA_1" not in df.columns or "AA_2" not in df.columns:
        return pd.DataFrame(index=df.index)

    ref = df["AA_1"].astype(str).str.strip()
    alt = df["AA_2"].astype(str).str.strip()
    pairs = list(zip(ref, alt))

    def difference(scale: dict) -> list[float]:
        return [scale[b] - scale[a] if a in scale and b in scale else np.nan for a, b in pairs]

    out = {
        "AA_is_nonsense": (alt == "*").astype(int),
        "AA_is_synonymous": (ref == alt).astype(int),
        "AA_ref_standard": ref.isin(STANDARD_AA).astype(int),
        "AA_alt_standard": alt.isin(STANDARD_AA).astype(int),
        "AA_hydro_diff": difference(HYDROPHOBICITY),
        "AA_volume_diff": difference(VOLUME),
        "AA_blosum_simple": [
            BLOSUM_DIAGONAL.get((a, b), BLOSUM_DIAGONAL.get((b, a), np.nan))
            if a in STANDARD_AA and b in STANDARD_AA
            else np.nan
            for a, b in pairs
        ],
        "AA_to_proline": (alt == "P").astype(int),
        "AA_from_proline": (ref == "P").astype(int),
        "AA_cys_loss": ((ref == "C") & (alt != "C")).astype(int),
        "AA_gly_loss": ((ref == "G") & (alt != "G")).astype(int),
    }
    return _to_frame(out, df.index)


def cat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Missing flag plus one-hot columns for the most frequent categories."""
    out = {}
    for col in _family_columns(df, "CAT"):
        labels = df[col].astype(str)
        out[f"{col}_missing"] = df[col].isna().astype(int)

        for value in labels.value_counts(dropna=True).head(TOP_CATEGORIES).index:
            clean = str(value).replace("/", "_").replace(".", "dot").replace(" ", "_")
            out[f"{col}_is_{clean}"] = (labels == value).astype(int)
    return _to_frame(out, df.index)


def missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Share of missing cells per row, overall and per feature family."""
    out = {"MISS_total_ratio": df[feature_columns(df)].isna().mean(axis=1)}
    for family in ("AL", "EK", "CAT", "AA"):
        cols = _family_columns(df, family)
        out[f"MISS_{family}_ratio"] = df[cols].isna().mean(axis=1) if cols else 0
    return _to_frame(out, df.index)


_FAMILY_BUILDERS = {
    "AL": al_features,
    "EK": ek_features,
    "AA": aa_features,
    "CAT": cat_features,
    "MISS": missing_features,
}


# --------------------------------------------------------------------------
# Assembling feature matrices
# --------------------------------------------------------------------------
def _numeric_only(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.select_dtypes(include=["number", "bool"])
    return frame.replace([np.inf, -np.inf], np.nan)


def build_feature_families(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Engineered features, one frame per family (used by the ablation study)."""
    return {name: _numeric_only(builder(df)) for name, builder in _FAMILY_BUILDERS.items()}


def combine_families(families: dict[str, pd.DataFrame], selected=FAMILIES) -> pd.DataFrame:
    """Concatenate the chosen families in the canonical order."""
    return pd.concat([families[name] for name in FAMILIES if name in selected], axis=1)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Full engineered feature matrix, labels and Variant_ID groups."""
    X = combine_families(build_feature_families(df))
    y, groups = target_and_groups(df)
    return X, y, groups


def raw_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Baseline feature set: the numeric competition columns, untouched."""
    X = df[feature_columns(df)].select_dtypes(include=["number", "bool"]).copy()
    y, groups = target_and_groups(df)
    return X, y, groups


def feature_family(feature: str) -> str:
    """Human-readable family of a feature name (used in importance summaries)."""
    if feature.startswith(("log_AL", "AL_")):
        return "AL / Population Frequency"
    if feature.startswith("EK_"):
        return "EK / In-silico Evidence"
    if feature.startswith("AA_"):
        return "AA / Protein Impact"
    if feature.startswith("MISS_"):
        return "Missingness Evidence"
    if feature.startswith("CAT_"):
        return "CAT / Genomic Context"
    if feature.startswith("ACMG_"):
        return "ACMG-like Evidence"
    if feature.startswith("AEGIS_"):
        return "AEGIS Combined Evidence"
    return "Other"
