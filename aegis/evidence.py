"""ACMG-inspired "evidence" features of the final AEGIS v2 models.

The ACMG/AMP guideline combines independent lines of evidence (how rare a
variant is in the population, what in-silico predictors say, how severe the
amino-acid change is). The functions below condense the engineered feature
matrix into a handful of such summary scores per variant. Every panel gets
the shared population and in-silico block plus a few panel-specific scores.

The scores are heuristics computed on anonymised columns; they are *inspired*
by ACMG criteria (PM2, BA1, PP3) and are not an implementation of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Thresholds on the -log10 scale produced by ``features.al_features``.
RARE = 6      # value below 1e-6   -> supports PM2 (absent / extremely rare)
ABSENT = 8    # value below 1e-8
COMMON = 2    # value above 1e-2   -> supports BA1 (too common to be pathogenic)

# In-silico scores treated as conservation-like evidence for the cancer panel.
CONSERVATION_COLS = ("EK_2", "EK_3", "EK_7", "EK_9")


def _columns(X: pd.DataFrame, *prefixes: str) -> pd.DataFrame:
    return X[[c for c in X.columns if c.startswith(prefixes)]]


def _population_evidence(al: pd.DataFrame) -> dict:
    return {
        "ACMG_PM2_population_rare_mean": (al > RARE).mean(axis=1),
        "ACMG_PM2_population_rare_max": (al > RARE).max(axis=1).astype(int),
        "ACMG_BA1_population_common_any": (al < COMMON).max(axis=1).astype(int),
    }


def _insilico_evidence(ek: pd.DataFrame) -> dict:
    return {
        "ACMG_PP3_insilico_mean": ek.mean(axis=1),
        "ACMG_PP3_insilico_max": ek.max(axis=1),
        # number of scores above their median across the panel
        "ACMG_PP3_insilico_support_count": (ek > ek.median()).sum(axis=1),
    }


def _shared_block(X: pd.DataFrame, population_extra: dict | None = None) -> dict:
    """Population + in-silico evidence, common to all four panels."""
    al = _columns(X, "log_AL", "AL_")
    new = _population_evidence(al)
    if population_extra:
        new.update({name: fn(al) for name, fn in population_extra.items()})
    new.update(_insilico_evidence(_columns(X, "EK_")))
    return new


def _master(X: pd.DataFrame) -> dict:
    new = _shared_block(X, {"ACMG_population_evidence_density": lambda al: al.notna().mean(axis=1)})
    pm2, pp3 = new["ACMG_PM2_population_rare_mean"], new["ACMG_PP3_insilico_mean"]

    new["ACMG_COMBINED_PM2_PP3"] = pm2 * pp3
    new["AEGIS_MASTER_evidence_strength"] = pm2 + pp3 - new["ACMG_BA1_population_common_any"]
    return new


def _kanser(X: pd.DataFrame) -> dict:
    new = _shared_block(X, {"AEGIS_population_absence_strength": lambda al: (al > ABSENT).mean(axis=1)})
    pm2, pp3 = new["ACMG_PM2_population_rare_mean"], new["ACMG_PP3_insilico_mean"]

    # Conservation / damage scores are made more visible for the cancer panel.
    conservation = X[[c for c in CONSERVATION_COLS if c in X.columns]]
    cons = conservation.mean(axis=1)
    new["AEGIS_KANSER_conservation_mean"] = cons
    new["AEGIS_KANSER_conservation_max"] = conservation.max(axis=1)
    new["AEGIS_KANSER_conservation_support_count"] = (conservation > conservation.median()).sum(axis=1)

    new["AEGIS_KANSER_missing_evidence_density"] = _columns(X, "MISS_").mean(axis=1)

    new["ACMG_COMBINED_PM2_PP3"] = pm2 * pp3
    new["AEGIS_KANSER_combined_conservation_evidence"] = pm2 * pp3 * cons
    new["AEGIS_KANSER_evidence_strength"] = pm2 + pp3 + cons - new["ACMG_BA1_population_common_any"]
    return new


def _pah(X: pd.DataFrame) -> dict:
    new = _shared_block(X)
    pm2, pp3 = new["ACMG_PM2_population_rare_mean"], new["ACMG_PP3_insilico_mean"]

    # Biochemical severity of the amino-acid substitution.
    aa = _columns(X, "AA_")
    aa_impact = aa.abs().mean(axis=1)
    new["AEGIS_PAH_aa_impact_mean"] = aa.mean(axis=1)
    new["AEGIS_PAH_aa_impact_abs_mean"] = aa_impact
    new["AEGIS_PAH_aa_impact_max"] = aa.abs().max(axis=1)
    new["AEGIS_PAH_hydrophobicity_shift"] = X["AA_hydro_diff"].abs()
    new["AEGIS_PAH_volume_shift"] = X["AA_volume_diff"].abs()
    new["AEGIS_PAH_proline_disruption"] = X["AA_to_proline"].fillna(0) + X["AA_from_proline"].fillna(0)
    new["AEGIS_PAH_cysteine_loss"] = X["AA_cys_loss"].fillna(0)
    new["AEGIS_PAH_glycine_loss"] = X["AA_gly_loss"].fillna(0)

    new["AEGIS_PAH_missing_evidence_density"] = _columns(X, "MISS_").mean(axis=1)

    new["ACMG_COMBINED_PM2_PP3"] = pm2 * pp3
    new["AEGIS_PAH_combined_biochemical_evidence"] = pm2 * pp3 * aa_impact
    new["AEGIS_PAH_evidence_strength"] = pm2 + pp3 + aa_impact - new["ACMG_BA1_population_common_any"]
    return new


def _cftr(X: pd.DataFrame) -> dict:
    new = _shared_block(X, {"AEGIS_CFTR_population_signal_density": lambda al: al.notna().mean(axis=1)})
    pm2, pp3 = new["ACMG_PM2_population_rare_mean"], new["ACMG_PP3_insilico_mean"]

    aa = _columns(X, "AA_")
    aa_impact = aa.abs().mean(axis=1)
    new["AEGIS_CFTR_aa_impact_abs_mean"] = aa_impact
    new["AEGIS_CFTR_aa_impact_max"] = aa.abs().max(axis=1)

    # The missingness pattern is summarised as evidence of its own.
    missing = _columns(X, "MISS_")
    miss = missing.mean(axis=1)
    new["AEGIS_CFTR_missing_density"] = miss
    new["AEGIS_CFTR_missing_max"] = missing.max(axis=1)
    new["AEGIS_CFTR_missing_support_count"] = (missing > missing.median()).sum(axis=1)

    new["ACMG_COMBINED_PM2_PP3"] = pm2 * pp3
    new["AEGIS_CFTR_channel_evidence_score"] = pm2 * pp3 * (1 + aa_impact)
    new["AEGIS_CFTR_missing_adjusted_evidence"] = (pm2 + pp3 + aa_impact) * (1 + miss)
    new["AEGIS_CFTR_evidence_strength"] = pm2 + pp3 + aa_impact + miss - new["ACMG_BA1_population_common_any"]
    return new


_PANEL_BUILDERS = {"MASTER": _master, "KANSER": _kanser, "PAH": _pah, "CFTR": _cftr}


def add_evidence_features(X: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Append the evidence scores of ``panel`` to the engineered feature matrix."""
    new = pd.DataFrame(_PANEL_BUILDERS[panel](X), index=X.index)
    return pd.concat([X, new], axis=1).replace([np.inf, -np.inf], np.nan)
