"""Unit tests on small synthetic data (no competition data needed).

Run with:  python -m unittest discover -s tests
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from aegis import config
from aegis.evaluation import (
    all_positive_f1,
    assert_no_group_overlap,
    best_f1_threshold,
    cv_folds,
    drop_constant_columns,
    nested_threshold,
    threshold_metrics,
)
from aegis.evidence import add_evidence_features
from aegis.features import aa_features, al_features, build_features, cat_features


def synthetic_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A tiny table with the same column families as the competition data."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    amino = list("ACDEFGHIKLMNPQRSTVWY")

    df = pd.DataFrame({
        "Variant_ID": [f"VAR_{i // 2:04d}" for i in range(n)],  # every id appears twice
        "AL_1": rng.uniform(1e-7, 0.2, n),   # frequency-like
        "AL_2": rng.integers(0, 2, n),       # binary flag
        "CAT_1": rng.choice(["x/y", "z", None], n),
        "AA_1": rng.choice(amino + [None], n),
        "AA_2": rng.choice(amino + ["*", None], n),
        "Label": label,
    })
    for i in range(1, 10):
        df[f"EK_{i}"] = rng.normal(label, 1.0)
    df.loc[rng.choice(n, 15, replace=False), "AL_1"] = np.nan
    # both rows of a variant carry the same label, as in the real data
    df["Label"] = df.groupby("Variant_ID")["Label"].transform("first")
    return df


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_panel()

    def test_feature_matrix_is_numeric_and_aligned(self):
        X, y, groups = build_features(self.df)
        self.assertEqual(len(X), len(self.df))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t) for t in X.dtypes))
        self.assertFalse(np.isinf(X.to_numpy(dtype=float)).any())
        self.assertEqual(len(y), len(groups))

    def test_allele_columns(self):
        al = al_features(self.df)
        self.assertIn("log_AL_1", al.columns)   # continuous -> -log10
        self.assertIn("AL_2", al.columns)       # binary -> unchanged
        observed = self.df["AL_1"].notna()
        np.testing.assert_allclose(al.loc[observed, "log_AL_1"], -np.log10(self.df.loc[observed, "AL_1"]))
        self.assertTrue(al.loc[~observed, "log_AL_1"].isna().all())

    def test_missing_amino_acid_is_not_a_synonymous_change(self):
        aa = aa_features(self.df)
        both_missing = self.df["AA_1"].isna() & self.df["AA_2"].isna()
        self.assertTrue(both_missing.any())
        self.assertTrue((aa.loc[both_missing, "AA_is_synonymous"] == 0).all())
        self.assertTrue(aa.loc[self.df["AA_1"].isna(), "AA_hydro_diff"].isna().all())

    def test_category_columns(self):
        cat = cat_features(self.df)
        self.assertIn("CAT_1_missing", cat.columns)
        self.assertIn("CAT_1_is_x_y", cat.columns)       # "/" is replaced
        self.assertNotIn("CAT_1_is_nan", cat.columns)    # missing is not a category
        self.assertEqual(int(cat["CAT_1_missing"].sum()), int(self.df["CAT_1"].isna().sum()))

    def test_evidence_features_for_every_panel(self):
        X, _, _ = build_features(self.df)
        for panel in config.PANELS:
            with self.subTest(panel=panel):
                out = add_evidence_features(X, panel)
                added = [c for c in out.columns if c not in X.columns]
                self.assertEqual(list(out.columns[: X.shape[1]]), list(X.columns))
                self.assertIn("ACMG_PM2_population_rare_mean", added)
                self.assertIn(f"AEGIS_{panel}_evidence_strength", added)
                self.assertEqual(len(out), len(X))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_panel()
        self.X, self.y, self.groups = build_features(self.df)

    def test_folds_never_share_a_variant(self):
        seen = []
        for fold in cv_folds(self.X, self.y, self.groups):
            val_groups = set(self.groups.iloc[fold.val_index])
            self.assertFalse(val_groups & set(fold.groups_train))
            seen.extend(fold.val_index)
        self.assertEqual(sorted(seen), list(range(len(self.X))))  # every row validated once

    def test_group_overlap_is_detected(self):
        with self.assertRaises(RuntimeError):
            assert_no_group_overlap(self.groups, [0, 2], [1, 3])  # rows 0 and 1 share an id

    def test_drop_constant_columns(self):
        train = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [np.nan] * 3})
        val = pd.DataFrame({"a": [5], "b": [6], "c": [7.0]})
        kept_train, kept_val = drop_constant_columns(train, val)
        self.assertEqual(list(kept_train.columns), ["b"])
        self.assertEqual(list(kept_val.columns), ["b"])

    def test_best_threshold_and_metrics(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.10, 0.40, 0.45, 0.90])
        threshold = best_f1_threshold(y, proba)
        self.assertTrue(0.40 < threshold <= 0.45)
        metrics = threshold_metrics(y, proba, threshold)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (2, 2, 0, 0))
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_all_positive_baseline(self):
        self.assertAlmostEqual(all_positive_f1(np.array([1, 1, 1, 0])), 2 * 0.75 / 1.75)

    def test_nested_threshold_uses_training_data_only(self):
        fold = next(cv_folds(self.X, self.y, self.groups))
        threshold = nested_threshold(
            fold.X_train.fillna(0), fold.y_train, fold.groups_train,
            lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
        )
        self.assertTrue(any(np.isclose(threshold, config.THRESHOLD_GRID)))


if __name__ == "__main__":
    unittest.main()
