"""Command line entry point: ``python -m aegis <step> [<step> ...]``."""

from __future__ import annotations

import argparse
import importlib
import time

# step name -> (module in aegis.experiments, one-line description)
STEPS = {
    "data-check": ("e01_data_check", "size, class balance and label consistency of the raw data"),
    "baseline": ("e02_lgbm_baseline", "LightGBM on raw numeric columns"),
    "threshold": ("e03_threshold_optimization", "effect of the decision threshold on the baseline"),
    "features": ("e04_feature_engineering", "LightGBM on the engineered features"),
    "ablation": ("e05_family_ablation", "feature-family ablation study"),
    "nested-threshold": ("e06_nested_threshold", "leakage-free threshold selection"),
    "model-selection": ("e07_model_selection", "LightGBM / XGBoost / Random Forest / Logistic Regression"),
    "xgb-threshold": ("e08_xgboost_threshold", "threshold study for untuned XGBoost"),
    "xgb-search": ("e09_xgboost_search", "randomised hyperparameter search per panel"),
    "xgb-tuned": ("e10_tuned_xgboost", "tuned XGBoost without evidence features"),
    "final": ("e11_final_models", "final AEGIS v2 models with evidence features"),
    "report": ("e12_report", "figures and markdown summary of the final models"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aegis",
        description="Run AEGIS experiments. Steps run in the order given; 'all' runs every step.",
        epilog="steps:\n" + "\n".join(f"  {name:<17} {text}" for name, (_, text) in STEPS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("steps", nargs="+", choices=[*STEPS, "all"], metavar="step")
    args = parser.parse_args()

    selected = list(STEPS) if "all" in args.steps else args.steps
    for name in selected:
        module = importlib.import_module(f"aegis.experiments.{STEPS[name][0]}")
        started = time.perf_counter()
        module.run()
        print(f"\n[{name}] finished in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
