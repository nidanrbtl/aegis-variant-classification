"""AEGIS - panel-based pathogenicity classification of missense variants.

Developed for the TEKNOFEST "Sağlıkta Yapay Zekâ" competition.
"""

import pandas as _pd

if int(_pd.__version__.split(".")[0]) < 3:
    # The feature code relies on the pandas 3 string dtype, where missing text
    # stays NaN. Under pandas 2 it becomes the string "nan" and silently
    # changes several features, so refuse to run rather than drift.
    raise ImportError(f"aegis requires pandas >= 3.0 (found {_pd.__version__}).")

__version__ = "2.0.0"
