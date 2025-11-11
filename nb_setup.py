"""
Common notebook setup: plotting style + pandas display.

Usage in notebooks:
    from nb_setup import apply_style
    apply_style()
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def apply_style() -> None:
    """Apply consistent visual style and display options across all notebooks."""
    # 1) Matplotlib style (.mplstyle in repo)
    style_path = pathlib.Path(__file__).parent / "styles" / "nb_style.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))

    # 2) Seaborn theme (optional; no hard dependency)
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    except Exception:
        # Seaborn not installed; Matplotlib style already applied
        pass

    # 3) Pandas display options
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 120)
