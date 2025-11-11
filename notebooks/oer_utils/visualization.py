# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Plotting utilities for OER notebooks.

This module provides reusable visualization functions for:
- Feature distributions by class
- Cross-validation performance comparisons
- Feature ranking visualizations
- Effect size visualizations
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

__all__ = [
    "plot_feature_distributions_by_class",
    "plot_cv_comparison",
    "plot_effect_sizes",
    "plot_performance_gap",
]


# ============================================================================
# Feature Distribution Plots
# ============================================================================


def plot_feature_distributions_by_class(
    x: pd.DataFrame,
    y: pd.Series | NDArray[np.integer],
    features: Sequence[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[int, int] | None = None,
    kind: str = "hist",
    **kwargs: Any,
) -> tuple[Figure, NDArray[Any]]:
    """Create grid of distribution plots for features, colored by class.

    Args:
        x: Feature matrix (DataFrame).
        y: Binary target vector (0/1 labels).
        features: Subset of features to plot (None = all).
        n_cols: Number of columns in subplot grid.
        figsize: Figure size (auto-calculated if None).
        kind: Plot type ("hist", "kde", or "violin").
        **kwargs: Additional arguments passed to plotting function.

    Returns:
        Tuple of (figure, axes_array).

    Examples:
        >>> fig, axes = plot_feature_distributions_by_class(
        ...     x, y, features=["i1", "i2", "n1"], kind="kde"
        ... )
        >>> plt.tight_layout()
        >>> plt.show()
    """
    features = list(x.columns) if features is None else list(features)

    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    y_arr = np.asarray(y)

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]
        data = pd.DataFrame({feature: x[feature].values, "class": y_arr})

        if kind == "hist":
            sns.histplot(
                data=data,
                x=feature,
                hue="class",
                stat="density",
                element="step",
                common_norm=False,
                alpha=0.4,
                kde=True,
                ax=ax,
                **kwargs,
            )
        elif kind == "kde":
            sns.kdeplot(
                data=data,
                x=feature,
                hue="class",
                common_norm=False,
                fill=True,
                alpha=0.4,
                ax=ax,
                **kwargs,
            )
        elif kind == "violin":
            sns.violinplot(
                data=data,
                x="class",
                y=feature,
                ax=ax,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown plot kind: {kind}. Use 'hist', 'kde', or 'violin'.")

        ax.set_title(feature)
        ax.set_ylabel("Density" if kind in ["hist", "kde"] else feature)

    # Hide unused subplots
    for idx in range(n_features, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    return fig, axes if n_features > 1 else axes_flat[0]


# ============================================================================
# Cross-Validation Comparison Plots
# ============================================================================


def plot_cv_comparison(
    results_df: pd.DataFrame,
    metric: str | None = None,
    kind: str = "bar",
    figsize: tuple[int, int] = (8, 5),
    show_std: bool = True,
) -> tuple[Figure, Axes]:
    """Plot cross-validation performance comparison between schemes.

    Args:
        results_df: DataFrame from compare_cv_schemes or evaluate_multiple_models.
        metric: Metric to visualize (None = all metrics in separate subplots).
        kind: Plot type ("bar" or "point").
        figsize: Figure size.
        show_std: Whether to show error bars (± std).

    Returns:
        Tuple of (figure, axes).

    Examples:
        >>> fig, ax = plot_cv_comparison(
        ...     results, metric="balanced_accuracy", kind="bar"
        ... )
        >>> plt.show()
    """
    if metric is not None:
        df_plot = results_df[results_df["metric"] == metric].copy()
        if len(df_plot) == 0:
            raise ValueError(f"No results found for metric: {metric}")
    else:
        df_plot = results_df.copy()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if kind == "bar":
        x_col = "scheme" if "scheme" in df_plot.columns else "model"
        sns.barplot(
            data=df_plot,
            x=x_col,
            y="mean",
            hue="metric" if metric is None else None,
            ax=ax,
            errorbar=None,  # We'll add custom error bars
        )

        # Add error bars manually if requested
        if show_std and "std" in df_plot.columns:
            x_positions = np.arange(len(df_plot))
            ax.errorbar(
                x_positions,
                df_plot["mean"],
                yerr=df_plot["std"],
                fmt="none",
                c="black",
                capsize=4,
            )

    elif kind == "point":
        x_col = "scheme" if "scheme" in df_plot.columns else "model"
        sns.pointplot(
            data=df_plot,
            x=x_col,
            y="mean",
            hue="metric" if metric is None else None,
            ax=ax,
            errorbar="sd" if show_std else None,
            capsize=0.1,
        )
    else:
        raise ValueError(f"Unknown plot kind: {kind}. Use 'bar' or 'point'.")

    ax.set_ylabel("Mean Score")
    ax.set_xlabel("CV Scheme" if "scheme" in df_plot.columns else "Model")
    title = f"CV Performance: {metric}" if metric else "CV Performance Comparison"
    ax.set_title(title)

    if metric is None:
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    return fig, ax


def plot_performance_gap(
    gap_dict: dict[str, float],
    figsize: tuple[int, int] = (6, 4),
) -> tuple[Figure, Axes]:
    """Visualize performance gap between CV schemes.

    Args:
        gap_dict: Output from compute_cv_performance_gap.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Examples:
        >>> from oer_utils.evaluation import compute_cv_performance_gap
        >>> gap = compute_cv_performance_gap(results, "naive", "grouped")
        >>> fig, ax = plot_performance_gap(gap)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    schemes = [gap_dict["baseline_scheme"], gap_dict["test_scheme"]]
    means = [gap_dict["baseline_mean"], gap_dict["test_mean"]]

    ax.bar(schemes, means, color=["#d62728", "#2ca02c"], alpha=0.7)

    # Annotate gap
    gap_val = gap_dict["absolute_gap"]
    gap_pct = gap_dict["relative_gap_pct"]

    ax.axhline(gap_dict["baseline_mean"], color="gray", linestyle="--", alpha=0.5)
    ax.text(
        0.5,
        max(means) * 0.95,
        f"Gap: {gap_val:+.3f} ({gap_pct:+.1f}%)",
        ha="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_ylabel("Mean Score")
    ax.set_title(f"Performance Gap: {gap_dict['metric']}")
    ax.set_ylim(0, max(means) * 1.1)

    return fig, ax


def plot_effect_sizes(
    effect_size_df: pd.DataFrame,
    top_k: int | None = None,
    figsize: tuple[int, int] = (8, 6),
    color_by_type: bool = True,
) -> tuple[Figure, Axes]:
    """Plot effect sizes for all features, sorted by magnitude.

    Args:
        effect_size_df: DataFrame from rank_features_by_effect_size.
        top_k: Show only top-k features (None = all).
        figsize: Figure size.
        color_by_type: Color bars by feature prefix (i/n/p/corr).

    Returns:
        Tuple of (figure, axes).

    Examples:
        >>> fig, ax = plot_effect_sizes(ranked_df, top_k=15, color_by_type=True)
        >>> plt.show()
    """
    df = effect_size_df.head(top_k).copy() if top_k is not None else effect_size_df.copy()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Determine colors based on feature prefix
    if color_by_type:
        colors = []
        for feat in df["feature"]:
            if feat.startswith("i"):
                colors.append("#1f77b4")  # blue for informative
            elif feat.startswith("n"):
                colors.append("#d62728")  # red for noise
            elif feat.startswith("p"):
                colors.append("#ff7f0e")  # orange for pseudo
            elif feat.startswith("corr"):
                colors.append("#9467bd")  # purple for correlated
            else:
                colors.append("#7f7f7f")  # gray for others
    else:
        colors = ["#1f77b4"] * len(df)

    y_positions = range(len(df))
    ax.barh(y_positions, df["|effect_size|"].values, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["feature"].values)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title(f"Effect Sizes (Top {len(df)})")
    ax.invert_yaxis()

    # Add legend if colored by type
    if color_by_type:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Informative"),
            Patch(facecolor="#d62728", label="Noise"),
            Patch(facecolor="#ff7f0e", label="Pseudo"),
            Patch(facecolor="#9467bd", label="Correlated"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    return fig, ax
