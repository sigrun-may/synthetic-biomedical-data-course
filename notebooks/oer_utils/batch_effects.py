# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tools for simulating and correcting batch effects.

This module provides functions to:
1. Create batch assignments (sites, instruments, recruitment years)
2. Apply batch effects (random intercepts, systematic shifts)
3. Diagnose batch effects (compute per-batch statistics)
4. Correct batch effects (simple mean centering)

Essential for Lesson 03c: Pseudo-classes and Batch Effects
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "per_batch_stats",
    "mean_center_per_batch",
]


def per_batch_stats(x: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Compute summary statistics per batch.

    Useful for diagnosing whether batch effects exist and quantifying
    their magnitude.

    Args:
        x: Feature DataFrame.
        groups: Batch assignments.
        cols: Feature columns to analyze.

    Returns:
        DataFrame with columns:
            - batch: Batch identifier
            - {feature}_mean: Mean of feature in this batch
            - {feature}_std: Standard deviation of feature in this batch
            - {feature}_count: Number of samples in this batch

    Examples:
        >>> X_batch, _ = apply_random_intercepts(x, batches, sigma_b=0.5)
        >>> stats = per_batch_stats(X_batch, batches, cols=["i1", "i2"])
        >>> print(stats)
           batch  i1_mean  i1_std  i1_count  i2_mean  i2_std  i2_count
        0      0     0.52    0.98        33     0.48    1.02        33
        1      1    -0.31    1.01        34    -0.29    0.95        34
        2      2     0.05    0.99        33     0.02    1.03        33

    Notes:
        If batches differ strongly in their means but have similar stds,
        this suggests additive batch effects (what apply_random_intercepts creates).
    """
    df = x.loc[:, cols].copy()
    df["__batch__"] = groups

    # Compute mean, std, count per batch
    agg = df.groupby("__batch__").agg(["mean", "std", "count"])

    # Flatten MultiIndex columns: (feature, stat) -> feature_stat
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index().rename(columns={"__batch__": "batch"})

    return agg


def mean_center_per_batch(x: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Remove batch-wise means (simple batch correction).

    For each feature, subtracts the batch-specific mean. This is a naive
    correction that works well for additive batch effects but has limitations.

    Args:
        x: Feature DataFrame.
        groups: Batch assignments.
        cols: Feature columns to correct.

    Returns:
        Copy of X with specified columns mean-centered per batch.

    Examples:
        >>> # Apply batch effect
        >>> X_batch, _ = apply_random_intercepts(x, batches, sigma_b=0.5, cols=["i1"])
        >>>
        >>> # Correct it
        >>> X_corrected = mean_center_per_batch(X_batch, batches, cols=["i1"])
        >>>
        >>> # Check: batch means should now be ~0
        >>> stats_before = per_batch_stats(X_batch, batches, cols=["i1"])
        >>> stats_after = per_batch_stats(X_corrected, batches, cols=["i1"])
        >>> print(stats_after["i1_mean"])  # All close to 0

    Warnings:
        This correction assumes:
        - Additive batch effects only (no multiplicative or interaction effects)
        - Batches have similar biological distributions
        - No confounding between batch and biology

        For real data, consider more sophisticated methods like ComBat.

    Notes:
        In teaching context, use this to show:
        1. Simple correction can help
        2. But it's not always sufficient (e.g., if batch confounded with class)
    """
    df = x.copy()
    batch_series = pd.Series(groups, index=df.index, name="batch")

    for col in cols:
        # Subtract batch-specific mean for this feature
        df[col] = df[col] - df.groupby(batch_series)[col].transform("mean")

    return df
