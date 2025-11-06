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

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "make_batches",
    "apply_random_intercepts",
    "per_batch_stats",
    "mean_center_per_batch",
]


def make_batches(
    n_samples: int,
    n_batches: int,
    proportions: Optional[Sequence[float]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Create batch assignments for samples.

    Assigns each sample to one of n_batches groups. Useful for simulating
    site effects, instrument batches, or recruitment cohorts.

    Args:
        n_samples: Total number of samples to assign.
        n_batches: Number of distinct batches/groups.
        proportions: Optional relative sizes of batches. Must sum to 1 and
            have length n_batches. If None, creates equal-sized batches.
        random_state: Seed for reproducible shuffling.

    Returns:
        Array of shape (n_samples,) with integer labels in range [0, n_batches).

    Examples:
        >>> # Equal batches
        >>> batches = make_batches(n_samples=300, n_batches=3)
        >>> np.bincount(batches)
        array([100, 100, 100])

        >>> # Unbalanced: Site A (50%), Site B (30%), Site C (20%)
        >>> batches = make_batches(100, 3, proportions=[0.5, 0.3, 0.2], random_state=42)
        >>> dict(zip(*np.unique(batches, return_counts=True)))
        {0: 50, 1: 30, 2: 20}

    Notes:
        Samples are randomly shuffled, so batch labels are not contiguous.
        This simulates realistic data collection where batch membership
        is discovered during analysis, not determined by row order.
    """
    rng = np.random.default_rng(random_state)

    if proportions is None:
        # Even split (as even as possible)
        base = n_samples // n_batches
        remainder = n_samples % n_batches
        counts = np.array([base + (1 if i < remainder else 0) for i in range(n_batches)], dtype=int)
    else:
        proportions = np.asarray(proportions, dtype=float)
        if not np.isclose(proportions.sum(), 1.0):
            proportions = proportions / proportions.sum()

        counts = (proportions * n_samples).round().astype(int)

        # Fix rounding to match n_samples exactly
        diff = n_samples - counts.sum()
        if diff != 0:
            # Adjust largest batches first
            idx = np.argsort(-proportions)[: abs(diff)]
            counts[idx] += np.sign(diff)

    # Create labels: [0,0,..., 1,1,..., n_batches-1,...]
    labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])

    # Shuffle to simulate realistic data
    rng.shuffle(labels)
    return labels


def apply_random_intercepts(
    X: pd.DataFrame,
    groups: np.ndarray,
    sigma_b: float,
    cols: Optional[Sequence[str]] = None,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add batch-specific intercepts to features.

    Simulates systematic differences between batches, such as:
    - Different instruments producing consistent offsets
    - Lab-to-lab calibration differences
    - Cohort effects (e.g., samples collected at different times)

    For each batch g, draws a random intercept b_g ~ N(0, sigma_b^2)
    and adds it to all specified features for samples in that batch.

    Args:
        X: Feature DataFrame (n_samples, n_features).
        groups: Batch assignments for each sample (from make_batches).
        sigma_b: Standard deviation of batch effects. Larger values create
            stronger systematic differences between batches.
        cols: Feature columns to affect. If None, affects all columns.
        random_state: Seed for reproducible intercept generation.

    Returns:
        Tuple of (X_affected, batch_intercepts):
            - X_affected: Copy of X with batch effects applied
            - batch_intercepts: Array of shape (n_batches,) with the random
              intercept drawn for each batch

    Examples:
        >>> from biomedical_data_generator import DatasetConfig, generate_dataset
        >>> cfg = DatasetConfig(
        ...     n_samples=100, n_informative=5, n_noise=3,
        ...     n_classes=2, class_counts={0: 50, 1: 50},
        ...     feature_naming="prefixed", random_state=42
        ... )
        >>> X, y, meta = generate_dataset(cfg, return_dataframe=True)
        >>>
        >>> # Simulate 3 sites with moderate batch effect
        >>> batches = make_batches(len(X), n_batches=3, random_state=42)
        >>> X_batch, intercepts = apply_random_intercepts(
        ...     X, batches, sigma_b=0.5,
        ...     cols=meta.feature_names[:3],  # Only affect first 3 features
        ...     random_state=42
        ... )
        >>> print(f"Batch intercepts: {intercepts}")

    Notes:
        - This is an *additive* batch effect model
        - Real batch effects can be more complex (multiplicative, feature-specific)
        - Use this to teach detection and correction strategies
    """
    if cols is None:
        cols = list(X.columns)
    else:
        cols = list(cols)

    rng = np.random.default_rng(random_state)
    n_groups = int(np.max(groups)) + 1

    # Draw random intercept for each batch
    batch_intercepts = rng.normal(loc=0.0, scale=float(sigma_b), size=n_groups)

    X_new = X.copy()

    # Vectorized addition: for each sample, add b[batch_of_sample]
    intercept_per_sample = np.take(batch_intercepts, groups)
    X_new[cols] = X_new[cols].add(intercept_per_sample[:, None], axis=0)

    return X_new, batch_intercepts


def per_batch_stats(X: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Compute summary statistics per batch.

    Useful for diagnosing whether batch effects exist and quantifying
    their magnitude.

    Args:
        X: Feature DataFrame.
        groups: Batch assignments.
        cols: Feature columns to analyze.

    Returns:
        DataFrame with columns:
            - batch: Batch identifier
            - {feature}_mean: Mean of feature in this batch
            - {feature}_std: Standard deviation of feature in this batch
            - {feature}_count: Number of samples in this batch

    Examples:
        >>> X_batch, _ = apply_random_intercepts(X, batches, sigma_b=0.5)
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
    df = X.loc[:, cols].copy()
    df["__batch__"] = groups

    # Compute mean, std, count per batch
    agg = df.groupby("__batch__").agg(["mean", "std", "count"])

    # Flatten MultiIndex columns: (feature, stat) -> feature_stat
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index().rename(columns={"__batch__": "batch"})

    return agg


def mean_center_per_batch(X: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Remove batch-wise means (simple batch correction).

    For each feature, subtracts the batch-specific mean. This is a naive
    correction that works well for additive batch effects but has limitations.

    Args:
        X: Feature DataFrame.
        groups: Batch assignments.
        cols: Feature columns to correct.

    Returns:
        Copy of X with specified columns mean-centered per batch.

    Examples:
        >>> # Apply batch effect
        >>> X_batch, _ = apply_random_intercepts(X, batches, sigma_b=0.5, cols=["i1"])
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
    df = X.copy()
    batch_series = pd.Series(groups, index=df.index, name="batch")

    for col in cols:
        # Subtract batch-specific mean for this feature
        df[col] = df[col] - df.groupby(batch_series)[col].transform("mean")

    return df
