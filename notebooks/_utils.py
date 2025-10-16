# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Small reusable helpers for notebooks."""

from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

# -------------------- Reproducibility --------------------

def set_seed(seed: int) -> None:
    """Set RNG seed for numpy and random."""
    random.seed(seed)
    np.random.seed(seed)

# -------------------- Metadata card --------------------

def data_card(cfg, meta) -> dict:
    """Return a small dict capturing config + meta info for reproducibility."""
    d = {
        "n_samples": getattr(cfg, "n_samples", None),
        "n_informative": getattr(cfg, "n_informative", None),
        "n_pseudo": getattr(cfg, "n_pseudo", None),
        "n_noise": getattr(cfg, "n_noise", None),
        "n_classes": getattr(cfg, "n_classes", None),
        "class_sep": getattr(cfg, "class_sep", None),
        "feature_naming": getattr(cfg, "feature_naming", None),
        "random_state": getattr(cfg, "random_state", None),
    }
    # Try to enrich from meta if present
    for key in ("informative_idx", "pseudo_idx", "noise_idx"):
        if hasattr(meta, key):
            val = getattr(meta, key)
            d[f"{key}_count"] = None if val is None else int(len(val))
    return d

# -------------------- Batches & random intercepts --------------------

def make_batches(n_samples: int, n_batches: int, proportions: Optional[Sequence[float]] = None,
                 random_state: Optional[int] = None) -> np.ndarray:
    """Return an array of length n_samples with batch labels 0..n_batches-1.
    If proportions is given, it must sum to 1 and have length n_batches.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    if proportions is None:
        # Even split (as even as possible)
        base = n_samples // n_batches
        rem = n_samples % n_batches
        counts = np.array([base + (1 if i < rem else 0) for i in range(n_batches)], dtype=int)
    else:
        proportions = np.asarray(proportions, dtype=float)
        proportions = proportions / proportions.sum()
        counts = (proportions * n_samples).round().astype(int)
        # Fix rounding to match n_samples exactly
        diff = n_samples - counts.sum()
        if diff != 0:
            idx = np.argsort(-proportions)[:abs(diff)]
            counts[idx] += np.sign(diff)

    labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])
    rng.shuffle(labels)
    return labels


def apply_random_intercepts(
    X: pd.DataFrame,
    groups: np.ndarray,
    sigma_b: float,
    cols: Optional[Sequence[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Add batch-specific intercept b_g ~ N(0, sigma_b^2) to selected columns.

    Returns (X_new, b_by_group).
    """
    if cols is None:
        cols = list(X.columns)
    cols = list(cols)

    rng = np.random.default_rng(random_state)
    n_groups = int(np.max(groups)) + 1
    b = rng.normal(loc=0.0, scale=float(sigma_b), size=n_groups)

    X_new = X.copy()
    # Fast vectorized add: for each group, add b[g] to cols
    add_vec = np.take(b, groups)
    X_new[cols] = X_new[cols].add(add_vec[:, None].astype(X_new[cols].dtypes[0]), axis=0)
    return X_new, b


def per_batch_stats(X: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Compute per-batch mean and std for selected columns."""
    df = X.loc[:, cols].copy()
    df["__batch__"] = groups
    agg = df.groupby("__batch__").agg(["mean", "std", "count"])  # MultiIndex columns
    # Flatten columns
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index().rename(columns={"__batch__": "batch"})
    return agg


def mean_center_per_batch(X: pd.DataFrame, groups: np.ndarray, cols: Sequence[str]) -> pd.DataFrame:
    """Return a copy with each column mean-centered per batch (simple correction)."""
    df = X.copy()
    g = pd.Series(groups, index=df.index, name="batch")
    for c in cols:
        df[c] = df[c] - df.groupby(g)[c].transform("mean")
    return df

# -------------------- Optional: small stats helper --------------------

def cohen_d(x0: Iterable[float], x1: Iterable[float]) -> float:
    x0 = np.asarray(list(x0), dtype=float)
    x1 = np.asarray(list(x1), dtype=float)
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return 0.0
    m0, m1 = x0.mean(), x1.mean()
    s0, s1 = x0.std(ddof=1), x1.std(ddof=1)
    s_p = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    return 0.0 if s_p == 0 else (m1 - m0) / s_p
