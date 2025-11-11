# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Feature selection and stability utilities for OER notebooks.

This module provides functions for:
- Ranking features by effect size or importance scores
- Computing feature stability across CV folds
- Analyzing overlap between different feature rankings
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .metrics import cohens_d

# ============================================================================
# Feature Ranking
# ============================================================================


def rank_features_by_effect_size(
    x: pd.DataFrame,
    y: pd.Series | NDArray[np.integer],
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank features by effect size magnitude.

    Computes effect size for each feature and returns a sorted DataFrame
    with feature names, effect sizes, absolute effect sizes, and ranks.

    Args:
        x: Feature matrix (DataFrame with column names).
        y: Binary target vector (0/1 labels).
        ascending: Sort order (False = largest |effect| first).

    Returns:
        DataFrame with columns:
            - feature: Feature name
            - effect_size: Raw effect size value
            - |effect_size|: Absolute effect size
            - rank: Integer rank (1 = best)
    """
    y_arr = np.asarray(y)
    # preserve order of first occurrence of labels
    labels = list(dict.fromkeys(y_arr.tolist()))
    if len(labels) != 2:
        raise ValueError("y must contain exactly two distinct labels.")
    if (y_arr == labels[0]).sum() < 2 or (y_arr == labels[1]).sum() < 2:
        raise ValueError("Each class needs at least 2 samples for Cohen's d (ddof=1).")

    results = []
    for col in x.columns:
        values = x[col].values
        es = cohens_d(values, y_arr, labels=(labels[0], labels[1]))
        results.append({"feature": col, "|effect_size|": abs(es)})

    df = pd.DataFrame(results)
    df = df.sort_values("|effect_size|", ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df[["feature", "|effect_size|", "rank"]]


def rank_features_by_importance(
    feature_names: Sequence[str],
    importance_scores: NDArray[np.floating] | Sequence[float],
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank features by importance scores from a fitted model.

    Args:
        feature_names: List or array of feature names.
        importance_scores: Importance scores (e.g., coef_, feature_importances_).
        ascending: Sort order (False = largest importance first).

    Returns:
        DataFrame with columns:
            - feature: Feature name
            - importance: Raw importance score
            - |importance|: Absolute importance
            - rank: Integer rank (1 = best)

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression().fit(X_train, y_train)
        >>> ranked = rank_features_by_importance(
        ...     X_train.columns,
        ...     model.coef_[0],
        ... )
    """
    if len(feature_names) != len(importance_scores):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} features vs " f"{len(importance_scores)} importance scores."
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_scores,
            "|importance|": np.abs(importance_scores),
        }
    )
    df = df.sort_values("|importance|", ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df[["feature", "importance", "|importance|", "rank"]]


# ============================================================================
# Stability & Overlap Metrics
# ============================================================================


def compute_jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two feature sets.

    Args:
        set_a: First set of feature names.
        set_b: Second set of feature names.

    Returns:
        Jaccard index in [0, 1]: |A ∩ B| / |A ∪ B|.

    Examples:
        >>> j = compute_jaccard_similarity({"i1", "i2", "n1"}, {"i1", "i3", "n1"})
        >>> print(f"{j:.2f}")  # 0.50
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def compute_precision_at_k(
    ranked_list_1: Sequence[str],
    ranked_list_2: Sequence[str],
    k: int,
) -> float:
    """Compute Precision@k overlap between two ranked feature lists.

    Args:
        ranked_list_1: First ranking (ordered feature names).
        ranked_list_2: Second ranking (ordered feature names).
        k: Number of top features to consider.

    Returns:
        Precision@k in [0, 1]: (overlap in top-k) / k.

    Examples:
        >>> r1 = ["i1", "i2", "n3", "i3"]
        >>> r2 = ["i1", "n4", "i2", "i3"]
        >>> p = compute_precision_at_k(r1, r2, k=2)
        >>> print(p)  # 1.0 (both have i1, i2 in top-2)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    top_k_1 = set(ranked_list_1[:k])
    top_k_2 = set(ranked_list_2[:k])
    overlap = top_k_1 & top_k_2
    return len(overlap) / k


def compute_feature_stability(
    feature_rankings: Sequence[Sequence[str]],
    top_k: int | None = None,
    method: str = "jaccard",
) -> dict[str, float | str | int | None]:
    """Compute stability metrics across multiple feature rankings.

    This function computes pairwise similarity between all rankings
    and returns the mean similarity as a stability measure.

    Args:
        feature_rankings: List of feature rankings (each a list/array of names).
        top_k: Consider only top-k features (None = all features).
        method: Stability metric ("jaccard" or "precision_at_k").

    Returns:
        Dictionary with:
            - stability_mean: Mean pairwise similarity
            - stability_std: Standard deviation of pairwise similarities
            - n_pairs: Number of ranking pairs compared
            - method: Method used

    Examples:
        >>> rankings = [
        ...     ["i1", "i2", "n3", "i3"],
        ...     ["i1", "n4", "i2", "i3"],
        ...     ["i2", "i1", "n5", "i3"],
        ... ]
        >>> stability = compute_feature_stability(rankings, top_k=3)
        >>> print(f"Stability: {stability['stability_mean']:.3f}")
    """
    if len(feature_rankings) < 2:
        raise ValueError("Need at least 2 rankings to compute stability.")

    # Truncate to top-k if requested
    if top_k is not None:
        rankings_truncated = [list(r[:top_k]) for r in feature_rankings]
    else:
        rankings_truncated = [list(r) for r in feature_rankings]

    similarities = []
    n_rankings = len(rankings_truncated)

    for i in range(n_rankings):
        for j in range(i + 1, n_rankings):
            if method == "jaccard":
                sim = compute_jaccard_similarity(set(rankings_truncated[i]), set(rankings_truncated[j]))
            elif method == "precision_at_k":
                if top_k is None:
                    raise ValueError("precision_at_k requires top_k to be specified.")
                sim = compute_precision_at_k(rankings_truncated[i], rankings_truncated[j], top_k)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'jaccard' or 'precision_at_k'.")
            similarities.append(sim)

    return {
        "stability_mean": float(np.mean(similarities)),
        "stability_std": float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0,
        "n_pairs": len(similarities),
        "method": method,
        "top_k": top_k,
    }


def compare_feature_rankings(
    ranking_a: pd.DataFrame,
    ranking_b: pd.DataFrame,
    top_k: int | None = None,
    label_a: str = "Ranking A",
    label_b: str = "Ranking B",
) -> pd.DataFrame:
    """Compare two feature rankings side-by-side.

    Args:
        ranking_a: First ranking (from rank_features_by_*).
        ranking_b: Second ranking (from rank_features_by_*).
        top_k: Show only top-k features (None = all).
        label_a: Label for first ranking.
        label_b: Label for second ranking.

    Returns:
        DataFrame with columns:
            - rank: Shared rank index
            - feature_a: Feature from ranking A
            - feature_b: Feature from ranking B
            - in_both: Boolean indicating if feature appears in both top-k

    Examples:
        >>> cmp = compare_feature_rankings(
        ...     rank_a, rank_b, top_k=5,
        ...     label_a="Naive", label_b="Batch-corrected"
        ... )
    """
    if top_k is not None:
        a = ranking_a.head(top_k).copy()
        b = ranking_b.head(top_k).copy()
    else:
        a = ranking_a.copy()
        b = ranking_b.copy()

    max_len = max(len(a), len(b))

    # Pad shorter ranking with None
    if len(a) < max_len:
        padding = pd.DataFrame({"feature": [None] * (max_len - len(a))})
        a = pd.concat([a, padding], ignore_index=True)
    if len(b) < max_len:
        padding = pd.DataFrame({"feature": [None] * (max_len - len(b))})
        b = pd.concat([b, padding], ignore_index=True)

    comparison = pd.DataFrame(
        {
            "rank": range(1, max_len + 1),
            f"feature_{label_a}": a["feature"].values,
            f"feature_{label_b}": b["feature"].values,
        }
    )

    # Mark features present in both rankings
    set_a = set(a["feature"].dropna())
    set_b = set(b["feature"].dropna())
    comparison["in_both"] = [
        (fa in set_b if pd.notna(fa) else False) and (fb in set_a if pd.notna(fb) else False)
        for fa, fb in zip(comparison[f"feature_{label_a}"], comparison[f"feature_{label_b}"], strict=False)
    ]

    return comparison
