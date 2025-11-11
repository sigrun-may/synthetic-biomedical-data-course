# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Evaluation utilities for OER notebooks.

This module provides functions for:
- Comparing different cross-validation schemes
- Evaluating multiple models
- Computing and comparing CV performance metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import cross_validate

__all__ = [
    "compare_cv_schemes",
    "evaluate_multiple_models",
    "summarize_cv_results",
    "compute_cv_performance_gap",
]


# ============================================================================
# Cross-Validation Comparison
# ============================================================================


def compare_cv_schemes(
    x: NDArray[Any] | pd.DataFrame,
    y: NDArray[Any] | pd.Series,
    pipeline: Any,
    cv_configs: dict[str, Any],
    groups: NDArray[Any] | pd.Series | None = None,
    scoring: dict[str, str] | str = "balanced_accuracy",
    n_jobs: int = -1,
    return_train_score: bool = False,
) -> pd.DataFrame:
    """Compare different cross-validation schemes side-by-side.

    Runs the same pipeline with multiple CV splitters and returns
    a tidy DataFrame with results from all schemes.

    Args:
        x: Feature matrix (array or DataFrame).
        y: Target vector (array or Series).
        pipeline: sklearn Pipeline or estimator to evaluate.
        cv_configs: Dict mapping scheme name -> CV splitter.
            Example: {"naive": StratifiedKFold(5), "grouped": GroupKFold(4)}
        groups: Group labels for GroupKFold (optional, only used by grouped schemes).
        scoring: Scoring metrics (dict or single string).
        n_jobs: Number of parallel jobs for cross_validate.
        return_train_score: Whether to include training scores.

    Returns:
        DataFrame with columns:
            - scheme: Name of CV scheme
            - metric: Metric name (e.g., "balanced_accuracy", "roc_auc")
            - mean: Mean score across folds
            - std: Standard deviation across folds
            - n_folds: Number of folds used
            - fold_scores: List of per-fold scores (optional)

    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import StratifiedKFold, GroupKFold
        >>>
        >>> pipe = Pipeline([
        ...     ("scaler", StandardScaler()),
        ...     ("clf", LogisticRegression(max_iter=1000))
        ... ])
        >>>
        >>> cv_configs = {
        ...     "naive": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        ...     "grouped": GroupKFold(n_splits=4)
        ... }
        >>>
        >>> results = compare_cv_schemes(
        ...     x, y, pipe, cv_configs,
        ...     groups=batch_id,
        ...     scoring={"bal_acc": "balanced_accuracy", "roc_auc": "roc_auc"}
        ... )
        >>> print(results)
    """
    # Ensure X and y are arrays for sklearn compatibility
    x_arr = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    if groups is not None:
        groups_arr = groups.values if isinstance(groups, pd.Series) else np.asarray(groups)
    else:
        groups_arr = None

    # Normalize scoring to dict format
    scoring_dict = {scoring: scoring} if isinstance(scoring, str) else scoring

    results = []

    for scheme_name, cv_splitter in cv_configs.items():
        # Determine if this splitter uses groups
        cv_class_name = cv_splitter.__class__.__name__
        uses_groups = "Group" in cv_class_name

        # Run cross-validation
        cv_results = cross_validate(
            pipeline,
            x_arr,
            y_arr,
            cv=cv_splitter,
            groups=groups_arr if uses_groups else None,
            scoring=scoring_dict,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )

        # Extract number of folds
        n_folds = cv_splitter.get_n_splits(x_arr, y_arr, groups=groups_arr if uses_groups else None)

        # Parse results for each metric
        for metric_key, metric_name in scoring_dict.items():
            test_key = f"test_{metric_key}"
            test_scores = cv_results[test_key]

            row = {
                "scheme": scheme_name,
                "metric": metric_name,
                "mean": float(np.mean(test_scores)),
                "std": float(np.std(test_scores, ddof=1)),
                "n_folds": n_folds,
            }

            # Optionally include fold-level scores
            if return_train_score:
                train_key = f"train_{metric_key}"
                train_scores = cv_results[train_key]
                row["train_mean"] = float(np.mean(train_scores))
                row["train_std"] = float(np.std(train_scores, ddof=1))

            results.append(row)

    return pd.DataFrame(results)


def summarize_cv_results(
    cv_results: dict[str, NDArray[np.floating]],
    scheme_label: str = "CV",
) -> pd.DataFrame:
    """Summarize results from sklearn cross_validate into a tidy DataFrame.

    Args:
        cv_results: Output from sklearn.model_selection.cross_validate.
        scheme_label: Label for the CV scheme (e.g., "naive", "grouped").

    Returns:
        DataFrame with columns: scheme, metric, mean, std.

    Examples:
        >>> from sklearn.model_selection import cross_validate, StratifiedKFold
        >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        >>> results = cross_validate(
        ...     model, X, y, cv=cv,
        ...     scoring={"bal_acc": "balanced_accuracy", "roc_auc": "roc_auc"}
        ... )
        >>> summary = summarize_cv_results(results, scheme_label="naive")
    """
    rows = []
    for key, scores in cv_results.items():
        if key.startswith("test_"):
            metric_name = key.replace("test_", "")
            rows.append(
                {
                    "scheme": scheme_label,
                    "metric": metric_name,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# Model Comparison
# ============================================================================


def evaluate_multiple_models(
    models: dict[str, Any],
    x: NDArray[Any] | pd.DataFrame,
    y: NDArray[Any] | pd.Series,
    cv: Any,
    groups: NDArray[Any] | pd.Series | None = None,
    scoring: dict[str, str] | str = "balanced_accuracy",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate multiple models with the same CV scheme.

    Args:
        models: Dict mapping model name -> sklearn estimator or pipeline.
        x: Feature matrix.
        y: Target vector.
        cv: Cross-validation splitter.
        groups: Group labels (for GroupKFold).
        scoring: Scoring metrics (dict or single string).
        n_jobs: Parallel jobs for cross_validate.

    Returns:
        DataFrame with columns:
            - model: Model name
            - metric: Metric name
            - mean: Mean score across folds
            - std: Standard deviation across folds

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import GroupKFold
        >>>
        >>> models = {
        ...     "LogReg": LogisticRegression(max_iter=1000),
        ...     "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        ... }
        >>>
        >>> results = evaluate_multiple_models(
        ...     models, x, y,
        ...     cv=GroupKFold(n_splits=4),
        ...     groups=batch_id,
        ...     scoring={"bal_acc": "balanced_accuracy"}
        ... )
    """
    # Ensure arrays
    x_arr = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    groups_arr = (
        groups.values if isinstance(groups, pd.Series) else (np.asarray(groups) if groups is not None else None)
    )

    # Normalize scoring
    scoring_dict = {scoring: scoring} if isinstance(scoring, str) else scoring

    results = []

    for model_name, model in models.items():
        cv_results = cross_validate(
            model,
            x_arr,
            y_arr,
            cv=cv,
            groups=groups_arr,
            scoring=scoring_dict,
            n_jobs=n_jobs,
        )

        for metric_key, metric_name in scoring_dict.items():
            test_key = f"test_{metric_key}"
            test_scores = cv_results[test_key]
            results.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "mean": float(np.mean(test_scores)),
                    "std": float(np.std(test_scores, ddof=1)),
                }
            )

    return pd.DataFrame(results)


# ============================================================================
# Performance Gap Analysis
# ============================================================================


def compute_cv_performance_gap(
    comparison_df: pd.DataFrame,
    baseline_scheme: str,
    test_scheme: str,
    metric: str = "balanced_accuracy",
) -> dict[str, float | str]:
    """Compute performance gap between two CV schemes.

    Useful for quantifying optimistic bias when comparing naive CV
    (e.g., StratifiedKFold) vs. group-aware CV (e.g., GroupKFold).

    Args:
        comparison_df: Output from compare_cv_schemes.
        baseline_scheme: Name of the baseline scheme (e.g., "naive").
        test_scheme: Name of the test scheme (e.g., "grouped").
        metric: Metric to compare (must be in comparison_df).

    Returns:
        Dictionary with:
            - baseline_mean: Mean score for baseline scheme
            - test_mean: Mean score for test scheme
            - absolute_gap: baseline - test (positive = optimistic baseline)
            - relative_gap_pct: 100 * absolute_gap / baseline
            - metric: Metric name

    Examples:
        >>> gap = compute_cv_performance_gap(
        ...     results, "naive", "grouped", metric="balanced_accuracy"
        ... )
        >>> print(f"Optimistic bias: {gap['absolute_gap']:.3f}")
    """
    df_filtered = comparison_df[comparison_df["metric"] == metric]

    if baseline_scheme not in df_filtered["scheme"].values:
        raise ValueError(f"Scheme '{baseline_scheme}' not found for metric '{metric}'.")
    if test_scheme not in df_filtered["scheme"].values:
        raise ValueError(f"Scheme '{test_scheme}' not found for metric '{metric}'.")

    baseline_mean = float(df_filtered[df_filtered["scheme"] == baseline_scheme]["mean"].iloc[0])
    test_mean = float(df_filtered[df_filtered["scheme"] == test_scheme]["mean"].iloc[0])

    absolute_gap = baseline_mean - test_mean
    relative_gap_pct = 100.0 * absolute_gap / baseline_mean if baseline_mean != 0 else 0.0

    return {
        "baseline_mean": baseline_mean,
        "test_mean": test_mean,
        "absolute_gap": absolute_gap,
        "relative_gap_pct": relative_gap_pct,
        "metric": metric,
        "baseline_scheme": baseline_scheme,
        "test_scheme": test_scheme,
    }
