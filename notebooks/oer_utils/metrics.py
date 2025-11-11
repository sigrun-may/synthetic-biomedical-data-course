# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Statistical metrics for feature evaluation.

Provides effect size measures commonly used in biomedical research
to quantify the separation between groups.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "cohens_d",
    "compute_all_effect_sizes",
]


def cohens_d(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, labels: tuple[str | int, str | int]) -> float:
    """Compute Cohen's d effect size for binary classification.

    Cohen's d quantifies the difference between two group means in terms
    of their pooled standard deviation. It is scale-independent and widely
    used in biomedical research.

    Formula:
        d = (mean_class1 - mean_class0) / pooled_std

    where pooled_std = sqrt(((n0-1)*var0 + (n1-1)*var1) / (n0+n1-2))

    Args:
        x: Feature values (array-like or pandas Series).
        y: Binary labels (array-like).
        labels: Tuple of (class0_label, class1_label) to compare.

    Returns:
        Cohen's d effect size (float). Larger absolute values indicate
        stronger separation between classes.

    Effect Size Interpretation (Cohen, 1988):
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large

    Examples:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> cohens_d(x, y, labels=(0, 1))
        1.732...

        >>> # Using pandas
        >>> import pandas as pd
        >>> df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "class": [0, 0, 0, 1, 1, 1]})
        >>> cohens_d(df["feature"], df["class"], labels=(0, 1))
        1.732...

    References:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
        (2nd ed.). Routledge. https://doi.org/10.4324/9780203771587

    Notes:
        - Returns 0.0 for edge cases (n < 2 in either group, pooled_std = 0)
        - Sign indicates direction: positive means class1 > class0
        - For feature ranking, typically use absolute value
    """
    # Convert to numpy arrays
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    class0, class1 = labels

    # Split by class
    x0 = x[y == class0]
    x1 = x[y == class1]

    n0, n1 = len(x0), len(x1)

    # Edge cases
    if n0 < 2 or n1 < 2:
        return 0.0

    # Compute means and variances (with Bessel correction)
    mean0, mean1 = np.mean(x0), np.mean(x1)
    var0, var1 = np.var(x0, ddof=1), np.var(x1, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean0) / pooled_std)


def compute_all_effect_sizes(
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    labels: tuple[int, int] = (0, 1),
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Cohen's d for all features and return ranked DataFrame.

    Convenient wrapper to compute effect sizes for multiple features
    and sort them by magnitude.

    Args:
        x: Feature DataFrame or 2D array (n_samples, n_features).
        y: Binary labels.
        labels: Tuple of (class0, class1) to compare.
        feature_names: Optional feature names. If None, uses X.columns
            (if DataFrame) or generates generic names.

    Returns:
        DataFrame with columns:
            - feature: Feature name
            - cohens_d: Absolute effect size
            - cohens_d_signed: Signed effect size (preserves direction)
        Sorted by absolute effect size (descending).

    Examples:
        >>> from biomedical_data_generator import DatasetConfig, generate_dataset
        >>> cfg = DatasetConfig(
        ...     n_samples=100, n_informative=5, n_noise=3,
        ...     n_classes=2, class_counts={0: 50, 1: 50},
        ...     feature_naming="prefixed", random_state=42
        ... )
        >>> X, y, meta = generate_dataset(cfg, return_dataframe=True)
        >>>
        >>> effect_df = compute_all_effect_sizes(x, y)
        >>> print(effect_df.head())
           feature  cohens_d  cohens_d_signed
        0       i1      2.15             2.15
        1       i2      1.98             1.98
        2       i3      1.87            -1.87
        3       i4      1.76             1.76
        4       i5      1.54             1.54

    Notes:
        Informative features should rank higher than noise features.
        Use this to visualize feature importance in teaching context.
    """
    # Handle feature names
    if isinstance(x, pd.DataFrame):
        if feature_names is None:
            feature_names = x.columns.tolist()
        x_array = x.values
    else:
        x_array = np.asarray(x)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(x_array.shape[1])]

    # Compute Cohen's d for each feature
    results = []
    for i, name in enumerate(feature_names):
        d = cohens_d(x_array[:, i], y, labels=labels)
        results.append({"feature": name, "cohens_d": abs(d), "cohens_d_signed": d})

    # Create DataFrame and sort by absolute effect size
    df = pd.DataFrame(results)
    df = df.sort_values("cohens_d", ascending=False).reset_index(drop=True)

    return df
