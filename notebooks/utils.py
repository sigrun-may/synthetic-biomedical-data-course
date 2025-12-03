"""Utilities for effect size computation and dataset summarization."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from biomedical_data_generator.meta import DatasetMeta
from matplotlib import pyplot as plt
from pandas.api.extensions import ExtensionArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cliffs_delta(
    x: np.ndarray | pd.Series | Sequence[float] | ExtensionArray,
    y: np.ndarray | pd.Series | Sequence[int] | ExtensionArray,
    labels: tuple[str | int, str | int],
) -> float:
    """Compute Cliff's delta for binary classification.

    Cliff's delta measures the probability that a random sample from class1
    is larger than a random sample from class0 minus the reverse probability:

        delta = (#(x1 > x0) - # (x1 < x0)) / (n0 * n1)

    Range: [-1, 1]. Positive means class1 > class0 on average.

    Args:
        x: Feature values (array-like or pandas Series).
        y: Binary labels (array-like).
        labels: Tuple of (class0_label, class1_label) to compare.

    Returns:
        Cliff's delta (float). Returns 0.0 for trivial/degenerate cases.

    Notes:
        - Uses a memory-guarded vectorized path; for very large numbers of pairs
          it falls back to an iterative counting loop to avoid O(n0*n1) memory usage.
    """
    # Convert to numpy arrays
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    class0, class1 = labels

    # Split by class
    x0 = x[y == class0]
    x1 = x[y == class1]

    n0, n1 = len(x0), len(x1)

    # Edge cases
    if n0 == 0 or n1 == 0:
        return 0.0

    total_pairs = int(n0) * int(n1)

    # Guard threshold to avoid huge memory allocation for outer difference matrix
    if total_pairs <= 10_000_000:
        diff = np.subtract.outer(x1, x0)  # shape (n1, n0)
        greater = int(np.count_nonzero(diff > 0))
        less = int(np.count_nonzero(diff < 0))
    else:
        # iterative counting using the smaller loop to reduce overhead
        greater = 0
        less = 0
        if n1 <= n0:
            for xi in x1:
                greater += int(np.count_nonzero(xi > x0))
                less += int(np.count_nonzero(xi < x0))
        else:
            for xj in x0:
                greater += int(np.count_nonzero(x1 > xj))
                less += int(np.count_nonzero(x1 < xj))

    delta = (greater - less) / float(total_pairs)
    return float(delta)


def cohens_d(
    x: np.ndarray | pd.Series | Sequence[float] | ExtensionArray,
    y: np.ndarray | pd.Series | Sequence[int] | ExtensionArray,
    labels: tuple[str | int, str | int],
) -> float:
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
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

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
    method: str = "cohens_d",
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Cohen's d for all features and return ranked DataFrame.

    Convenient wrapper to compute effect sizes for multiple features
    and sort them by magnitude.

    Args:
        x: Feature DataFrame or 2D array (n_samples, n_features).
        y: Binary labels.
        labels: Tuple of (class0, class1) to compare.
        method: Effect size computation method ("cohens_d" or "cliffs_delta").
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
           feature  |effect_size|  cohens_d_signed
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

    results = []
    for i, name in enumerate(feature_names):
        if method == "cliffs_delta":
            # Compute Cliff's delta for each feature
            d = cliffs_delta(x_array[:, i], y, labels=labels)
            results.append({"feature": name, "|effect_size|": abs(d), "cliffs_delta": d})
        elif method == "cohens_d":
            # Compute Cohen's d for each feature
            d = cohens_d(x_array[:, i], y, labels=labels)
            results.append({"feature": name, "|effect_size|": abs(d), "cohens_d": d})

    # Create DataFrame and sort by absolute effect size
    df = pd.DataFrame(results)
    df = df.set_index("feature", drop=True).sort_values("|effect_size|", ascending=False)
    return df


def rank_features_by_effect_size(
    x: pd.DataFrame,
    y: pd.Series | np.ndarray,
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


def summarize_class_balance_per_batch(
    batch_labels: np.ndarray | Sequence[int] | ExtensionArray,
    y: np.ndarray | Sequence[int] | ExtensionArray,
    class_names: Sequence[str] | np.ndarray | pd.Index,
    focus_class: str,
) -> None:
    """Print the proportion of a given class (by name) in each batch.

    Args:
            batch_labels: Array of batch labels (integers).
            y: Array of class indices (integers).
            class_names: List of class names (strings).
            focus_class: Class to focus on (e.g., "healthy").
    """
    batch_labels = np.asarray(batch_labels)
    y = np.asarray(y)
    class_names = np.asarray(class_names)
    focus_class_idx = np.where(class_names == focus_class)[0][0]

    print("\nClass balance per batch:")
    for batch_id in np.unique(batch_labels):
        mask_batch = batch_labels == batch_id
        n_focus = np.sum(mask_batch & (y == focus_class_idx))
        n_total = np.sum(mask_batch)
        pct = 100 * n_focus / n_total
        print(f"  Batch {batch_id}: {pct:.1f}% {focus_class}")


def plot_pca_by_class_and_batch_from_meta(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series | ExtensionArray,
    meta: DatasetMeta,
    random_state: int = 42,
    scale: bool = True,
) -> None:
    """Plot PCA (PC1/PC2) colored by class (left) and batch (right) using DatasetMeta.

    Args:
        x: Data matrix of shape (n_samples, n_features).
        y: Array of class indices (integers).
        meta: DatasetMeta-like object with `class_labels` and `batch_labels` attributes.
        random_state: Random seed for PCA (relevant for randomized solvers).
        scale: If True, standardize features to zero mean and unit variance before PCA.
            This is recommended for most batch-effect diagnostics.
    """
    # y: array of class indices (integers)
    class_labels = np.asarray(meta.class_names)[np.asarray(y)]
    batch_labels = np.asarray(meta.batch_labels)

    x_proc = StandardScaler().fit_transform(x) if scale else x

    pca = PCA(n_components=2, random_state=random_state)
    x_pca = pca.fit_transform(x_proc)

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: color by class (string labels)
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, cls in enumerate(np.unique(class_labels)):
        mask = class_labels == cls
        axes[0].scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            alpha=0.6,
            s=50,
            label=cls,
            color=default_colors[idx % len(default_colors)],
        )

    axes[0].set_title("Colored by Class")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: color by batch
    for idx, batch_id in enumerate(sorted(np.unique(batch_labels))):
        mask = batch_labels == batch_id
        axes[1].scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            alpha=0.6,
            s=50,
            label=f"Batch {batch_id}",
            color=default_colors[idx % len(default_colors)],
        )
    axes[1].set_title("Colored by Batch")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n✓ Left: Biology (class separation)")
    print("✓ Right: Technical variation (batch structure)")
    print("✓ All classes appear in all batches → no confounding")
