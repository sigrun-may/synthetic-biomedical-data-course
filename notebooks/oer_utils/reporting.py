# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Reporting and documentation helpers for reproducibility.

Provides tools to document dataset generation and analysis steps,
teaching good practices for computational reproducibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

__all__ = [
    "data_card",
    "create_experiment_log",
]


def data_card(cfg: Any, meta: Any) -> dict[str, Any]:
    """Create reproducibility card for generated dataset.

    Extracts key parameters from DatasetConfig and DatasetMeta into
    a simple dictionary that can be saved alongside analysis results.

    Args:
        cfg: DatasetConfig object used to generate data.
        meta: DatasetMeta object returned by generate_dataset.

    Returns:
        Dictionary with essential parameters for reproducing the dataset:
            - Dataset dimensions (n_samples, n_features, n_classes)
            - Feature counts by role (n_informative, n_pseudo, n_noise)
            - Generation parameters (class_sep, random_state, feature_naming)
            - Actual feature counts from metadata

    Examples:
        >>> from biomedical_data_generator import DatasetConfig, generate_dataset
        >>> import json
        >>>
        >>> cfg = DatasetConfig(
        ...     n_samples=100, n_informative=5, n_noise=3,
        ...     n_classes=2, class_counts={0: 50, 1: 50},
        ...     random_state=42
        ... )
        >>> X, y, meta = generate_dataset(cfg)
        >>>
        >>> card = data_card(cfg, meta)
        >>> print(json.dumps(card, indent=2))
        {
          "n_samples": 100,
          "n_informative": 5,
          "n_pseudo": 0,
          "n_noise": 3,
          "n_classes": 2,
          "class_sep": 1.2,
          "random_state": 42,
          ...
        }
        >>>
        >>> # Save for reproducibility
        >>> with open("dataset_card.json", "w") as f:
        ...     json.dump(card, f, indent=2)

    Notes:
        This is a teaching tool to demonstrate good documentation practices.
        In real research, you'd also include:
        - Software versions (numpy, pandas, biomedical-data-generator)
        - Platform info (OS, Python version)
        - Analysis scripts used
        - Results and figures produced
    """
    card = {
        # Core dataset dimensions
        "n_samples": getattr(cfg, "n_samples", None),
        "n_informative": getattr(cfg, "n_informative", None),
        "n_pseudo": getattr(cfg, "n_pseudo", None),
        "n_noise": getattr(cfg, "n_noise", None),
        "n_classes": getattr(cfg, "n_classes", None),
        # Generation parameters
        "class_sep": getattr(cfg, "class_sep", None),
        "feature_naming": getattr(cfg, "feature_naming", None),
        "random_state": getattr(cfg, "random_state", None),
    }

    # Add actual counts from metadata (may differ slightly due to clusters)
    if hasattr(meta, "informative_idx"):
        card["informative_idx_count"] = len(meta.informative_idx) if meta.informative_idx else 0
    if hasattr(meta, "pseudo_idx"):
        card["pseudo_idx_count"] = len(meta.pseudo_idx) if meta.pseudo_idx else 0
    if hasattr(meta, "noise_idx"):
        card["noise_idx_count"] = len(meta.noise_idx) if meta.noise_idx else 0

    # Class distribution
    if hasattr(meta, "y_counts"):
        card["class_counts"] = dict(meta.y_counts)

    # Correlation clusters info (if present)
    if hasattr(meta, "corr_cluster_indices") and meta.corr_cluster_indices:
        card["n_corr_clusters"] = len(meta.corr_cluster_indices)

    return card


def create_experiment_log(
    cfg: Any,
    meta: Any,
    results: dict[str, Any],
    description: str = "",
    include_timestamp: bool = True,
) -> dict[str, Any]:
    """Create structured experiment log combining data card and results.

    Useful for documenting complete analysis workflows in notebooks.

    Args:
        cfg: DatasetConfig used to generate data.
        meta: DatasetMeta returned by generate_dataset.
        results: Dictionary of analysis results (e.g., accuracies, effect sizes).
        description: Human-readable description of the experiment.
        include_timestamp: Whether to add timestamp to log.

    Returns:
        Dictionary with sections:
            - description: Experiment description
            - timestamp: When the experiment was run (if include_timestamp=True)
            - dataset: Data card
            - results: Analysis results

    Examples:
        >>> cfg = DatasetConfig(n_samples=100, n_informative=5, n_noise=3, ...)
        >>> X, y, meta = generate_dataset(cfg)
        >>>
        >>> # Run some analysis
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import cross_val_score
        >>> model = LogisticRegression()
        >>> scores = cross_val_score(model, X, y, cv=5)
        >>>
        >>> # Log the experiment
        >>> log = create_experiment_log(
        ...     cfg, meta,
        ...     results={
        ...         "cv_mean": scores.mean(),
        ...         "cv_std": scores.std(),
        ...         "cv_scores": scores.tolist()
        ...     },
        ...     description="Baseline logistic regression with 5-fold CV"
        ... )
        >>>
        >>> import json
        >>> with open("experiment_log.json", "w") as f:
        ...     json.dump(log, f, indent=2)

    Notes:
        Teaching tip: Have students create experiment logs for each
        notebook section to practice reproducible research workflows.
    """
    log: dict[str, Any] = {}

    if description:
        log["description"] = description

    if include_timestamp:
        log["timestamp"] = datetime.now().isoformat()

    log["dataset"] = data_card(cfg, meta)
    log["results"] = results

    return log
