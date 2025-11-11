# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""OER utilities for synthetic biomedical data course.

This package provides helper functions for the OER notebooks, organized by topic:

- metrics: Effect size calculations (Cohen's d, etc.)
- batch_effects: Simulate and correct batch/site effects
- reporting: Reproducibility helpers (data cards, experiment logs)
- evaluation: CV helpers, model comparison, performance gap analysis
- feature_selection: Feature ranking, stability metrics, overlap analysis
- plotting: Visualization utilities for distributions, CV comparison, rankings
"""

__version__ = "1.0.0"

# Metrics
# Batch effects
from .batch_effects import (
    mean_center_per_batch,
    per_batch_stats,
)

# Evaluation
from .evaluation import (
    compare_cv_schemes,
    compute_cv_performance_gap,
    evaluate_multiple_models,
    summarize_cv_results,
)

# Feature selection
from .feature_selection import (
    compare_feature_rankings,
    compute_feature_stability,
    compute_jaccard_similarity,
    compute_precision_at_k,
    rank_features_by_effect_size,
    rank_features_by_importance,
)
from .metrics import cohens_d, compute_all_effect_sizes

# Reporting
from .reporting import create_experiment_log, data_card

# Plotting
from .visualization import (
    plot_cv_comparison,
    plot_effect_sizes,
    plot_feature_distributions_by_class,
    plot_performance_gap,
)

__all__ = [
    # Metrics
    "cohens_d",
    "compute_all_effect_sizes",
    # Batch effects
    "per_batch_stats",
    "mean_center_per_batch",
    # Reporting
    "data_card",
    "create_experiment_log",
    # Evaluation
    "compare_cv_schemes",
    "evaluate_multiple_models",
    "summarize_cv_results",
    "compute_cv_performance_gap",
    # Feature selection
    "rank_features_by_effect_size",
    "rank_features_by_importance",
    "compute_jaccard_similarity",
    "compute_precision_at_k",
    "compute_feature_stability",
    "compare_feature_rankings",
    # Plotting
    "plot_feature_distributions_by_class",
    "plot_cv_comparison",
    "plot_effect_sizes",
    "plot_performance_gap",
]
