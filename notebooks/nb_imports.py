# Centralized imports for notebooks. Prefer explicit __all__ re-exports.
from __future__ import annotations

# Core
import sys

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # keep optional
    sns = None

from biomedical_data_generator import BatchEffectsConfig, ClassConfig, CorrClusterConfig, DatasetConfig
from biomedical_data_generator.features.correlated import sample_correlated_cluster
from biomedical_data_generator.generator import generate_dataset
from biomedical_data_generator.utils.correlation_tools import (
    compute_correlation_matrix,
)
from biomedical_data_generator.utils.visualization import (
    plot_all_correlation_clusters,
    plot_correlation_matrix,
    plot_correlation_matrix_for_cluster,
)
from oer_utils.batch_effects import summarize_batch_distribution_by_class, summarize_class_balance_per_batch
from oer_utils.evaluation import compare_cv_schemes, evaluate_multiple_models
from oer_utils.feature_selection import rank_features_by_effect_size
from oer_utils.visualization import (
    plot_effect_sizes,
    plot_feature_distributions_by_class,
    plot_pca_by_class_and_batch_from_meta,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

__all__ = [
    "sys",
    "np",
    "pd",
    "plt",
    "sns",
    "DatasetConfig",
    "CorrClusterConfig",
    "ClassConfig",
    "BatchEffectsConfig",
    "generate_dataset",
    "rank_features_by_effect_size",
    "summarize_batch_distribution_by_class",
    "summarize_class_balance_per_batch",
    "compare_cv_schemes",
    "evaluate_multiple_models",
    "plot_feature_distributions_by_class",
    "plot_effect_sizes",
    "plot_correlation_matrix_for_cluster",
    "plot_correlation_matrix",
    "plot_all_correlation_clusters",
    "plot_pca_by_class_and_batch_from_meta",
    "compute_correlation_matrix",
    "sample_correlated_cluster",
    "make_pipeline",
    "RobustScaler",
    "StandardScaler",
    "LogisticRegression",
    "StratifiedKFold",
    "train_test_split",
    "cross_val_score",
    "balanced_accuracy_score",
    "make_scorer",
]
