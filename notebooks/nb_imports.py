# Centralized imports for notebooks. Prefer explicit __all__ re-exports.
from __future__ import annotations

# Core
import sys
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:  # keep optional
    sns = None

from biomedical_data_generator import DatasetConfig, NoiseDistribution, CorrClusterConfig
from biomedical_data_generator.generator import generate_dataset

from oer_utils.feature_selection import rank_features_by_effect_size
from oer_utils.evaluation import compare_cv_schemes, evaluate_multiple_models
from oer_utils.visualization import (
    plot_feature_distributions_by_class,
    plot_effect_sizes,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score

__all__ = [
    "sys", "np", "pd",
    "plt", "sns",
    "DatasetConfig", "NoiseDistribution", "CorrClusterConfig", "generate_dataset",
    "rank_features_by_effect_size", "compare_cv_schemes", "evaluate_multiple_models",
    "plot_feature_distributions_by_class", "plot_effect_sizes",
    "make_pipeline", "RobustScaler", "LogisticRegression",
    "StratifiedKFold", "cross_val_score", "balanced_accuracy_score", "make_scorer",
]
