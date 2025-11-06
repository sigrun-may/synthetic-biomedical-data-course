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

Future modules (to be added):
- evaluation: CV helpers, model comparison
- plotting: Visualization utilities
- sweeps: Parameter sweep helpers
"""

__version__ = "1.0.0"

# Metrics
from .metrics import cohens_d, compute_all_effect_sizes

# Batch effects (essential for Lesson 03c)
from .batch_effects import (
    apply_random_intercepts,
    make_batches,
    mean_center_per_batch,
    per_batch_stats,
)

# Reporting
from .reporting import create_experiment_log, data_card

__all__ = [
    # Metrics
    "cohens_d",
    "compute_all_effect_sizes",
    # Batch effects
    "make_batches",
    "apply_random_intercepts",
    "per_batch_stats",
    "mean_center_per_batch",
    # Reporting
    "data_card",
    "create_experiment_log",
]
