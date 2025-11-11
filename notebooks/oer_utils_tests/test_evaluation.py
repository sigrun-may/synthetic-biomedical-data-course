# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for evaluation utilities."""
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy._typing import NDArray
from oer_utils.evaluation import (
    compare_cv_schemes,
    compute_cv_performance_gap,
    evaluate_multiple_models,
    summarize_cv_results,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def synthetic_data() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Generate synthetic binary classification data."""
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    groups = np.repeat(np.arange(5), 20)  # 5 groups of 20 samples
    return x, y, groups


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """Create a simple sklearn pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])


class TestCVComparison:
    """Tests for cross-validation comparison functions."""

    def test_compare_cv_schemes_basic(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test basic CV scheme comparison."""
        x, y, groups = synthetic_data

        cv_configs = {
            "naive": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "grouped": GroupKFold(n_splits=5),
        }

        results = compare_cv_schemes(x, y, simple_pipeline, cv_configs, groups=groups, scoring="balanced_accuracy")

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 schemes
        assert set(results["scheme"]) == {"naive", "grouped"}
        assert "mean" in results.columns
        assert "std" in results.columns
        assert "n_folds" in results.columns

    def test_compare_cv_schemes_multiple_metrics(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test CV comparison with multiple metrics."""
        x, y, _groups = synthetic_data

        cv_configs = {"naive": StratifiedKFold(n_splits=3, shuffle=True, random_state=42)}

        results = compare_cv_schemes(
            x, y, simple_pipeline, cv_configs, scoring={"bal_acc": "balanced_accuracy", "roc_auc": "roc_auc"}
        )

        assert len(results) == 2  # 1 scheme × 2 metrics
        assert set(results["metric"]) == {"balanced_accuracy", "roc_auc"}

    def test_compare_cv_schemes_with_train_scores(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test CV comparison with training scores."""
        x, y, _ = synthetic_data

        cv_configs = {"naive": StratifiedKFold(n_splits=3, shuffle=True, random_state=42)}

        results = compare_cv_schemes(
            x, y, simple_pipeline, cv_configs, scoring="balanced_accuracy", return_train_score=True
        )

        assert "train_mean" in results.columns
        assert "train_std" in results.columns

    def test_compare_cv_schemes_dataframe_input(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test CV comparison with DataFrame input."""
        x, y, groups = synthetic_data
        x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
        y_series = pd.Series(y)
        groups_series = pd.Series(groups)

        cv_configs = {"naive": StratifiedKFold(n_splits=3, shuffle=True, random_state=42)}

        results = compare_cv_schemes(x_df, y_series, simple_pipeline, cv_configs, groups=groups_series)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_summarize_cv_results_basic(self) -> None:
        """Test CV results summarization."""
        cv_results = {
            "test_bal_acc": np.array([0.8, 0.82, 0.78, 0.81, 0.79]),
            "test_roc_auc": np.array([0.85, 0.87, 0.83, 0.86, 0.84]),
            "fit_time": np.array([0.1, 0.12, 0.11, 0.1, 0.11]),
        }

        summary = summarize_cv_results(cv_results, scheme_label="test")

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # Only test metrics
        assert set(summary["metric"]) == {"bal_acc", "roc_auc"}
        assert all(summary["scheme"] == "test")
        assert "mean" in summary.columns
        assert "std" in summary.columns


class TestModelEvaluation:
    """Tests for model evaluation functions."""

    def test_evaluate_multiple_models_basic(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
    ) -> None:
        """Test evaluation of multiple models."""
        x, y, groups = synthetic_data

        models = {
            "LogReg": LogisticRegression(max_iter=1000, random_state=42),
            "LogReg_L1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42),
        }

        results = evaluate_multiple_models(
            models, x, y, cv=GroupKFold(n_splits=3), groups=groups, scoring="balanced_accuracy"
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 models
        assert set(results["model"]) == {"LogReg", "LogReg_L1"}
        assert "mean" in results.columns
        assert "std" in results.columns

    def test_evaluate_multiple_models_multiple_metrics(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
    ) -> None:
        """Test model evaluation with multiple metrics."""
        x, y, _ = synthetic_data

        models = {"LogReg": LogisticRegression(max_iter=1000, random_state=42)}

        results = evaluate_multiple_models(
            models, x, y, cv=StratifiedKFold(n_splits=3), scoring={"bal_acc": "balanced_accuracy", "acc": "accuracy"}
        )

        assert len(results) == 2  # 1 model × 2 metrics
        assert set(results["metric"]) == {"balanced_accuracy", "accuracy"}


class TestPerformanceGap:
    """Tests for performance gap analysis."""

    def test_compute_cv_performance_gap_basic(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test performance gap computation."""
        x, y, groups = synthetic_data

        cv_configs = {
            "naive": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "grouped": GroupKFold(n_splits=5),
        }

        results = compare_cv_schemes(
            x, y, simple_pipeline, cv_configs, groups=groups, scoring={"bal_acc": "balanced_accuracy"}
        )

        gap = compute_cv_performance_gap(results, "naive", "grouped", metric="balanced_accuracy")

        assert isinstance(gap, dict)
        assert "baseline_mean" in gap
        assert "test_mean" in gap
        assert "absolute_gap" in gap
        assert "relative_gap_pct" in gap
        assert isinstance(gap["absolute_gap"], float)
        assert isinstance(gap["relative_gap_pct"], float)
        assert isinstance(gap["baseline_mean"], float)
        assert isinstance(gap["test_mean"], float)

    def test_compute_cv_performance_gap_missing_scheme(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test that missing scheme raises error."""
        x, y, _ = synthetic_data

        cv_configs = {"naive": StratifiedKFold(n_splits=3, shuffle=True, random_state=42)}

        results = compare_cv_schemes(x, y, simple_pipeline, cv_configs, scoring="balanced_accuracy")

        with pytest.raises(ValueError, match="not found"):
            compute_cv_performance_gap(results, "naive", "nonexistent", metric="balanced_accuracy")

    def test_compute_cv_performance_gap_missing_metric(
        self, synthetic_data: tuple[NDArray[Any], NDArray[Any], NDArray[Any]], simple_pipeline: Pipeline
    ) -> None:
        """Test that missing metric raises error."""
        x, y, groups = synthetic_data

        cv_configs = {
            "naive": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            "grouped": GroupKFold(n_splits=3),
        }

        results = compare_cv_schemes(x, y, simple_pipeline, cv_configs, groups=groups, scoring="balanced_accuracy")

        with pytest.raises(ValueError, match="not found"):
            compute_cv_performance_gap(results, "naive", "grouped", metric="nonexistent_metric")
