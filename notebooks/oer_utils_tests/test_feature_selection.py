# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for feature selection utilities."""

import pandas as pd
import pytest
from oer_utils.feature_selection import (
    compare_feature_rankings,
    compute_feature_stability,
    compute_jaccard_similarity,
    compute_precision_at_k,
    rank_features_by_effect_size,
    rank_features_by_importance,
)


class TestFeatureRanking:
    """Tests for feature ranking functions."""

    def test_rank_features_by_effect_size_basic(self) -> None:
        """Test basic ranking by effect size."""
        x = pd.DataFrame({"i1": [1, 2, 3, 4, 5, 6], "n1": [0.1, 0.2, 0.15, 0.18, 0.12, 0.16]})
        y = pd.Series([0, 0, 0, 1, 1, 1])

        ranked = rank_features_by_effect_size(x=x, y=y, ascending=False)

        assert len(ranked) == 2
        assert "feature" in ranked.columns
        assert "|effect_size|" in ranked.columns
        assert "rank" in ranked.columns
        # i1 should rank higher (larger difference)
        assert ranked.iloc[0]["feature"] == "i1"
        assert ranked.iloc[0]["rank"] == 1

    def test_rank_features_by_effect_size_invalid_labels(self) -> None:
        """Test that non-binary labels raise error."""
        x = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.Series([0, 1, 2])  # Three classes

        with pytest.raises(ValueError, match="y must contain exactly two distinct labels."):
            rank_features_by_effect_size(x, y)

    def test_rank_features_by_importance_basic(self) -> None:
        """Test ranking by importance scores."""
        features = ["f1", "f2", "f3"]
        importances = [0.1, 0.5, 0.3]

        ranked = rank_features_by_importance(features, importances)

        assert len(ranked) == 3
        assert ranked.iloc[0]["feature"] == "f2"  # Highest importance
        assert ranked.iloc[0]["rank"] == 1
        assert ranked.iloc[2]["feature"] == "f1"  # Lowest importance
        assert ranked.iloc[2]["rank"] == 3

    def test_rank_features_by_importance_length_mismatch(self) -> None:
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="Length mismatch"):
            rank_features_by_importance(["f1", "f2"], [0.1, 0.2, 0.3])


class TestStabilityMetrics:
    """Tests for stability and overlap metrics."""

    def test_compute_jaccard_similarity_identical(self) -> None:
        """Test Jaccard on identical sets."""
        set_a = {"i1", "i2", "i3"}
        set_b = {"i1", "i2", "i3"}
        j = compute_jaccard_similarity(set_a, set_b)
        assert j == 1.0

    def test_compute_jaccard_similarity_disjoint(self) -> None:
        """Test Jaccard on disjoint sets."""
        set_a = {"i1", "i2"}
        set_b = {"n1", "n2"}
        j = compute_jaccard_similarity(set_a, set_b)
        assert j == 0.0

    def test_compute_jaccard_similarity_partial_overlap(self) -> None:
        """Test Jaccard with partial overlap."""
        set_a = {"i1", "i2", "n1"}
        set_b = {"i1", "i3", "n1"}
        j = compute_jaccard_similarity(set_a, set_b)
        # Intersection: {i1, n1} = 2, Union: {i1, i2, i3, n1} = 4
        assert j == 0.5

    def test_compute_jaccard_similarity_empty_sets(self) -> None:
        """Test Jaccard on empty sets."""
        j = compute_jaccard_similarity(set(), set())
        assert j == 1.0  # Convention: empty sets are identical

    def test_compute_precision_at_k_perfect_overlap(self) -> None:
        """Test Precision@k with perfect overlap."""
        r1 = ["i1", "i2", "n1", "i3"]
        r2 = ["i1", "i2", "n2", "i4"]
        p = compute_precision_at_k(r1, r2, k=2)
        assert p == 1.0  # Top-2 in both: i1, i2

    def test_compute_precision_at_k_no_overlap(self) -> None:
        """Test Precision@k with no overlap."""
        r1 = ["i1", "i2", "i3"]
        r2 = ["n1", "n2", "n3"]
        p = compute_precision_at_k(r1, r2, k=2)
        assert p == 0.0

    def test_compute_precision_at_k_partial_overlap(self) -> None:
        """Test Precision@k with partial overlap."""
        r1 = ["i1", "i2", "n1"]
        r2 = ["i1", "n2", "i2"]
        p = compute_precision_at_k(r1, r2, k=2)
        assert p == 0.5  # Only i1 overlaps in top-2

    def test_compute_precision_at_k_invalid_k(self) -> None:
        """Test that k <= 0 raises error."""
        with pytest.raises(ValueError, match="k must be positive"):
            compute_precision_at_k(["i1"], ["i2"], k=0)

    def test_compute_feature_stability_jaccard(self) -> None:
        """Test feature stability with Jaccard method."""
        rankings = [["i1", "i2", "n1"], ["i1", "i2", "n2"], ["i2", "i1", "n3"]]

        stability = compute_feature_stability(rankings, top_k=2, method="jaccard")

        assert "stability_mean" in stability
        assert "stability_std" in stability
        assert "n_pairs" in stability
        assert stability["n_pairs"] == 3  # 3 choose 2
        assert stability["method"] == "jaccard"
        assert isinstance(stability["stability_mean"], float)
        assert 0 <= stability["stability_mean"] <= 1

    def test_compute_feature_stability_precision_at_k(self) -> None:
        """Test feature stability with Precision@k."""
        rankings = [["i1", "i2", "n1"], ["i1", "n2", "i2"], ["i2", "i1", "n3"]]

        stability = compute_feature_stability(rankings, top_k=2, method="precision_at_k")

        assert stability["method"] == "precision_at_k"
        assert stability["top_k"] == 2
        assert isinstance(stability["stability_mean"], float)
        assert 0 <= stability["stability_mean"] <= 1

    def test_compute_feature_stability_insufficient_rankings(self) -> None:
        """Test that <2 rankings raise error."""
        with pytest.raises(ValueError, match="at least 2 rankings"):
            compute_feature_stability([["i1"]], top_k=1)

    def test_compute_feature_stability_unknown_method(self) -> None:
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_feature_stability([["i1"], ["i2"]], method="unknown")


class TestFeatureRankingComparison:
    """Tests for ranking comparison utilities."""

    def test_compare_feature_rankings_basic(self) -> None:
        """Test basic ranking comparison."""
        rank_a = pd.DataFrame({"feature": ["i1", "i2", "n1"], "rank": [1, 2, 3]})
        rank_b = pd.DataFrame({"feature": ["i1", "n2", "i2"], "rank": [1, 2, 3]})

        cmp = compare_feature_rankings(rank_a, rank_b, top_k=2)

        assert len(cmp) == 2
        assert "rank" in cmp.columns
        assert "in_both" in cmp.columns

    def test_compare_feature_rankings_different_lengths(self) -> None:
        """Test comparison with different ranking lengths."""
        rank_a = pd.DataFrame({"feature": ["i1", "i2"], "rank": [1, 2]})
        rank_b = pd.DataFrame({"feature": ["i1", "n1", "i2", "n2"], "rank": [1, 2, 3, 4]})

        cmp = compare_feature_rankings(rank_a, rank_b, top_k=None)

        assert len(cmp) == 4  # Padded to max length
