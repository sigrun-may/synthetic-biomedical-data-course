from dataclasses import dataclass
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from utils import (  # noqa: E402
    cliffs_delta,
    cohens_d,
    compute_all_effect_sizes,
    plot_pca_by_class_and_batch_from_meta,
    rank_features_by_effect_size,
    summarize_class_balance_per_batch,
)


def test_cliffs_delta_basic_and_edge_cases() -> None:
    # Perfect separation (all class1 values greater than class0)
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    delta = cliffs_delta(x, y, labels=(0, 1))
    assert delta == pytest.approx(1.0)

    # Reverse perfect separation
    delta_rev = cliffs_delta(x, y, labels=(1, 0))
    assert delta_rev == pytest.approx(-1.0)

    # One empty class -> returns 0.0
    x2 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([0, 0, 0])
    assert cliffs_delta(x2, y2, labels=(0, 1)) == 0.0


def test_cohens_d_example_and_edge_cases() -> None:
    # Example from docstring: two groups with same size and increasing values
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    d = cohens_d(x, y, labels=(0, 1))
    # expected pooled std == 1 -> d == 3.0
    assert d == pytest.approx(3.0, rel=1e-3)

    # If any group has fewer than 2 samples -> 0.0
    x_small = np.array([1.0, 2.0, 3.0])
    y_small = np.array([0, 0, 1])
    # class 1 has only one sample -> returns 0.0
    assert cohens_d(x_small, y_small, labels=(0, 1)) == 0.0

    # both groups have zero variance -> pooled_std == 0
    # contrive equal values per group
    assert cohens_d(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1]), labels=(0, 1)) == 0.0


def test_compute_all_effect_sizes_both_methods() -> None:
    # Create DataFrame with two informative features and one noise
    rng = np.random.RandomState(0)
    x = pd.DataFrame(
        {
            "informative": np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)]),
            "noise": rng.normal(0, 1, 100),
        }
    )
    y = np.array([0] * 50 + [1] * 50)

    df_cd = compute_all_effect_sizes(x, y, labels=(0, 1), method="cohens_d")
    assert "cohens_d" in df_cd.columns
    # informative should have larger absolute effect than noise
    assert abs(df_cd.loc["informative", "cohens_d"]) > abs(df_cd.loc["noise", "cohens_d"])

    df_cd2 = compute_all_effect_sizes(x, y, labels=(0, 1), method="cliffs_delta")
    assert "cliffs_delta" in df_cd2.columns
    assert abs(df_cd2.loc["informative", "cliffs_delta"]) > abs(df_cd2.loc["noise", "cliffs_delta"])


def test_rank_features_by_effect_size_and_ranks() -> None:
    # Simple DataFrame with clear ordering
    x = pd.DataFrame(
        {
            "f1": [0, 0, 10, 10],  # large effect
            "f2": [0, 1, 0, 1],  # small effect
            "f3": [0, 0, 0, 0],  # zero effect
        }
    )
    y = np.array([0, 0, 1, 1])

    ranked = rank_features_by_effect_size(x, y, ascending=False)
    # best (rank 1) should be f1
    assert ranked.loc[0, "feature"] == "f1"
    assert ranked.loc[0, "|effect_size|"] >= ranked.loc[1, "|effect_size|"]
    # ranks 1..n
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_summarize_class_balance_per_batch_outputs_percentages(capsys: Any) -> None:
    # two batches, three classes names (we'll pick one to focus)
    batch_labels = np.array([0, 0, 1, 1, 1])
    y = np.array([0, 1, 0, 0, 1])  # class indices
    class_names = np.array(["healthy", "disease"])
    summarize_class_balance_per_batch(batch_labels, y, class_names, focus_class="healthy")
    captured = capsys.readouterr()
    out = captured.out
    assert "Class balance per batch:" in out
    assert "Batch 0" in out and "Batch 1" in out
    # Batch 0: one healthy of two -> 50.0%
    assert "Batch 0: 50.0% healthy" in out or "Batch 0: 50.0% healthy\n" in out


@dataclass
class DummyMeta:
    class_names: np.ndarray
    batch_labels: np.ndarray


def test_plot_pca_by_class_and_batch_from_meta_runs_and_prints(capsys: Any, monkeypatch: Any) -> None:
    # Small toy dataset
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [3.0, 3.1],
            [3.2, 2.9],
        ]
    )
    # y are indices into meta.class_names
    y = np.array([0, 0, 1, 1])
    class_names = np.array(["A", "B"])
    batch_labels = np.array([0, 0, 1, 1])

    meta = DummyMeta(class_names=class_names, batch_labels=batch_labels)

    # Prevent show from blocking and capture stdout
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_pca_by_class_and_batch_from_meta(x, y, meta, random_state=0, scale=True)
    captured = capsys.readouterr()
    out = captured.out
    assert "Left: Biology" in out or "Biology" in out
    assert "Technical variation" in out or "batch structure" in out
