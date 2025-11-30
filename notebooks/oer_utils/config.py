"""Small collection of helper functions that create standardized dataset and batch-effect configurations.

The goal of this module is not to hide configuration details from learners
but to provide a few reusable presets so that different notebooks can build
on the same baseline setup.

Typical usage in a notebook
---------------------------

    from oer_utils.config import (
        make_default_config,
        make_batch_no_effect,
        make_batch_additive,
        make_batch_multiplicative,
        make_batch_partial_confounding,
    )

    # Example: dataset without batch effects
    cfg_no_batch = make_default_config(batch=make_batch_no_effect())

    # Example: strong additive batch effect, no confounding with class
    cfg_add_strong = make_default_config(
        batch=make_batch_additive(strength=1.0, confounding=0.0)
    )

    # Example: partially confounded additive batch effect
    cfg_partial = make_default_config(
        batch=make_batch_partial_confounding(strength=0.5, confounding=0.6)
    )

These helpers keep the *biological signal* (class separation, number of
informative/noise features) fixed while varying only the batch-effect design.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from biomedical_data_generator import (
    BatchEffectsConfig,
    ClassConfig,
    DatasetConfig,
)

# ---------------------------------------------------------------------------
# Global defaults used across the OER notebooks
# ---------------------------------------------------------------------------

DEFAULT_N_INFORMATIVE: int = 10
DEFAULT_N_NOISE: int = 100
DEFAULT_CLASS_SEP: float = 1.5
DEFAULT_N_SAMPLES_PER_CLASS: int = 50
DEFAULT_CLASS_LABELS: tuple[str, str] = ("healthy", "diseased")
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Batch-effects helper factories
# ---------------------------------------------------------------------------
def make_batch_no_effect(
    n_batches: int = 3,
    affected_features: list[int] | Literal["all"] = "all",  # 0-based column indices; "all" => all
) -> BatchEffectsConfig:
    """
    Create a BatchEffectsConfig with **no actual batch effect**.

    This preset is useful as a clean baseline:
    - batches are present as a grouping variable,
    - but effect_strength = 0.0, so they do not change the data.

    Args:
        n_batches:
            Number of batches to simulate.
        affected_features:
            Which features are affected by the batch effect: List of column indices (0-based), "informative" or "all".
            Even though the effect strength is zero here, we keep the same
            API as for other batch presets.

    Returns:
        BatchEffectsConfig
            Configuration with zero-strength additive batch effect.
    """
    return BatchEffectsConfig(
        n_batches=n_batches,
        confounding_with_class=0.0,
        effect_type="additive",
        effect_strength=0.0,
        affected_features=affected_features,
    )


def make_batch_additive(
    strength: float = 0.5,
    confounding: float = 0.0,
    n_batches: int = 3,
    affected_features: list[int] | Literal["all"] = "all",
) -> BatchEffectsConfig:
    """
      Create an **additive** batch-effect configuration.

      Additive effects shift feature means by a batch-specific offset.
      This is a common model for technical artifacts such as different
      scanner offsets or plate effects.

          Args:
      strength:
          Magnitude of the batch effect. Higher values create stronger
          mean shifts between batches.
      confounding:
          Degree of alignment between batch and class labels
          (0.0 = independent, 1.0 = perfectly aligned).
      n_batches:
          Number of batches to simulate.
      affected_features:
          Which features are affected by the batch effect
          (e.g. "all", "informative", "noise").

    Returns:
      BatchEffectsConfig
          Configuration for an additive batch effect.
    """
    return BatchEffectsConfig(
        n_batches=n_batches,
        confounding_with_class=confounding,
        effect_type="additive",
        effect_strength=strength,
        affected_features=affected_features,
    )


def make_batch_multiplicative(
    strength: float = 0.5,
    confounding: float = 0.0,
    n_batches: int = 3,
    affected_features: list[int] | Literal["all"] = "all",
) -> BatchEffectsConfig:
    """
      Create a **multiplicative** batch-effect configuration.

      Multiplicative effects rescale features by a batch-specific factor.
      This changes variances and can mimic situations where certain
      experiments systematically amplify or attenuate signal.

          Args:
      strength:
          Magnitude of the multiplicative effect. Larger values lead to
          larger differences in scale between batches.
      confounding:
          Degree of alignment between batch and class labels
          (0.0 = independent, 1.0 = perfectly aligned).
      n_batches:
          Number of batches to simulate.
      affected_features:
          Which features are affected by the batch effect.

    Returns:
      BatchEffectsConfig
          Configuration for a multiplicative batch effect.
    """
    return BatchEffectsConfig(
        n_batches=n_batches,
        confounding_with_class=confounding,
        effect_type="multiplicative",
        effect_strength=strength,
        affected_features=affected_features,
    )


def make_batch_partial_confounding(
    effect_type: str = "additive",
    strength: float = 0.5,
    confounding: float = 0.6,
    n_batches: int = 3,
    affected_features: list[int] | Literal["all"] = "all",
) -> BatchEffectsConfig:
    """
      Create a batch-effect configuration with **partial confounding**.

      This is useful when demonstrating that:
      - batches and classes are *not* fully aligned,
      - but still statistically associated (e.g. one class is
        overrepresented in certain batches).

          Args:
      effect_type:
          One of "additive" or "multiplicative".
      strength:
          Magnitude of the batch effect.
      confounding:
          Degree of confounding between batch and class.
          Values between 0.2 and 0.8 work well for demonstrations.
      n_batches:
          Number of batches to simulate.
      affected_features:
          Which features are affected by the batch effect.

    Returns:
      BatchEffectsConfig
          Configuration capturing both a batch effect and partial confounding.
    """
    if effect_type not in {"additive", "multiplicative"}:
        raise ValueError(f"effect_type must be 'additive' or 'multiplicative', got {effect_type!r}")

    return BatchEffectsConfig(
        n_batches=n_batches,
        confounding_with_class=confounding,
        effect_type=effect_type,
        effect_strength=strength,
        affected_features=affected_features,
    )


# ---------------------------------------------------------------------------
# DatasetConfig factories used across notebooks
# ---------------------------------------------------------------------------
def make_binary_class_configs(
    n_samples_per_class: int = DEFAULT_N_SAMPLES_PER_CLASS,
    labels: Sequence[str] = DEFAULT_CLASS_LABELS,
) -> list[ClassConfig]:
    """
      Build a simple two-class configuration with equal sample size.

      This helper is used to keep the biological signal constant across
      different batch-effect experiments.

          Args:
      n_samples_per_class:
          Number of samples per class.
      labels:
          Class labels to use, typically ("healthy", "diseased").

    Returns:
      list[ClassConfig]
          Two ClassConfig objects, one per class.
    """
    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 class labels for a binary setup, got {len(labels)}.")

    return [
        ClassConfig(n_samples=n_samples_per_class, label=labels[0]),
        ClassConfig(n_samples=n_samples_per_class, label=labels[1]),
    ]


def make_default_config(
    batch: BatchEffectsConfig | None,
    n_informative: int = DEFAULT_N_INFORMATIVE,
    n_noise: int = DEFAULT_N_NOISE,
    n_samples_per_class: int = DEFAULT_N_SAMPLES_PER_CLASS,
    class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
    class_sep: float = DEFAULT_CLASS_SEP,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetConfig:
    """
      Create a **standard binary dataset configuration** used throughout the OER.

      This function fixes all aspects of the dataset except the batch design.
      It is intended for scenarios where you want to:
      - keep the underlying biological signal identical, and
      - vary only how batch effects are added.

          Args:
      batch:
          BatchEffectsConfig describing how batches are assigned and how they
          affect the data. Use the helpers in this module (e.g.
          `make_batch_no_effect`, `make_batch_additive`, `make_batch_multiplicative`,
          `make_batch_partial_confounding`) to construct it.
          If None, no batch effects are applied.
      n_informative:
          Number of informative features.
      n_noise:
          Number of pure noise features.
      n_samples_per_class:
          Number of samples per class (binary setting).
      class_labels:
          Names of the two classes, e.g. ("healthy", "diseased").
      class_sep:
          Separation strength between classes. Larger values make the
          classification problem easier.
      random_state:
          Global random seed for the generator.

    Returns:
      DatasetConfig
          Fully specified configuration ready to be passed to `generate_dataset`.
    """
    class_configs = make_binary_class_configs(
        n_samples_per_class=n_samples_per_class,
        labels=class_labels,
    )

    # The generator expects `class_sep` as a list to allow for more complex
    # setups (e.g. different separations per informative block). For the OER
    # presets we keep a single scalar value.
    class_sep_list: list[float] = [float(class_sep)]

    return DatasetConfig(
        n_informative=n_informative,
        n_noise=n_noise,
        class_configs=class_configs,
        class_sep=class_sep_list,
        batch_effects=batch,
        random_state=random_state,
    )


def make_default_configs_for_effect_types(
    batch_configs: dict[str, BatchEffectsConfig],
    **kwargs,
) -> dict[str, DatasetConfig]:
    """
      Wrap a collection of batch-effect settings into DatasetConfig objects.

      This helper is convenient when you want to compare several effect types
      or strengths side by side in a notebook.

          Args:
      batch_configs:
          Mapping from a human-readable name (e.g. "Additive (weak)") to a
          BatchEffectsConfig instance.
      **kwargs:
          Additional keyword arguments forwarded to `make_default_config`
          (e.g. `n_informative`, `n_noise`, `class_sep`).

    Returns:
      dict[str, DatasetConfig]
          Dictionary with the same keys as `batch_configs`, where each value
          is a full DatasetConfig constructed via `make_default_config`.
    """
    configs: dict[str, DatasetConfig] = {}
    for name, batch_cfg in batch_configs.items():
        configs[name] = make_default_config(batch=batch_cfg, **kwargs)
    return configs
