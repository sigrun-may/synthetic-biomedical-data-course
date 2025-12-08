![KI4ALL](notebooks/img/header.png)

# Microcredit Synthetic Biomedical Data

**Author:** Sigrun May, Johann Katron, Daria Kober\
**Date:** November 2025\
**Version:** 4\
**Credits:** 1 ECTS\
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)\
**Developed by:** [TU Braunschweig](https://www.tu-braunschweig.de/), [Ostfalia Hochschule](https://landing.ostfalia.de/) and [TU Clausthal](https://www.tu-clausthal.de/)\
**Sponsored by** [Bundesministerium für Bildung und Forschung](https://www.bmbf.de/bmbf/de/home/home_node.html)

______________________________________________________________________

## Overview

The **Synthetic Biomedical Data** Microcredit teaches you how to generate, understand, and use synthetic biomedical datasets for education, benchmarking, and method testing.
You'll learn how to safely generate synthetic data that mimics biomedical patterns in oder to avoid common pitfalls in feature selection, model evaluation, and interpretation.
This course is based on the [**`biomedical-data-generator`**](https://github.com/sigrun-may/biomedical-data-generator) Python package.

### What you'll learn

- Generate biomedical datasets with controlled properties and known ground truth
- Understand the difference between **informative features** (true biomarkers) and **noise features** (irrelevant information)
- Model correlated feature clusters that mimic, for example, biological pathways
- Generate artificial artifacts from non-causal sources (batch effects, site differences, recruitment bias)
- How to evaluate feature selection leakage, improper cross-validation in p≫n settings
- Apply principles of reproducibility through deterministic seeding and configuration

### Why synthetic data?

Synthetic biomedical data provides a controlled environment where:

- **Ground truth is known**: You define which features are truly informative
- **Privacy is guaranteed**: No real patient data is involved
- **Parameters are controlled**: You can systematically vary signal strength, noise levels, and correlations
- **Methods can be benchmarked**: Compare feature selection and classification algorithms objectively
- **Learning is safe**: Experiment freely without ethical or legal constraints

______________________________________________________________________

## Target Audience

The course is designed for advanced Bachelor/Master students or learners interested in data science, bioinformatics and biomedical AI.

### Prerequisites

You should:

- have basic knowledge of **statistics** (distributions, correlation)
- be comfortable using **Python** (see [Python Introduction](https://git.rz.tu-bs.de/ifn-public/ki4all/python-introduction))
- have basic understanding of **NumPy** and **Pandas** for data manipulation
- be familiar with **machine learning fundamentals** (classification, features, labels, cross-validation) – see [Machine Learning Introduction](https://git.rz.tu-bs.de/ifn-public/ki4all/machine-learning-introduction)

______________________________________________________________________
## Module Structure

The module consists of **9 progressive lessons**, each delivered as a Jupyter Notebook with theory, code examples, and hands-on exercises.

### Launch the course in your browser

You can run all notebooks directly in your browser via Binder (no local installation required):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sigrun-may/synthetic-biomedical-data-course/HEAD?urlpath=lab/tree/notebooks/01_intro.ipynb)

### 📘 Lesson 1: Introduction to Synthetic Biomedical Data

**Notebook:** `01_intro.ipynb`

- Define synthetic biomedical data and contrast with real patient data
- Understand motivations: privacy, benchmarking, education, method testing
- Explore applications in research and teaching
- Learn about the structure of synthetic datasets

**Learning outcomes:**

- Differentiate between synthetic and real biomedical data
- Explain key benefits and limitations
- Describe dataset components (features, classes, noise, dependencies)

______________________________________________________________________

### 📗 Lesson 2: Data Generation Fundamentals

This lesson is split into two complementary notebooks covering the basics of synthetic data generation.

#### 📗 Lesson 2a: Data Generation Basics

**Notebook:** `02a_data_generation_intro.ipynb`

- Generate your first synthetic biomedical dataset
- Understand the structure: samples, features, labels
- Explore dataset properties through visualization
- Learn about reproducibility with random seeds

**Key concepts:**

- Sample-feature matrix structure (n×p)
- Class labels and class balance
- Feature distributions (Gaussian baseline)
- Visual exploration with scatter plots and histograms

#### 📗 Lesson 2b: Feature Distributions and Effect Size

**Notebook:** `02b_feature_distributions_effect_size.ipynb`

- Control **class separation** through effect size
- Understand **signal-to-noise ratio** in biomedical context
- Generate features with different distribution properties
- Learn about Cohen's d and standardized mean differences

**Key concepts:**

- Effect size as a measure of biological relevance
- Relationship between effect size and classification difficulty
- Realistic effect sizes in biomarker studies (small: 0.2–0.5, medium: 0.5–0.8, large: >0.8)
- Balancing statistical significance with clinical relevance

______________________________________________________________________

### 📙 Lesson 3: Advanced Features and Realism

This lesson introduces three critical aspects of realistic biomedical data: noise, correlations, and artifacts.

#### 📙 Lesson 3a: Irrelevant Features and Noise

**Notebook:** `03a_irrelevant_features_noise.ipynb`

- Add **irrelevant features** (noise variables) to datasets
- Understand p≫n challenges (many features, few samples)
- Test robustness to increasing noise ratios
- Recognize overfitting risks in high-dimensional settings

**Key concepts:**

- Curse of dimensionality in biomedical data
- Random features vs. informative features
- Impact on model performance and generalization

#### 📙 Lesson 3b: Noise Distributions

**Notebook:** `03b_noise_distributions.ipynb`

- Explore different noise distributions
- Model realistic measurement variability
- Understand how distribution shape affects analysis
- Generate features with skewed or heavy-tailed distributions

**Key concepts:**

- Distribution families beyond Gaussian
- Biological relevance of different distributions (e.g., log-normal for concentrations)
- Impact on parametric vs. non-parametric methods
- Robustness testing with non-ideal data

#### 📙 Lesson 3c: Correlated Features

**Notebook:** `03c_correlated_features.ipynb`

- Generate **correlated feature clusters** that mimic biological pathways
- Implement equicorrelated and Toeplitz correlation structures
- Create anchor-proxy architectures for biological redundancy
- Model class-specific correlations (e.g., pathway activation in disease)

**Key concepts:**

- Biological pathway modeling through feature correlation
- Anchor features (main biomarker) and proxy features (correlated measurements)
- Class-dependent correlation patterns

______________________________________________________________________

### 📕 Lesson 4: Non-Causal Variation and Applications

This lesson addresses the critical challenge of **confounding** and **batch effects** in biomedical studies.

#### 📕 Lesson 4a: Understanding Non-Causal Variation

**Notebook:** `04a_non_causal_variation_understanding.ipynb`

- Define **non-causal variation** (batch effects, site differences)
- Distinguish between causal signals and artifacts
- Understand **confounding** and **spurious associations**

**Key concepts:**

- Sources of non-causal variation (measurement site, instrument, time)
- Confounding vs. correlation
- Why models fail on out-of-batch data

#### 📕 Lesson 4b: Generating Non-Causal Variation

**Notebook:** `04b_non_causal_variation_generating.ipynb`

- Generate datasets with **batch effects** and **site differences**
- Control confounding strength systematically
- Test cross-validation strategies (random vs. group-aware)

**Key concepts:**

- Batch effect implementation with controllable confounding
- Group-aware cross-validation to prevent leakage
- Detecting and visualizing batch structure in data

#### 📕 Lesson 4c: Use Cases and Applications

**Notebook:** `04c_use_cases_applications.ipynb`

- Apply synthetic data to real-world scenarios
- Design benchmarking studies for method comparison
- Create educational demonstrations of common pitfalls

**Key applications:**

- Method development and validation
- Teaching proper cross-validation
- Demonstrating the importance of batch correction
- Creating reproducible research examples

______________________________________________________________________

## Learning Outcomes

After completing this module, you will be able to generate synthetic biomedical datasets including:

- Controlled signal-to-noise ratios
- Correlated feature clusters mimicking biological pathways
- Non-causal variation (batch effects, site differences)
- Mixture of informative and irrelevant features

______________________________________________________________________
## Getting Started

You have two options to work with this course:

### Option 1 – Run online (recommended for learners)

The easiest way to use this module is to run it directly in your browser via Binder.

1. Click the **Binder badge above** or use this link:
   [Open the course on Binder](https://mybinder.org/v2/gh/sigrun-may/synthetic-biomedical-data-course/HEAD?urlpath=lab/tree/notebooks/01_intro.ipynb)
2. Wait until the environment has finished building and JupyterLab opens.
3. Start with `01_intro.ipynb`.

> Note: Binder sessions are temporary. If you want to save your changes permanently,
> download the notebooks or run the course locally (see Option 2).

### Option 2 – Run locally with Poetry

If you prefer a local installation (e.g., for offline teaching or development), you can use
[Poetry](https://python-poetry.org/) for dependency management.

**Step 1: Install Poetry**

We recommend installing Poetry via `pipx`. For more details see the [Poetry installation guide](https://python-poetry.org/docs/#installation).

```bash
pipx install poetry
```


**Step 2: Clone the repository**

```bash
git clone https://github.com/sigrun-may/synthetic-biomedical-data-course.git
cd synthetic-biomedical-data-course
```

**Step 3: Install dependencies**

```bash
poetry install
```

**Step 4: Launch JupyterLab**

```bash
poetry run jupyter lab
```

Then open the notebooks/ folder and start with 01_intro.ipynb.
______________________________________________________________________

## Learning Approach

This hands-on module is designed around a critical challenge in biomedical data science: working with **high-dimensional data** where the number of features (p) greatly exceeds the number of samples (n) – the so-called **p≫n problem**.

1. **Concrete examples first**: Start with simple, clean datasets before adding complexity
1. **Progressive difficulty**: Each lesson builds on previous concepts
1. **Hands-on exercises**: Code cells for experimentation and reflection questions
1. **Visual learning**: Extensive use of plots to build intuition
1. **Known ground truth**: Always compare results to what you defined
1. **Transfer to reality**: Explicit connections to real-world biomedical challenges

______________________________________________________________________

## License

Unless otherwise noted, all contents of this repository
(Jupyter notebooks, text, figures, and example code)
are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The materials are provided **“as is”**, **without any warranties** or guarantees of correctness or fitness for a particular purpose.
The authors and institutions assume **no liability** for any use of the content.
______________________________________________________________________

## Acknowledgments

This work was developed by TU Braunschweig, Ostfalia Hochschule, and TU Clausthal with funding from the Bundesministerium für Bildung und Forschung (BMBF) through the KI4ALL initiative.

______________________________________________________________________

## Support and Feedback

- **Questions about content**: Open an issue on GitHub
- **Bug reports**: Use the issue tracker with the `bug` label
- **Feature requests**: Use the issue tracker with the `enhancement` label
- **General discussions**: Join the [GitHub Discussions](https://github.com/sigrun-may/synthetic-biomedical-data-course/discussions)
- **Contact**: s.may[at]ostfalia.de
