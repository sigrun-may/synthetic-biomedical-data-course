# Synthetic Biomedical Data — Learning Module

Welcome to the **Synthetic Biomedical Data** learning module.

This project is an **Open Educational Resource (OER)** that introduces the concepts of synthetic biomedical datasets,
explains why they are useful, and provides hands-on exercises in **Jupyter Notebooks**.

______________________________________________________________________

## 📂 Module Structure

The module is organized into lessons, each implemented as a Jupyter Notebook.\
You can follow them sequentially or revisit individual notebooks as needed.

______________________________________________________________________

### Lesson 1 — Introduction

- Define synthetic biomedical data.
- Explain motivations and use cases (benchmarking, teaching, privacy-preserving research).
- Discuss advantages and limitations.

📓 Notebook: `01_intro.ipynb`

______________________________________________________________________

### Lesson 2 — Data Generation Basics

- Generate your first synthetic dataset with `scikit-learn`.
- Explore features, samples, and classes.
- Visualize class separability and feature distributions.

📓 Notebook: `02_data_generation_basics.ipynb`

______________________________________________________________________

### Lesson 3 — Advanced Data Generation

Lesson 3 is split into **four focused notebooks**, each adding realism:

- **3a: Irrelevant Features**\
  Add noise features that dilute the signal and test model robustness.\
  📓 `03a_irrelevant_features.ipynb`

- **3b: Correlated Features**\
  Simulate biologically realistic correlations (e.g., genes in pathways, metabolites).\
  📓 `03b_correlated_features.ipynb`

- **3c: Pseudo-classes**\
  Create artificial subgroups (e.g., site, hospital, eye color) that may mislead models.\
  📓 `03c_pseudo_classes.ipynb`

- **3d: Random Effects**\
  Add systematic external variation (e.g., batch effects, measurement day).\
  📓 `03d_random_effects.ipynb`

______________________________________________________________________

### Lesson 4 — Visualization and Exploration

- Apply visualization techniques (heatmaps, scatterplots, PCA).
- Detect noise, correlations, pseudo-classes, and random effects.
- Build intuition for spotting such challenges in real data.

📓 Notebook: `04_visualization_exploration.ipynb`

______________________________________________________________________

### Lesson 5 — Export and Integration

- Save synthetic datasets (CSV/Parquet).
- Use them in downstream workflows (feature selection, ML pipelines).
- Connect to the **Feature Selection Module** for benchmarking.

📓 Notebook: `05_export_integration.ipynb`

______________________________________________________________________

## 🔄 Suggested Workflow

1. **Start with Lesson 1–2** for fundamentals.
1. **Work through Lesson 3a–d** — each introduces additional complexity.
1. **Continue with Lesson 4** to practice visualization.
1. **Finish with Lesson 5** to export and integrate datasets.

______________________________________________________________________

## 🎯 Learning Outcomes

After completing all notebooks, you will be able to:

- Generate synthetic biomedical datasets with controlled properties.
- Understand the effects of irrelevant features, correlations, pseudo-classes, and random effects.
- Visualize and interpret these challenges.
- Export datasets for feature selection and machine learning tasks.
- Apply these insights when working with real biomedical data.

______________________________________________________________________

## 🌐 Online Launch Options (coming soon)

In the final version of this course, we will provide **one-click online environments**\
(e.g., Google Colab, Binder, or GitHub Codespaces) to run the notebooks without local setup.

🔜 This section will be updated once the course is finalized.

______________________________________________________________________

## 🛠️ Local Setup

This project uses [Poetry](https://python-poetry.org/) for reproducible environments.

1. Install Poetry (if not already installed):

```bash
pipx install poetry
```

2. Clone the repository:

```bash
git clone https://github.com/your-org/synthetic-biomedical-data-course.git
cd synthetic-biomedical-data-course
```

3. Install dependencies:

```bash
poetry install
```

## 📖 License

This project uses a dual-license model:

- **Code**: MIT (see `LICENSE`)
- **Educational content (notebooks, text, images)**: CC BY 4.0 (see `OER-LICENSE.txt`)

You are free to use, adapt, and share the materials under these terms.
