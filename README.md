# Synthetic Biomedical Data — Learning Module

Welcome to the **Synthetic Biomedical Data** learning module.  
This course introduces the concept of artificially generated biomedical datasets, 
explains why they are useful, and provides hands-on exercises in **Jupyter Notebooks**.

---

## Structure of the Module

The module is divided into several lessons.  
Each lesson is one Jupyter Notebook focusing on a specific concept.  
You can go through them in order, or revisit individual notebooks as needed.  

---

### Lesson 1 — Introduction
- Define synthetic biomedical data.  
- Explain its motivation and use cases (benchmarking, teaching, privacy-preserving research).  
- Understand the limitations and advantages.  

📓 Notebook: `01_intro.ipynb`

---

### Lesson 2 — Data Generation Basics
- Generate your first synthetic dataset with `scikit-learn`.  
- Explore features, samples, and classes.  
- Visualize class separability and feature distributions.  

📓 Notebook: `02_data_generation_basics.ipynb`

---

### Lesson 3 — Advanced Data Generation
This lesson is split into **four focused notebooks**, each adding a layer of realism.  

- **3a: Irrelevant Features**  
  Add noise features that dilute the signal and test model robustness.  
  📓 Notebook: `03a_irrelevant_features.ipynb`  

- **3b: Correlated Features**  
  Simulate biologically realistic correlations (genes in pathways, metabolites).  
  📓 Notebook: `03b_correlated_features.ipynb`  

- **3c: Pseudo-classes**  
  Create artificial subgroups (e.g., site, hospital, eye color) that can mislead models.  
  📓 Notebook: `03c_pseudo_classes.ipynb`  

- **3d: Random Effects**  
  Add systematic external variation (batch effects, measurement day).  
  📓 Notebook: `03d_random_effects.ipynb`  

---

### Lesson 4 — Visualization and Exploration
- Apply visualization techniques (heatmaps, scatterplots, PCA).  
- Detect noise, correlations, pseudo-classes, and random effects.  
- Build intuition for identifying such challenges in real data.  

📓 Notebook: `04_visualization_exploration.ipynb`

---

### Lesson 5 — Export and Integration
- Save synthetic datasets (CSV/Parquet).  
- Use them in downstream workflows (feature selection, ML pipelines).  
- Connect to the **Feature Selection Module** for benchmarking.  

📓 Notebook: `05_export_integration.ipynb`

---

## Suggested Workflow

1. **Start with Lesson 1–2** to learn the basics.  
2. **Work through Lesson 3a–d** in sequence, each adds more realism.  
3. **Continue with Lesson 4** to learn visualization strategies.  
4. **Finish with Lesson 5** to export and integrate your datasets.  

---

## Learning Outcomes

After completing all notebooks, you will be able to:
- Generate synthetic biomedical datasets with controlled properties.  
- Understand the effects of irrelevant features, correlations, pseudo-classes, and random effects.  
- Visualize and interpret these challenges.  
- Export datasets for testing feature selection and machine learning methods.  
- Apply these insights when working with real biomedical data.  


## 🚀 Get Started Instantly (with GitHub Codespaces)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/sigrun-may/synthetic-data-tutorial/codespaces)

No setup required! Just click the button above (or "Code" > "Open with Codespaces") and start exploring in your browser.

---

## 📚 Contents

- `notebooks/` – Jupyter notebooks for lessons and exercises
- `data/` – Sample synthetic datasets
- `requirements.txt` – Python dependencies
- `.devcontainer/` – Codespaces/VS Code development environment config

## 🛠️ Local Setup (optional)

If you prefer working locally:

```bash
git clone https://github.com/sigrun-may/synthetic-data-tutorial.git
cd synthetic-data-tutorial
python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
jupyter lab
