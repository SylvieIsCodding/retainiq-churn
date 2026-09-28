# RetainIQ — End-to-End Customer Churn Prediction System

> A reproducible, CPU-only machine learning pipeline that predicts telecom customer churn, explains each prediction with SHAP, and serves results through a Streamlit application.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-orange)
![Keras](https://img.shields.io/badge/Keras-deep%20learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [Overview](#overview)
2. [Business Problem](#business-problem)
3. [Targets and Results](#targets-and-results)
4. [Dataset](#dataset)
5. [Approach](#approach)
6. [Repository Structure](#repository-structure)
7. [Getting Started](#getting-started)
8. [Usage](#usage)
9. [Design Decisions](#design-decisions)
10. [Project Status](#project-status)
11. [Tech Stack](#tech-stack)
12. [Author](#author)

---

## Overview

RetainIQ is an end-to-end machine learning project built around a realistic brief: a regional telecom operator loses roughly **26% of its customers every quarter**, and its retention strategy is reactive. The goal is to identify customers likely to churn *before* they cancel, so a retention team can intervene with targeted offers.

The project is organised as a sequence of 14 sprint notebooks that move from mathematical foundations through data engineering, classical ML, deep learning, NLP, explainability, and deployment. Reusable logic is progressively extracted from notebooks into a tested Python package (`src/`).

**Key characteristics**

- One dataset used consistently across every stage, so model families are directly comparable
- Leakage-safe preprocessing: fit on the training split only, transform everywhere else
- Fixed seeds and a single stratified split for reproducibility
- Explainable predictions (SHAP, with a LIME comparison) suitable for non-technical stakeholders
- 100% free and open-source tooling; no GPU required

## Business Problem

| Factor | Detail |
|---|---|
| Churn rate | ~26% of customers per quarter |
| Cost per churned customer | ~$400 (lost lifetime value + replacement acquisition) |
| Current strategy | Reactive: outreach happens after cancellation |
| Desired strategy | Proactive: rank customers by churn probability and act before they leave |

Because a missed churner is more costly than a false alarm, the evaluation prioritises **recall on the churn class** alongside F1 and ROC-AUC.

## Targets and Results

Success criteria defined for the final system:

| Metric | Target | Achieved |
|---|---|---|
| F1-score (churn class) | ≥ 0.75 | _TBD_ |
| ROC-AUC | ≥ 0.85 | _TBD_ |
| False negative rate | < 20% | _TBD_ |
| Inference time per customer | < 1 s on CPU | _TBD_ |
| Deployment | Streamlit app on localhost | _TBD_ |

Full model comparison: [`outputs/model_comparison.csv`](outputs/model_comparison.csv)

<!-- Fill the "Achieved" column from notebook 09 once the held-out evaluation is complete. -->

## Dataset

**Telco Customer Churn** (IBM Watson sample data, hosted on Kaggle)

- Source: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Size: 7,043 rows × 21 columns (~1 MB)
- Target: `Churn` (binary, imbalanced)
- Features: demographics, account information, subscribed services, billing (mixed numeric and categorical)

The CSV is not committed to the repository. See [Getting Started](#getting-started) for download instructions.

## Approach

| Stage | Focus | Notebooks | Techniques |
|---|---|---|---|
| **1. Foundations** | Math and Python fundamentals applied to the dataset | 01–02 | Linear algebra (eigendecomposition, SVD), gradients, Bayes' rule on churn probability, OOP, data validation, pandas |
| **2. Data Engineering** | Exploration, cleaning, feature engineering | 03–04 | EDA "data passport", missing-value handling, encoding, engineered features, `ChurnPreprocessor`, PCA |
| **3. Classical ML** | Baselines, model comparison, tuning, segmentation, evaluation | 05–09 | Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting, `GridSearchCV`, class weighting, K-Means, hierarchical clustering, ROC/PR analysis |
| **4. Deep Learning** | From-scratch and framework networks | 10–12 | NumPy neural network, Keras MLP, architecture survey |
| **5. Advanced and Applied** | NLP, explainability, deployment | 13–14 | TF-IDF text classifier, SHAP, LIME, pytest, Streamlit |

### Preprocessing and leakage control

`ChurnPreprocessor` (`src/data_preprocessing.py`) follows the scikit-learn `fit` / `transform` contract. Parameters such as scaler statistics are learned from the training split only. Every notebook reuses the identical split:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### Explainability

SHAP (`TreeExplainer`) provides global feature importance and per-customer waterfall plots; LIME is included as a comparison. The goal is to give a retention team a defensible reason for why each customer was flagged, not only a score.

### Application

The Streamlit app exposes three views:

1. **Single customer predictor** — input form returning churn probability and top risk factors
2. **Batch prediction** — CSV upload with downloadable predictions
3. **Model performance dashboard** — metrics, ROC curve, confusion matrix

## Repository Structure

```
retainiq-churn-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # not tracked; see setup
├── notebooks/
│   ├── 01_math_foundations.ipynb
│   ├── 02_python_foundations.ipynb
│   ├── 03_data_collection_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_classification_models.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_customer_segmentation.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_neural_network_scratch.ipynb
│   ├── 11_keras_model.ipynb
│   ├── 12_dl_architecture_survey.ipynb
│   ├── 13_nlp_customer_profiles.ipynb
│   └── 14_explainability.ipynb
├── src/
│   ├── data_preprocessing.py                  # ChurnPreprocessor
│   └── prediction_pipeline.py                 # ChurnPredictionPipeline
├── app/
│   └── streamlit_app.py
├── models/
│   ├── best_model.pkl                         # tuned classifier (joblib)
│   └── preprocessor.pkl                       # fitted preprocessor (joblib)
├── figures/                                   # eda/, models/, shap/, nlp/
├── outputs/
│   └── model_comparison.csv
└── tests/
    └── test_pipeline.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- ~1 GB free disk space (more if installing TensorFlow)

### Installation

```bash
git clone https://github.com/<your-username>/retainiq-churn-prediction.git
cd retainiq-churn-prediction

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download the dataset

Manually from Kaggle, or with the Kaggle CLI:

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip
```

## Usage

**Run the notebooks**

```bash
jupyter notebook --notebook-dir=notebooks/
```

Notebooks are numbered and intended to run in order. Each one runs cleanly with *Kernel → Restart & Run All* and writes its figures to `figures/`.

**Launch the app**

```bash
streamlit run app/streamlit_app.py
```

**Run the tests**

```bash
pytest tests/
```

**Programmatic prediction**

```python
from src.prediction_pipeline import ChurnPredictionPipeline

pipeline = ChurnPredictionPipeline.load("models/")
probability = pipeline.predict_proba(customer_dataframe)
```

## Design Decisions

- **Single dataset across all stages.** Holding data constant makes the comparison between classical ML, neural networks, and NLP features meaningful.
- **Notebooks for exploration, `src/` for reuse.** Logic that survives a sprint (preprocessing, prediction) is promoted into a tested module, mirroring how prototypes become production code.
- **Recall-aware evaluation.** Accuracy is misleading under class imbalance; the project reports F1, ROC-AUC, precision-recall behaviour, and false negative rate.
- **Class imbalance handled explicitly** via `class_weight` and stratified splitting rather than left implicit.
- **Honest deep learning conclusion.** On ~7K structured rows, tree ensembles are expected to match or beat a neural network; the deep learning stage documents this comparison rather than assuming otherwise.
- **Explanations as a deliverable.** SHAP output is treated as a first-class result because stakeholders need to justify why a customer was targeted.

## Project Status

| Stage | Status |
|---|---|
| 1. Foundations (01–02) | 🚧 In progress |
| 2. Data Engineering (03–04) | ⏳ Planned |
| 3. Classical ML (05–09) | ⏳ Planned |
| 4. Deep Learning (10–12) | ⏳ Planned |
| 5. Advanced and Applied (13–14, app, tests) | ⏳ Planned |

## Tech Stack

**Language:** Python
**Data and numerics:** NumPy, pandas, SciPy
**Visualisation:** Matplotlib, seaborn
**Machine learning:** scikit-learn
**Deep learning:** TensorFlow / Keras
**Explainability:** SHAP, LIME
**Application:** Streamlit
**Testing and persistence:** pytest, joblib
**Environment:** Jupyter, VS Code, Git

## Author

**Marielle Daha**
Computer Science, Université de Yaoundé I · Aspiring ML researcher

[GitHub](https://github.com/SylvieIsCodding) · [LinkedIn](https://linkedin.com/in/marielledaha) · [Email](mailto:marielledaha@gmail.com)

---

*The dataset is a public IBM sample. RetainIQ Analytics is a fictional company used to frame the project brief.*