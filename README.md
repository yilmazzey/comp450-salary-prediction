# Developer Salary Prediction Using Stack Overflow 2025 Survey Data

> Automated developer salary prediction with a full ML pipeline, article-style baselines, and an OCR-enabled CV demo app.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Data Preparation Pipeline](#-data-preparation-pipeline)
- [Modeling & Results](#-modeling--results)
- [OCR + Salary Prediction App](#-ocr--salary-prediction-app)
- [Dataset Details](#-dataset-details)
- [Requirements](#-requirements)
- [Team](#-team)
- [References](#-references)

## 🎯 Overview

This project builds a **regression pipeline to predict developer salaries** using the **Stack Overflow 2025 Developer Survey**.  
We:

- Clean and encode the survey data into a modeling-ready dataset.
- Train multiple models (linear, tree-based, boosted trees).
- Compare our results to the reference Medium article *“Predicting Developer Salaries with Machine Learning”*.
- Deploy the best model in a **Streamlit OCR UI** that can take a CV (PDF/image), extract text with **DeepSeek OCR via Ollama**, map it to survey-style features, and output a salary estimate.

## 💡 Motivation

Salaries vary widely based on:

- Country / region  
- Experience and education  
- Role / developer type  
- Remote vs in-person work  

The goal is to:

- Provide **data-driven, reproducible salary estimates**.
- Show how **cleaning + feature engineering + proper evaluation** can dramatically improve performance vs a naive baseline.
- Connect an ML model to a **practical UI** where users can upload a CV and receive an estimated salary.

## 📁 Project Structure

```text
comp450/
├── apps/
│   └── ocr_ui/
│       └── app.py                      # Streamlit OCR + salary prediction demo
├── data/
│   ├── raw/
│   │   └── stack-overflow-developer-survey-2025-2/
│   │       ├── survey_results_public.csv
│   │       └── survey_results_schema.csv
│   ├── interim/
│   │   └── so_2025_clean.csv           # Cleaned tabular survey data
│   └── processed/
│       ├── so_2025_model_ready.parquet # Encoded features + target
│       ├── so_2025_train.parquet / .csv
│       ├── so_2025_test.parquet / .csv
│       ├── so_2025_feature_columns.json
│       ├── best_model.joblib           # Final trained model (HGB + log-target)
│       ├── metrics_summary.csv         # Validation metrics for candidate models
│       └── predictions_best.csv        # Predictions of best model on test set
├── docs/
│   ├── COMP 450 GROUP PROJECT.pdf      # Course project brief
│   └── Predicting Developer Salaries with Machine Learning | by Pratiti Soumya | Medium.pdf
├── env/
│   └── requirements.txt                # Python dependencies
├── src/
│   ├── data_prep.py                    # End-to-end data prep script
│   ├── data_preparation/
│   │   ├── 01_data_processing.ipynb    # Detailed processing & encoding steps
│   │   └── 02_data_analysis.ipynb      # (Optional) EDA / analysis notebook
│   └── models/
│       ├── 02_modeling.ipynb           # Main modeling & best-model selection
│       └── 03_article_models.ipynb     # Article-style models + comparison
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites

- Python **3.13**
- Git
- Virtual environment support (`venv`)
- (Optional) [Ollama](https://ollama.com) with `deepseek-ocr` model pulled, for the OCR app.

### Installation

   ```bash
   git clone https://github.com/yilmazzey/comp450-salary-prediction.git
   cd comp450-salary-prediction

   python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r env/requirements.txt
   ```

Optional: register a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name comp450-env --display-name "comp450 (py3)"
   ```

Verify core libs:

```bash
python -c "import pandas, numpy, sklearn; print('All packages installed successfully!')"
```

## 📊 Data Preparation Pipeline

### Automated script (`src/data_prep.py`)

Run:

```bash
python src/data_prep.py
```

This script:

1. Loads raw Stack Overflow 2025 survey responses.
2. Filters to **employed / self-employed** respondents.
3. Cleans and engineers key fields:
   - `EdLevelSimplified` (coarse education buckets).
   - `YearsCodeNum` (numeric years of coding experience).
   - `DevTypePrimary` (primary role from multi-select).
   - `RemoteCategory` (remote / hybrid / in-person).
4. Renames salary to `CompYearlyUSD` and:
   - Drops rows with missing crucial fields.
   - Restricts salaries to **USD 1,000–600,000**.
5. One-hot encodes categorical variables and builds a **193-feature** matrix.
6. Performs a **stratified train/test split** by binned salary (10 quantiles).
7. Saves:
   - `data/interim/so_2025_clean.csv`
   - `data/processed/so_2025_model_ready.parquet`
   - `data/processed/so_2025_train.parquet` / `.csv`
   - `data/processed/so_2025_test.parquet` / `.csv`
   - `data/processed/so_2025_feature_columns.json`

### Interactive notebooks

- `src/data_preparation/01_data_processing.ipynb`  
  Step-by-step walkthrough of the same pipeline (with checks, histograms, summaries).

- `src/data_preparation/02_data_analysis.ipynb`  
  Optional EDA on raw and processed data (distributions, correlations, split validation).

Run with:

```bash
jupyter lab   # or: jupyter notebook
```

Select the `comp450-env` kernel if you registered it.

## 🤖 Modeling & Results

### Main modeling notebook (`src/models/02_modeling.ipynb`)

This notebook works on `so_2025_train.csv` / `so_2025_test.csv` and:

1. Loads the processed splits and creates an **internal train/validation** split (stratified by salary bins).
2. Defines a metric helper computing **MAE, MedAE, MAPE, RMSE, R²**.
3. Trains and evaluates a range of models:
   - Dummy regressors (mean/median baselines).
   - Linear Regression (with scaling).
   - Regularized models: Ridge, Lasso, ElasticNet (with light grid search).
   - Tree ensembles: RandomForest, GradientBoosting, HistGradientBoosting.
   - Log-target variants with `TransformedTargetRegressor` to reduce skew.
4. Tunes the best tree models (e.g., HistGradientBoosting, RandomForest) with `GridSearchCV`.
5. Selects the **best model by validation MAE**.
6. Evaluates that best model on the held-out test set and saves artifacts:
   - `data/processed/best_model.joblib`
   - `data/processed/predictions_best.csv`
   - `data/processed/metrics_summary.csv`

**Best-performing model (on held-out test):**

- **Model**: `HistGradientBoostingRegressor` with a **log-transformed target** (via `TransformedTargetRegressor`).
- **Test metrics** (approximate):
  - MAE ≈ **\$34.6k**
  - RMSE ≈ **\$59.2k**
  - R² ≈ **0.44**

This means the model explains about **44% of salary variance** in the cleaned 2025 survey subset, with substantially better error than naive baselines.

The notebook also includes:

- Residual histograms (train vs test).
- Predicted vs actual scatter plots.
- Permutation importance plots for the final model.
- Extended visual diagnostics (relative errors, error by salary band, etc.).

### Article-style models & comparison (`src/models/03_article_models.ipynb`)

This notebook reproduces the style of models used in the Medium article and compares them directly on our processed dataset:

- **Models run on our data:**
  - Linear Regression (scaled).
  - DecisionTreeRegressor.
  - RandomForestRegressor.
  - GradientBoostingRegressor (as a boosted-tree analogue).
- Metrics computed on the **same** test set:
  - MAE, MedAE, MAPE, RMSE, R².
- Visuals:
  - Bar charts comparing metrics across these models.
  - Actual vs predicted grids per model.
  - Residual distribution overlays.

Then the notebook **loads `best_model.joblib`** and:

- Evaluates the best pipeline model on `X_test`, `y_test`.
- Combines its metrics with the article-style models in a single table.
- Draws a **side-by-side bar chart** of RMSE and R².
- Adds a second comparison using the **metrics reported by the Medium article** (their tuned Random Forest, etc.) vs our best model.

**Key comparison against the article:**

- Article’s best tuned RandomForest reports roughly:
  - Test RMSE ≈ **\$113k–129k**
  - Test R² ≈ **0.045** (≈4.5% variance explained)
- Our best model (HGB + log-target) on 2025 data:
  - Test RMSE ≈ **\$59k**
  - Test R² ≈ **0.44**

So our approach reduces RMSE by roughly **half** and improves R² by about **10×** compared to the article’s best model, while following the same general idea (linear + tree ensembles, plus better preprocessing and target transform).

## 📄 OCR + Salary Prediction App

The **OCR UI** (in `apps/ocr_ui/app.py`) is a Streamlit application that ties everything together:

- Upload a **CV as PDF or image**.
- Convert PDF pages to images (`pdf2image`).
- Run OCR using **DeepSeek-OCR via an Ollama server**:
  - Default endpoint: `OLLAMA_URL=http://localhost:11434`.
  - Default model: `OLLAMA_MODEL=deepseek-ocr`.
- Heuristically parse key fields from extracted text:
  - Country (from a list of common countries).
  - Education level (mapped to `EdLevelSimplified`).
  - Years of experience (`YearsCodeNum`).
  - Developer type (`DevTypePrimary`).
- Let the user **review and edit** these fields in the UI.
- Build a feature vector matching the model’s training columns using `so_2025_feature_columns.json`.
- Load `best_model.joblib` and output a **predicted annual salary in USD**.
- Offer **download buttons** for:
  - Parsed fields as CSV.
  - Prediction as JSON.

### Running the app

1. Ensure you have:
   - `data/processed/best_model.joblib`
   - `data/processed/so_2025_feature_columns.json`
   - Ollama running with `deepseek-ocr` pulled:
     ```bash
     ollama pull deepseek-ocr
     ollama serve
     ```
2. From the project root (with the venv active):

   ```bash
   streamlit run apps/ocr_ui/app.py
   ```

3. Open the Streamlit URL in your browser, upload a CV, and follow the steps to get a salary estimate.

## 📈 Dataset Details

### Raw data

- **Source**: Stack Overflow 2025 Developer Survey (public CSV).
- **Original respondents**: ~49k.
- **Columns used**:
  - `Country`, `EdLevel`, `YearsCode`, `Employment`, `DevType`,
  - `ConvertedCompYearly`, `RemoteWork`, `Currency`.

### Processed data

- **After cleaning**:
  - Filtered to employed / self-employed and valid salary range.
  - Dropped rows with missing key fields.
  - Final clean table: **~20,907 rows**, with:
    - `CompYearlyUSD`, `Country`, `EdLevelSimplified`, `YearsCodeNum`,
      `Employment`, `DevTypePrimary`, `RemoteCategory`, `SalaryLog10`.
- **Encoded dataset**:
  - 193 one-hot encoded features + numeric `YearsCodeNum` + target.
  - Train/test split:
    - Train: **16,725** rows.
    - Test: **4,182** rows.
  - No missing values in the final encoded matrices.

## 📦 Requirements

Main dependencies (see `env/requirements.txt` for exact versions):

- `pandas`, `numpy`, `scikit-learn`, `scipy`
- `pyarrow`
- `matplotlib`, `seaborn`
- `jupyterlab`, `ipykernel`
- `streamlit`, `pdf2image`, `Pillow`, `requests`, `joblib`

You can install them via:

```bash
pip install -r env/requirements.txt
```

## 👥 Team

- **Zehra Mert** (042201058)
- **Onat Sarıbıyık** (042101097)
- **Zeynep Yılmaz** (042101088)

## 📚 References

1. Course brief: `docs/COMP 450 GROUP PROJECT.pdf`
2. Medium article: `docs/Predicting Developer Salaries with Machine Learning | by Pratiti Soumya | Medium.pdf`
3. Dataset: [Stack Overflow 2025 Developer Survey](https://survey.stackoverflow.co)

Additional related work:

- Chen, Y. & Li, X. (2023). *Salary Prediction Based on the Resumes of the Candidates*.
- Akay, M. F., et al. (2025). *Development of Salary Prediction Models for the Information Technology Industry*.
- Ji, Y., et al. (2025). *Enhancing Job Salary Prediction with Disentangled Composition Effect Modeling*.

## 🔮 Future Work

- Add SHAP-based interpretability for the final boosted model.
- Explore calibrated prediction intervals (quantile regression / conformal prediction).
- Extend OCR parsing to more fields (tech stack, specific job titles).
- Improve UI/UX of the Streamlit app and add batch processing.

---

**Last Updated:** December 2025

