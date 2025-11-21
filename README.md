# Developer Salary Prediction Using Stack Overflow 2025 Survey Data

> Automated Developer Salary Prediction using Machine Learning with OCR-Enabled CV Parsing

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Requirements](#requirements)
- [Team Members](#team-members)
- [References](#references)

## 🎯 Overview

This project aims to build a machine learning model to predict developer salaries using the **Stack Overflow 2025 Developer Survey** dataset. The methodology follows best practices from similar salary prediction projects, with a focus on data quality, feature engineering, and model interpretability.

### Key Objectives

- Build a predictive regression model for developer salaries
- Perform comprehensive data cleaning and preprocessing
- Implement stratified train/test splitting
- Analyze class imbalance and data distributions
- Provide interpretable results using SHAP values (future work)

## 💡 Motivation

Determining fair compensation for software developers is crucial and challenging in today's job market. Salaries vary significantly based on:
- Geographic location
- Technologies and skills
- Experience levels
- Educational background

This project addresses the need for data-driven salary estimation by:
- Leveraging up-to-date survey data (2025)
- Providing transparent and fair compensation insights
- Enabling informed career decisions for developers
- Supporting HR professionals in implementing fair compensation practices

## 📁 Project Structure

```
comp450/
├── data/
│   ├── raw/                          # Original, unprocessed data
│   │   └── stack-overflow-developer-survey-2025-2/
│   │       ├── survey_results_public.csv
│   │       └── survey_results_schema.csv
│   ├── interim/                      # Cleaned, transformed data
│   │   └── so_2025_clean.csv
│   └── processed/                    # Final datasets for modeling
│       ├── so_2025_model_ready.parquet
│       ├── so_2025_train.parquet
│       ├── so_2025_test.parquet
│       ├── so_2025_train.csv
│       ├── so_2025_test.csv
│       └── so_2025_feature_columns.json
├── src/
│   ├── data_prep.py                  # Automated data preparation script
│   └── data_preparation/
│       ├── 01_data_processing.ipynb  # Interactive processing notebook
│       └── 02_data_analysis.ipynb   # Exploratory data analysis notebook
├── docs/                             # Project documentation
│   └── COMP 450 GROUP PROJECT.pdf
├── references/                       # Reference materials
│   ├── Predicting Developer Salaries with Machine Learning | by Pratiti Soumya | Medium.pdf
│   └── medium_article.txt
├── env/
│   └── requirements.txt              # Python dependencies
├── notebooks/                        # Additional analysis notebooks (future)
└── README.md                         # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.13 or higher
- Git
- Virtual environment support (venv)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yilmazzey/comp450-salary-prediction.git
   cd comp450-salary-prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r env/requirements.txt
   ```

4. **Register Jupyter kernel (optional, for notebook usage)**
   ```bash
   python -m ipykernel install --user --name comp450-env --display-name "comp450 (py3)"
   ```

### Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('All packages installed successfully!')"
```

## 📊 Data Preparation

### Automated Pipeline

Run the automated data preparation script:

```bash
python src/data_prep.py
```

This script will:
1. Load raw survey data
2. Filter and clean data (employment status, salary range)
3. Engineer features (education simplification, experience parsing)
4. Encode categorical variables (one-hot encoding)
5. Create stratified train/test splits
6. Save processed datasets

**Output:**
- `data/interim/so_2025_clean.csv` - Cleaned dataset
- `data/processed/so_2025_model_ready.parquet` - Encoded features + target
- `data/processed/so_2025_train.parquet` - Training set (80%)
- `data/processed/so_2025_test.parquet` - Test set (20%)
- `data/processed/so_2025_feature_columns.json` - Feature metadata

### Interactive Notebooks

For detailed exploration and analysis:

1. **Data Processing Notebook** (`src/data_preparation/01_data_processing.ipynb`)
   - Step-by-step data cleaning
   - Feature engineering
   - Encoding and splitting

2. **Data Analysis Notebook** (`src/data_preparation/02_data_analysis.ipynb`)
   - Raw dataset exploration
   - Processed dataset analysis
   - Class imbalance checks
   - Train/test split validation
   - Feature correlations

**To run notebooks:**
```bash
jupyter lab  # or jupyter notebook
```
Select the `comp450 (py3)` kernel when opening notebooks.

## 💻 Usage

### Quick Start

```python
import pandas as pd

# Load processed data
train = pd.read_parquet('data/processed/so_2025_train.parquet')
test = pd.read_parquet('data/processed/so_2025_test.parquet')

# Separate features and target
X_train = train.drop('CompYearlyUSD', axis=1)
y_train = train['CompYearlyUSD']
X_test = test.drop('CompYearlyUSD', axis=1)
y_test = test['CompYearlyUSD']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Data Preprocessing Details

**Selected Features:**
- `Country` - Geographic location (one-hot encoded)
- `EdLevelSimplified` - Education level (simplified categories)
- `YearsCodeNum` - Years of coding experience (numeric)
- `DevTypePrimary` - Primary developer type (one-hot encoded)
- `RemoteCategory` - Remote work arrangement (one-hot encoded)

**Data Cleaning Steps:**
1. Filter to employed/self-employed respondents
2. Remove rows with missing critical values
3. Filter salary range: $1,000 - $600,000 USD
4. Simplify education categories
5. Parse experience strings to numeric values
6. Normalize remote work categories
7. Extract primary developer type from multi-select

**Stratified Splitting:**
- Uses quantile-based binning of target variable
- Ensures similar salary distributions in train/test sets
- 80/20 split with random seed (450)

## 📈 Dataset Information

### Raw Dataset
- **Source:** Stack Overflow 2025 Developer Survey
- **Original size:** ~50,000+ responses
- **Columns analyzed:** 8 key features

### Processed Dataset
- **Final size:** ~20,907 valid observations
- **Features:** 193 encoded features + 1 target
- **Target variable:** `CompYearlyUSD` (annual salary in USD)
- **Train/Test split:** 16,725 / 4,182 samples

### Data Quality
- ✅ No missing values in processed data
- ✅ Salary range: $1,000 - $600,000
- ✅ Stratified train/test split validated
- ✅ Feature distributions analyzed for imbalance

## 🔧 Features

### Data Processing
- Automated data cleaning pipeline
- Feature engineering and transformation
- One-hot encoding for categorical variables
- Stratified train/test splitting

### Analysis
- Comprehensive exploratory data analysis
- Missing value analysis
- Distribution visualizations
- Class imbalance detection
- Train/test split validation
- Feature correlation analysis

### Code Quality
- Modular, reusable functions
- Well-documented code
- Interactive Jupyter notebooks
- Reproducible results (fixed random seeds)

## 📦 Requirements

Core dependencies are listed in `env/requirements.txt`:

```
pandas>=2.3.3
numpy>=2.3.5
scikit-learn>=1.7.2
pyarrow>=22.0.0
matplotlib>=3.10.7
seaborn>=0.13.2
scipy>=1.16.3
jupyterlab>=4.5.0
ipykernel>=7.1.0
```

## 👥 Team Members

- **Zehra Mert** (042201058)
- **Onat Sarıbıyık** (042101097)
- **Zeynep Yılmaz** (042101088)

## 📚 References

1. **Project Proposal:** `docs/COMP 450 GROUP PROJECT.pdf`
2. **Reference Implementation:** `references/Predicting Developer Salaries with Machine Learning | by Pratiti Soumya | Medium.pdf`
3. **Dataset:** [Stack Overflow 2025 Developer Survey](https://survey.stackoverflow.co)

### Key Papers and Resources

- Chen, Y. & Li, X. (2023). "Salary Prediction Based on the Resumes of the Candidates"
- Akay, M. F., et al. (2025). "Development of Salary Prediction Models for the Information Technology Industry"
- Ji, Y., et al. (2025). "Enhancing Job Salary Prediction with Disentangled Composition Effect Modeling"
- Stack Overflow Developer Survey Results (2025)

## 🔮 Future Work

- [ ] Model development (Linear Regression, Random Forest, XGBoost, etc.)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Model evaluation (RMSE, MAE, R²)
- [ ] SHAP/LIME interpretability analysis
- [ ] OCR-based CV parsing integration
- [ ] Streamlit/Flask web application

## 📝 License

This project is part of COMP 450 coursework. Please refer to the project proposal document for detailed information.

## 🤝 Contributing

This is a course project. For questions or suggestions, please contact the team members.

---

**Last Updated:** November 2025

