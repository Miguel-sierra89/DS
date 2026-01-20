# AI Copilot Instructions for Spotify Popularity Prediction

## Project Overview
Binary classification project predicting track popularity from Spotify audio features. The pipeline flows: raw data → preprocessing → EDA → modeling.

**Data Flow**: `data/raw/*.csv` → [preprocessing.ipynb] → `data/processed/{X_scaled.csv, y.csv}` → [modeling.ipynb]

## Architecture & Notebooks

The project is organized as three sequential Jupyter notebooks with specific responsibilities:

1. **preprocessing.ipynb**: Data cleaning, feature engineering, target creation
   - Removes duplicate tracks (keeps highest popularity version)
   - Caps outliers on `duration_ms`, `tempo`, `loudness` using IQR method
   - Creates binary target: `is_popular = popularity >= 75th percentile`
   - One-hot encodes categorical features (`key`, `mode`, `time_signature`)
   - Standardizes numerical features with `StandardScaler`
   - Outputs: `X_scaled.csv` (features), `y.csv` (target)

2. **eda.ipynb**: Exploratory data analysis
   - Checks for missing values and data types
   - Generates distribution plots, correlation matrices
   - Identifies data quality issues that inform preprocessing decisions

3. **modeling.ipynb**: Model training and evaluation
   - Loads preprocessed data: `X_scaled.csv`, `y.csv`
   - Splits with `train_test_split(test_size=0.2, random_state=42, stratify=y)`
   - Tests multiple models: Logistic Regression (baseline), Random Forest, Gradient Boosting
   - Evaluates with ROC-AUC score and classification reports

## Key Development Patterns

### File Paths
Always use **relative paths from notebook directory**:
- Input: `../data/raw/SpotifyAudioFeaturesApril2019.csv`
- Output: `../data/processed/X_scaled.csv`
- Do NOT use absolute paths

### Train-Test Split Convention
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
Stratification preserves class distribution. **Always include `random_state=42`** for reproducibility.

### Data Preprocessing
- Target is **binary classification** (not regression)—popularity is dichotomized at 75th percentile
- Numerical features are StandardScaled; categorical features are one-hot encoded
- Duplicates resolved by keeping highest popularity (proxy for most current data)

### Model Evaluation
Always report:
- `roc_auc_score()` for primary metric
- `classification_report()` for precision/recall/F1
- Predictions stored as `y_pred_*` and probabilities as `y_proba_*`

### Variable Naming
- Features DataFrame: `X` (or `X_train`, `X_test`)
- Target Series: `y` (or `y_train`, `y_test`)
- Model instances: `log_reg`, `rf`, `gb` (abbreviations)
- Predictions: `y_pred_*` (binary), `y_proba_*` (probabilities)

## Dependencies
Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

Full environment specified in `requirements.txt` (includes JupyterLab). Install with:
```
pip install -r requirements.txt
```

## Common Tasks

**Adding new models**: Follow the pattern in modeling.ipynb—train → predict → evaluate with ROC-AUC and classification_report.

**Modifying preprocessing**: Edit preprocessing.ipynb before running modeling.ipynb. Regenerate `X_scaled.csv` and `y.csv` to ensure consistency.

**Running notebooks**: Execute sequentially: preprocessing → eda → modeling. Each depends on preceding outputs.

**Debugging**: Check pandas dtypes with `.info()`, distributions with `.describe()`, missing values with `.isnull().sum()`.
