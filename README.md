# Dubai House Price Predictor 🏠

<div align="center">

```
██████╗ ██╗   ██╗██████╗  █████╗ ██╗    ██╗  ██╗ ██████╗ ██╗   ██╗███████╗███████╗
██╔══██╗██║   ██║██╔══██╗██╔══██╗██║    ██║  ██║██╔═══██╗██║   ██║██╔════╝██╔════╝
██║  ██║██║   ██║██████╔╝███████║██║    ███████║██║   ██║██║   ██║███████╗█████╗  
██║  ██║██║   ██║██╔══██╗██╔══██║██║    ██╔══██║██║   ██║██║   ██║╚════██║██╔══╝  
██████╔╝╚██████╔╝██████╔╝██║  ██║██║    ██║  ██║╚██████╔╝╚██████╔╝███████║███████╗
╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
```

### Multiple Linear Regression Pipeline — From Dirty Data to Trained Model

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)](.)

<br/>

> **Median Imputation · Feature Engineering · OHE · StandardScaler · Residual Analysis · Pickle**

</div>

---

## What is This Project?

A complete **Multiple Linear Regression** pipeline that predicts housing prices using the California Housing Dataset (renamed Dubai for context). The project covers the full ML workflow from raw dirty data to a trained, serialized model — with proper assumption checks along the way.

- ✅ **Data Cleaning** — 207 missing values handled via median imputation (chosen over mean after visual comparison)
- ✅ **EDA** — Correlation heatmap identifying multicollinearity across all numeric features
- ✅ **Feature Engineering** — 3 new ratio features created, 4 raw count features dropped
- ✅ **Encoding** — OneHotEncoder with `drop='first'` on `ocean_proximity` (avoids dummy variable trap)
- ✅ **Scaling** — StandardScaler fit only on train, applied to test
- ✅ **Model Training** — Multiple Linear Regression with coefficient and intercept analysis
- ✅ **Assumption Checks** — Residuals vs Predicted, Residuals distribution (normality), Actual vs Predicted
- ✅ **Serialization** — Model pickled to `model.pkl` for reuse

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        RAW DATASET                           │
│      housing.csv · 20,640 rows · 10 columns · 207 nulls      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      DATA CLEANING                           │
│   Null check → Imputation comparison → Median fill           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                         EDA                                  │
│         Correlation heatmap · ocean_proximity counts         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│   rooms_per_household · bedrooms_per_room                    │
│   population_per_household · drop 4 raw count columns        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 ENCODING + SCALING                            │
│   OneHotEncoder (drop=first) · StandardScaler on X_train     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              MULTIPLE LINEAR REGRESSION                      │
│         fit → predict → MAE · RMSE · R² · Adj R²            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  ASSUMPTION CHECKS                           │
│   Residual plot · Normality check · Actual vs Predicted      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    PICKLE MODEL                              │
│              model.pkl saved for reuse                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Cleaning

### Imputation Strategy — Why Median over Mean

207 null values in `total_bedrooms` (out of 20,640 rows). Before choosing the imputation strategy, all three options were compared visually:

- **No imputation** — baseline distribution with missing gap
- **Mean imputation** (537.87) — pulls the distribution rightward due to outliers
- **Median imputation** (435.0) — preserves original shape, robust to skew

Median was selected as it better preserves the right-skewed distribution of `total_bedrooms`.

![Imputation Comparison — No Imputation vs Mean vs Median](ADD_IMPUTATION_SS_URL)

---

## EDA

### Correlation Heatmap

Key findings from the heatmap:

- `median_income` → `median_house_value` correlation: **0.69** — strongest predictor
- `total_rooms`, `total_bedrooms`, `population`, `households` are **highly intercorrelated** (0.86–0.97) — clear multicollinearity
- `longitude` ↔ `latitude`: **-0.92** — expected geographical inverse

This directly motivated the feature engineering step — replacing the 4 raw count columns with ratio features to break multicollinearity.

![Correlation Heatmap](ADD_HEATMAP_SS_URL)

---

## Feature Engineering

Instead of keeping raw count columns that are heavily correlated with each other, 3 ratio features were derived:

| New Feature | Formula | Why |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Captures density, not volume |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Bedroom ratio independent of size |
| `population_per_household` | `population / households` | Avg household occupancy |

Then `total_rooms`, `total_bedrooms`, `population`, `households` were dropped — eliminating the multicollinearity seen in the heatmap.

---

## Encoding

`ocean_proximity` has 5 categories: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`.

OneHotEncoder with `drop='first'` was applied — this drops `<1H OCEAN` as the reference category, producing 4 binary columns and avoiding the **dummy variable trap** (perfect multicollinearity between dummies).

---

## Model Results

```
MAE   =  53,132.66
RMSE  =  76,530.60
R²    =  0.5530
Adj R²=  0.5527
```

R² of 0.55 means the model explains ~55% of variance in house prices — reasonable for a baseline linear model on this dataset given the non-linearity in the data (visible in residual plots).

---

## Assumption Checks

### 1. Residuals vs Predicted
Used to check **homoscedasticity** (equal variance of errors). A good model shows residuals randomly scattered around 0.

The funnel pattern visible here indicates **heteroscedasticity** — variance increases as predicted value grows. This is a known limitation of linear regression on this dataset and suggests tree-based models would fit better.

![Residuals vs Predicted](ADD_RESIDUALS_SS_URL)

### 2. Actual vs Predicted
Scatter of `y_test` vs `y_pred`. A perfect model would show all points on a diagonal line. The vertical cluster at 500,001 reflects the dataset's price cap — all houses above the cap are recorded as 500,001, creating an artificial ceiling effect.

![Actual vs Predicted](ADD_ACTUAL_VS_PRED_SS_URL)

### 3. Residuals Distribution — Normality Check
Residuals should follow a normal distribution centered at 0 for linear regression assumptions to hold. The histogram with KDE shows the residuals are approximately normal and centered near 0 — confirming no systematic bias in predictions.

![Residuals Distribution — Normality Check](ADD_RESIDUALS_DIST_SS_URL)

---

## Input Features (Final)

| Feature | Type | Description |
|---|---|---|
| `longitude` | float | Geographic coordinate |
| `latitude` | float | Geographic coordinate |
| `housing_median_age` | float | Median age of housing block |
| `median_income` | float | Median income (tens of thousands) |
| `rooms_per_household` | float | Engineered — avg rooms per house |
| `bedrooms_per_room` | float | Engineered — bedroom ratio |
| `population_per_household` | float | Engineered — occupancy density |
| `ocean_proximity_INLAND` | int | OHE binary |
| `ocean_proximity_ISLAND` | int | OHE binary |
| `ocean_proximity_NEAR BAY` | int | OHE binary |
| `ocean_proximity_NEAR OCEAN` | int | OHE binary |

**Target:** `median_house_value`

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **Data** | Pandas, NumPy | Cleaning, feature engineering |
| **Visualization** | Matplotlib, Seaborn | EDA and assumption checks |
| **ML** | Scikit-Learn | Encoding, scaling, regression, metrics |
| **Serialization** | Pickle | Model persistence |

---

## Project Structure

```
Dubai-House-Price-Predictor/
│
├── notebooks/
│   └── housing_price_prediction.ipynb   # Full pipeline notebook
│
├── data/
│   ├── housing.csv                      # Raw dataset (dirty)
│   └── Dubai.csv                        # Cleaned dataset
│
├── model.pkl                            # Trained Linear Regression model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ayushhh026/Dubai-House-Price-Predictor.git
cd Dubai-House-Price-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/housing_price_prediction.ipynb
```

### 4. Load the saved model
```python
import pickle
model = pickle.load(open('model.pkl', 'rb'))
predictions = model.predict(X_test_scaled)
```

---

## Key Engineering Decisions

**Why median over mean for imputation?**
`total_bedrooms` is right-skewed (mean = 537, median = 435). Mean imputation pulls the distribution rightward, distorting the shape. Median is robust to outliers and preserves the original distribution — confirmed visually with the 3-panel comparison plot.

**Why ratio features over raw counts?**
The heatmap revealed `total_rooms`, `total_bedrooms`, `population`, and `households` are correlated at 0.86–0.97 — severe multicollinearity. Replacing them with ratio features (rooms per household, bedrooms per room, population per household) captures the same signal with far less redundancy.

**Why `drop='first'` in OneHotEncoder?**
With k categories, you only need k-1 dummy variables. Including all k creates perfect multicollinearity (the dummy variable trap), making the coefficient matrix singular and regression unstable.

**Why StandardScaler fit only on train?**
Fitting on the full dataset leaks test distribution statistics into training — a subtle but real form of data leakage. Scaler is fit on `X_train` only, then applied to `X_test`.

**Why Adjusted R² alongside R²?**
R² always increases when more features are added, even irrelevant ones. Adjusted R² penalizes for extra predictors — a fairer measure of model fit. Here R² = 0.5530, Adj R² = 0.5527, confirming the features added genuine value.

---

## Roadmap

- [ ] Try Ridge / Lasso / ElasticNet regularization
- [ ] XGBoost / Random Forest comparison
- [ ] Remove capped values (500,001) and retrain
- [ ] FastAPI deployment
- [ ] Interactive price prediction UI

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## Author

**Ayush Shetty**
AI & Data Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-ayushhh026-181717?style=flat-square&logo=github)](https://github.com/ayushhh026)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Shetty-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ayush-shetty-830a03281/)

---

<div align="center">

**⭐ Star this repo if it helped you — it keeps the project alive.**

*Clean data in, clean predictions out.*

</div>

---

> **Disclaimer:** This project uses the California Housing Dataset for educational purposes. The "Dubai" branding is for portfolio context only.
