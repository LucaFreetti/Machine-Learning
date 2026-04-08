# Titanic Survival Classifier 🚢

Binary classification project that predicts whether a Titanic passenger survived, based on features like sex, age, passenger class, and port of embarkation.

## Overview

The project follows a complete machine learning pipeline:

1. **Exploratory Data Analysis (EDA)** — survival rates by sex, class, and port; age distribution; correlation analysis
2. **Preprocessing** — median imputation for `Age`, mode imputation + OneHotEncoding for `Sex` and `Embarked`, all wrapped in a `ColumnTransformer` pipeline
3. **Model selection** — `DecisionTreeClassifier` with depth tuning (2, 5, 10, 25, None) on a held-out validation set
4. **Final evaluation** — retrain on train + val, evaluate on unseen test set

## Dataset

The dataset (`titanic_sub.csv`) contains **891 records** with the following features:

| Feature | Type | Notes |
|---|---|---|
| `Pclass` | int | Passenger class (1, 2, 3) |
| `Sex` | str | male / female |
| `Age` | float | ~20% missing → imputed with median |
| `Embarked` | str | C / Q / S — 2 missing → imputed with mode |
| `Survived` | int | Target variable (0 = No, 1 = Yes) |

## Results

| Metric | Value |
|---|---|
| Validation accuracy (`max_depth=5`) | 80.24% |
| **Test accuracy** | **81.17%** |
| Macro F1-score | 0.79 |

The model clearly outperforms the naive baseline of always predicting "Not Survived" (~62%).  
The main weakness is recall on survivors (0.64) — the minority class — which is expected given the class imbalance.

## Key Design Decisions

- **Three-way split** (train / val / test): the test set is never used for tuning, ensuring the final score is unbiased.
- **Median imputation for `Age`**: the age distribution is right-skewed, so the median is more robust than the mean.
- **`drop='first'` in OneHotEncoder**: avoids redundant dummy columns (dummy variable trap).
- **`fit_transform` only on train**: prevents data leakage — imputation statistics come exclusively from training data.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `DecisionTreeClassifier`)
- seaborn, matplotlib

## How to Run

```bash
pip install pandas scikit-learn seaborn matplotlib
jupyter notebook LucaFrittittaMLI.ipynb
```

> Update the CSV path in the data loading cell to point to your local copy of `titanic_sub.csv`.

## Project Structure

```
├── LucaFrittittaMLI.ipynb   # Main notebook
└── titanic_sub.csv          # Dataset (not included)
```
