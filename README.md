# Titanic Survival Classifier 🚢

Binary classification project that predicts whether a Titanic passenger survived,
using a Decision Tree Classifier tuned via a train/validation/test split strategy.

---

## Objective

Predict survival from features like sex, age, passenger class, and port of embarkation.
The goal is to identify the best `max_depth` hyperparameter and evaluate final performance
on completely unseen test data.

---

## Dataset

The dataset (`titanic_sub.csv`) contains **891 passengers** and **5 features**,
with an overall survival rate of ~38%.

| Feature | Type | Notes |
|---|---|---|
| `Sex` | Categorical | One-Hot Encoded → `Sex_female` / `Sex_male` |
| `Age` | Numeric | 177 missing values → imputed with training median |
| `Pclass` | Ordinal | 1st, 2nd, 3rd class |
| `Embarked` | Categorical | One-Hot Encoded → `Embarked_C` / `Embarked_Q` / `Embarked_S` |
| `Survived` | Target | 0 = No, 1 = Yes |

---

## Exploratory Data Analysis

Key patterns found before modeling:

- **Sex** is the strongest predictor: women survived at **74%** vs **19%** for men.
- **Passenger class** had a decisive impact: 1st class **63%** | 2nd **47%** | 3rd **24%**.
- **Port of embarkation**: Cherbourg (C) **55%** | Queenstown (Q) **39%** | Southampton (S) **34%**.
- **Age**: younger passengers had slightly higher survival rates; distribution is right-skewed
  (median = 28, mean = 30), which informed the imputation strategy.

---

## Pipeline

### 1. Train / Val / Test Split
Split performed *before* any preprocessing to prevent data leakage.

75% train+val → 25% test
then: 75% train → 25% val
Final: ~501 train / ~167 val / ~223 test

### 2. Preprocessing (`ColumnTransformer`)
- **Numeric** (`Age`, `Pclass`): median imputation (robust to right-skewed distribution)
- **Categorical** (`Sex`, `Embarked`): mode imputation + `OneHotEncoder(drop='first')`
- `fit_transform()` applied **only on training data** — no leakage into val/test

### 3. Model Tuning — Finding the Best Depth

| `max_depth` | Train Acc | Val Acc |
|---|---|---|
| 2 | ~0.79 | ~0.79 |
| **5** | **~0.85** | **0.8024 ✅** |
| 10 | ~0.91 | ~0.78 |
| 25 | ~0.91 | ~0.78 |
| None | ~0.91 | ~0.778 |

`max_depth=5` selected: best validation accuracy. Deeper trees show classic overfitting —
training accuracy climbs while validation drops.

### 4. Final Evaluation
Model retrained on train + val combined, evaluated on the held-out test set.

---

## Results

**Test accuracy: 81.17%** vs naive baseline of ~62% (always predict "Not Survived")

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not Survived | 0.81 | 0.91 | 0.86 | 139 |
| Survived | 0.82 | 0.64 | 0.72 | 84 |
| **Macro avg** | **0.81** | **0.78** | **0.79** | 223 |
| Weighted avg | 0.81 | 0.81 | 0.81 | 223 |

The model is better at identifying non-survivors (recall 0.91) than survivors (recall 0.64),
which is expected given class imbalance. The macro F1 of 0.79 confirms reasonable
performance on both classes.

---

## Tech Stack

- Python 3
- `pandas`, `numpy`
- `scikit-learn` — `Pipeline`, `ColumnTransformer`, `DecisionTreeClassifier`
- `seaborn`, `matplotlib`

---

## How to Run

```bash
pip install pandas scikit-learn seaborn matplotlib
jupyter notebook LucaFrittittaMLI.ipynb
```

> Update the CSV path in the data loading cell to point to your local copy of `titanic_sub.csv`.

---

## Project Structure

├── LucaFrittittaMLI.ipynb              # Main notebook
├── LucaFrittitta_Titanic_ML.pdf        # Project presentation
└── titanic_sub.csv                     # Dataset (not included)

---

## Possible Next Steps

- Try ensemble methods: **Random Forest**, **Gradient Boosting**
- Address class imbalance with `class_weight='balanced'`
- Feature engineering: family size (`SibSp` + `Parch`), title extraction from `Name`
- Cross-validation instead of a single val split
