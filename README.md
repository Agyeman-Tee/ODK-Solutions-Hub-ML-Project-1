# 🎓 Student Grade Predictor — ML Project 1

> Predicting student academic performance using Machine Learning  
> **Supervised by Obeng Dominic Koranteng** — Lead Tutor & Researcher, ODK Solutions Hub

---

## 👩‍🎓 Project Author

**Augustina Agyapomaa Agyeman**  
ML Project 1 | ODK Solutions Hub Training Programme

---

## 👨‍🏫 Supervisor

**Obeng Dominic Koranteng**  
Lead Tutor & Researcher — [ODK Solutions Hub](https://odk-tech.github.io/ODK-Solutions-Hub/)

---

## 📌 Project Overview

This project builds a machine learning pipeline to classify students into GPA performance categories based on demographic, social, and academic features drawn from the Student Mathematics dataset.

Three GPA classes are predicted:

| Class | Final Grade Range | Meaning |
|---|---|---|
| `Pass` | 0 – 9 | Below average / marginal pass |
| `2nd_Class` | 10 – 15 | Average to good performance |
| `1st_Class` | 16 – 20 | Excellent performance |

---

## 🗂️ Project Structure

```
ml-project-1/
├── app.py                    # Streamlit web application
├── ML_Nic_Corrected.ipynb    # Corrected Jupyter notebook
├── requirements.txt          # Python dependencies
├── student_math_clean.csv    # Dataset (place in root folder)
└── README.md                 # This file
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, missing value audit, and descriptive statistics
- Final grade distribution histogram
- GPA category class distribution
- Correlation heatmap of numeric features

### 2. Feature Engineering
- Target variable (`final_grade`) bucketed into three GPA categories
- Categorical features one-hot encoded with `drop_first=True` to avoid multicollinearity
- `student_id` excluded to prevent identifier leakage

### 3. Models Trained
| Model | Preprocessing | Notes |
|---|---|---|
| **Logistic Regression** | `StandardScaler` via `Pipeline` | Scale-sensitive; scaler fit on train data only |
| **Random Forest** | None (scale-invariant) | 200 estimators, `random_state=42` |

### 4. Evaluation
- Accuracy, Precision, Recall, F1-score (per class)
- Confusion matrix with corrected class labels
- 5-fold Stratified Cross-Validation for reliable generalisation estimates
- Side-by-side model comparison table

---

## 📊 Key Corrections Applied

This project addresses several common ML pitfalls, including:

- ✅ **Data leakage prevention** — `StandardScaler` wrapped in a `sklearn.Pipeline` so it never sees test data
- ✅ **Label alignment** — Confusion matrix and distribution charts use `le_y.classes_` (encoder-derived) instead of hardcoded strings
- ✅ **`np.bincount` safety** — `minlength` argument added to prevent crashes when a class is absent from predictions
- ✅ **Cross-validation** — Single train/test split supplemented with `StratifiedKFold` CV
- ✅ **Portable paths** — Absolute Windows path replaced with `pathlib.Path` for cross-platform compatibility
- ✅ **Clean imports** — Unused `import math` removed; duplicate imports consolidated

---

## 🌐 Streamlit App

The model is deployed as an interactive web app with four sections:

| Tab | Description |
|---|---|
| 📊 EDA | Dataset exploration and visualisations |
| 🤖 Model Results | Confusion matrices, classification reports, model comparison |
| 🔍 Feature Importance | Top-10 most predictive features |
| 🎯 Predict a Student | Live prediction form with class probabilities |

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## 📁 Dataset

The dataset used is a cleaned version of the **Student Performance (Mathematics)** dataset. Each row represents a student with features covering:

- **Demographics** — age, sex, address type, family size
- **Family background** — parent status, mother/father education and occupation
- **Academic history** — prior grades (`grade_1`, `grade_2`), study failures
- **Social factors** — free time, social activity, alcohol consumption
- **School-related** — study time, school support, absences, health

---

## 📜 Acknowledgements

Special thanks to **Obeng Dominic Koranteng** and the entire team at **ODK Solutions Hub** for their mentorship, curriculum design, and continuous support throughout this ML training programme.

---

*ML Project 1 · ODK Solutions Hub Training Programme*
