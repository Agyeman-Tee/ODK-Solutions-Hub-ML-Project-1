"""
Student Grade Predictor — Streamlit App
Converted from ML_Nic_Corrected.ipynb
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { color: #ffffff; font-size: 2.4rem; margin: 0 0 0.4rem 0; }
    .hero p  { color: #a8c8d8; font-size: 1.05rem; margin: 0; }

    .metric-card {
        background: #f7f9fc;
        border-left: 4px solid #2c5364;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 600; color: #0f2027; }

    .stTabs [data-baseweb="tab"] { font-weight: 500; font-size: 0.95rem; }

    .fix-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 4px;
        font-size: 0.75rem;
        padding: 2px 8px;
        font-weight: 600;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 Student Grade Predictor</h1>
    <p>Augustina Agyapomaa Agyeman · Project 1 · Supervisor: Obeng Dominic Koranteng</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ── Helper: load & preprocess data ──────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_prepare(uploaded_file):
    df = pd.read_csv(uploaded_file)

    def gpa_category(gpa):
        if gpa <= 9:   return "Pass"
        elif gpa <= 15: return "2nd_Class"
        else:           return "1st_Class"

    df["GPA_Category"] = df["final_grade"].apply(gpa_category)
    return df


@st.cache_resource
def train_models(df):
    X = df.drop(columns=["final_grade", "GPA_Category", "student_id"])
    y = df["GPA_Category"]

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    X_enc = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Logistic Regression pipeline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cv = cross_val_score(lr_pipeline, X_enc, y_enc, cv=cv, scoring="accuracy")
    rf_cv = cross_val_score(rf,          X_enc, y_enc, cv=cv, scoring="accuracy")

    return {
        "le_y": le_y,
        "X_enc": X_enc,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "lr_pipeline": lr_pipeline, "y_pred_lr": y_pred_lr,
        "rf": rf,            "y_pred_rf": y_pred_rf,
        "lr_cv": lr_cv,      "rf_cv": rf_cv,
    }


# ════════════════════════════════════════════════════════════════════════════
# ── Sidebar: data upload ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload student_math_clean.csv", type="csv")
    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit")

if uploaded is None:
    st.info("👈 Upload your **student_math_clean.csv** in the sidebar to begin.")
    st.stop()

# ── Load & train ─────────────────────────────────────────────────────────────
df   = load_and_prepare(uploaded)
res  = train_models(df)
le_y = res["le_y"]

# ════════════════════════════════════════════════════════════════════════════
# ── Tabs ─────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA", "🤖 Model Results", "🔍 Feature Importance", "🎯 Predict a Student"
])


# ══════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", df.isna().sum().sum())

    with st.expander("📋 Raw data preview"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("📈 Descriptive statistics"):
        st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # Grade distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Final Grade Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["final_grade"], bins=20, edgecolor="black", color="#2c5364")
        ax.set_xlabel("Final Grade")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Final Grades")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**GPA Category Distribution**")
        class_counts = df["GPA_Category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(class_counts.index, class_counts.values,
                      color=["#2196F3", "#4CAF50", "#FF9800"], edgecolor="black")
        ax.set_xlabel("GPA Category")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution of GPA Categories")
        for bar, val in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("**Correlation Heatmap (numeric features)**")
    numeric_cols = df.select_dtypes(include="number").drop(columns=["student_id"])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════
# TAB 2 — Model Results
# ══════════════════════════════════════
with tab2:
    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model":            ["Logistic Regression", "Random Forest"],
        "Test Accuracy":    [
            round(accuracy_score(res["y_test"], res["y_pred_lr"]), 4),
            round(accuracy_score(res["y_test"], res["y_pred_rf"]), 4),
        ],
        "CV Mean Accuracy": [round(res["lr_cv"].mean(), 4), round(res["rf_cv"].mean(), 4)],
        "CV Std":           [round(res["lr_cv"].std(),  4), round(res["rf_cv"].std(),  4)],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    for col, name, y_pred in [
        (col1, "Logistic Regression", res["y_pred_lr"]),
        (col2, "Random Forest",       res["y_pred_rf"]),
    ]:
        with col:
            st.markdown(f"**{name}**")

            # Confusion matrix
            cm = confusion_matrix(res["y_test"], y_pred)
            classes = le_y.classes_
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, rotation=30, ha="right")
            ax.set_yticklabels(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Classification report
            report = classification_report(
                res["y_test"], y_pred, target_names=le_y.classes_, output_dict=True
            )
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

    st.divider()
    st.subheader("True vs Predicted Distribution (Random Forest)")
    n_classes   = len(le_y.classes_)
    true_counts = np.bincount(res["y_test"],    minlength=n_classes)
    pred_counts = np.bincount(res["y_pred_rf"], minlength=n_classes)
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, true_counts, width=0.4, label="True",      color="#2c5364")
    ax.bar(x + 0.2, pred_counts, width=0.4, label="Predicted", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(le_y.classes_)
    ax.set_xlabel("GPA Category")
    ax.set_ylabel("Number of Students")
    ax.set_title("True vs Predicted GPA Category Distribution")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════
# TAB 3 — Feature Importance
# ══════════════════════════════════════
with tab3:
    st.subheader("Top 10 Feature Importances (Random Forest)")
    rf         = res["rf"]
    X_enc      = res["X_enc"]
    importances = rf.feature_importances_
    indices     = np.argsort(importances)
    top_idx     = indices[-10:]
    features    = np.array(X_enc.columns)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features[top_idx], importances[top_idx], color="#2c5364", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features Affecting Final Grade Category")
    for bar, val in zip(bars, importances[top_idx]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Full feature importance table"):
        fi_df = pd.DataFrame({
            "Feature":    features,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        st.dataframe(fi_df, use_container_width=True)


# ══════════════════════════════════════
# TAB 4 — Predict a Student
# ══════════════════════════════════════
with tab4:
    st.subheader("🎯 Predict GPA Category for a New Student")
    st.caption("Fill in the student's details and click **Predict** to see which GPA class the models assign.")

    # Build input widgets from training column names
    X_enc   = res["X_enc"]
    rf      = res["rf"]
    lr_pipe = res["lr_pipeline"]

    # We'll collect raw numeric inputs for the features that exist in X_enc
    # (post one-hot). We reconstruct a dummy row matching the training columns.
    numeric_defaults = {
        "age": 17, "mother_education": 2, "father_education": 2,
        "study_time": 2, "failures": 0, "family_relationship": 3,
        "free_time": 3, "social": 3, "weekday_alcohol": 1,
        "weekend_alcohol": 1, "health": 3, "absences": 4,
        "grade_1": 10, "grade_2": 10,
    }

    with st.form("predict_form"):
        cols = st.columns(3)
        user_inputs = {}
        numeric_features = [c for c in numeric_defaults if c in df.columns]
        for i, feat in enumerate(numeric_features):
            default = numeric_defaults[feat]
            user_inputs[feat] = cols[i % 3].number_input(
                feat.replace("_", " ").title(),
                min_value=0, max_value=100,
                value=default, step=1
            )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Build a one-row DataFrame matching training schema
        row = {col: 0 for col in X_enc.columns}
        for feat, val in user_inputs.items():
            if feat in row:
                row[feat] = val
        X_new = pd.DataFrame([row])

        rf_pred  = rf.predict(X_new)[0]
        lr_pred  = lr_pipe.predict(X_new)[0]
        rf_label = le_y.inverse_transform([rf_pred])[0]
        lr_label = le_y.inverse_transform([lr_pred])[0]

        rf_proba = rf.predict_proba(X_new)[0]
        lr_proba = lr_pipe.predict_proba(X_new)[0]

        st.success("✅ Prediction complete!")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🌲 Random Forest\n**Predicted class:** `{rf_label}`")
            proba_df = pd.DataFrame({
                "Class":       le_y.classes_,
                "Probability": rf_proba.round(4)
            })
            st.dataframe(proba_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown(f"### 📈 Logistic Regression\n**Predicted class:** `{lr_label}`")
            proba_df2 = pd.DataFrame({
                "Class":       le_y.classes_,
                "Probability": lr_proba.round(4)
            })
            st.dataframe(proba_df2, use_container_width=True, hide_index=True)
