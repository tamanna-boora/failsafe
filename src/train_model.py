"""
train_model.py
Loads the student dataset, encodes categorical features, trains an XGBoost
classifier to predict at-risk students, evaluates it, and saves all
artifacts to models/ for use by the Streamlit app.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "students.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess(path: str):
    """Load CSV, encode categoricals, return X, y, encoders, feature_names."""
    df = pd.read_csv(path)

    # G3 is the source of truth for at_risk; drop it to prevent data leakage.
    # G1 and G2 (first/second period grades) are legitimate predictors kept as features.
    TARGET     = "at_risk"
    DROP_COLS  = ["G3", TARGET]
    feature_df = df.drop(columns=DROP_COLS)

    # Detect categorical columns (object dtype)
    cat_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col] = le

    X = feature_df
    y = df[TARGET]
    return X, y, encoders, list(X.columns)


def train():
    print(f"Loading data from {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        sys.exit("ERROR: data/students.csv not found. Run download_data.py first.")

    X, y, encoders, feature_names = load_and_preprocess(DATA_PATH)

    # 80/20 stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"At-risk in train: {y_train.sum()} ({100*y_train.mean():.1f}%)")

    # XGBoost with scale_pos_weight to handle class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost classifier ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print("  MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not At-Risk", "At-Risk"]))

    # ── Save artifacts ────────────────────────────────────────────────────
    model_path    = os.path.join(MODELS_DIR, "xgb_model.joblib")
    encoder_path  = os.path.join(MODELS_DIR, "label_encoders.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    metrics_path  = os.path.join(MODELS_DIR, "metrics.joblib")

    joblib.dump(model,        model_path)
    joblib.dump(encoders,     encoder_path)
    joblib.dump(feature_names, features_path)
    joblib.dump({"accuracy": acc, "f1": f1, "confusion_matrix": cm}, metrics_path)

    print(f"\nArtifacts saved to {MODELS_DIR}/")
    print(f"  xgb_model.joblib, label_encoders.joblib, feature_names.joblib, metrics.joblib")

    return model, encoders, feature_names


if __name__ == "__main__":
    train()
