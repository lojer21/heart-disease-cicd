"""
Heart Disease Prediction — Random Forest Model Training
=======================================================
Trains the model from dataset/heart.csv and saves it as models/random_forest_model.pkl
This matches the Jupyter Notebook (Randomforest.ipynb) workflow.

Usage:
    python scripts/train_model.py

Output:
    models/random_forest_model.pkl  — raw sklearn RandomForestClassifier
"""

import os
import sys
import warnings
import pickle
import json
import joblib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "heart.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

print("=" * 60)
print("  HEART DISEASE PREDICTION — MODEL TRAINING")
print("=" * 60)

# ── Imports ──────────────────────────────────────────────────────
print("\n[1] Importing libraries...")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
print("    ✓ Libraries imported")

# ── Load Dataset ─────────────────────────────────────────────────
print(f"\n[2] Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Strip BOM if present in column name
df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

print(f"    ✓ Shape: {df.shape}")
print(f"    ✓ Columns: {list(df.columns)}")
print(f"    ✓ Missing values: {df.isnull().sum().sum()}")
print(f"    ✓ Duplicates: {df.duplicated().sum()}")

# Target distribution
target_dist = df['target'].value_counts()
print(f"    ✓ Class 0 (No disease): {target_dist.get(0, 0)}")
print(f"    ✓ Class 1 (Disease)   : {target_dist.get(1, 0)}")

# ── Preprocessing ────────────────────────────────────────────────
print("\n[3] Preprocessing...")
df = df.drop_duplicates().dropna()
print(f"    ✓ Records after cleaning: {len(df)}")

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[FEATURES]
y = df['target']

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=323, stratify=y
)
print(f"    ✓ Training: {len(X_train)} | Testing: {len(X_test)}")

# ── Baseline Model ───────────────────────────────────────────────
print("\n[4] Training baseline Random Forest...")
baseline = RandomForestClassifier(n_estimators=100, random_state=323)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"    ✓ Baseline accuracy: {baseline_acc * 100:.2f}%")

# ── GridSearchCV Tuning ──────────────────────────────────────────
print("\n[5] Hyperparameter tuning (GridSearchCV)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=323),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
print(f"    ✓ Best params: {grid.best_params_}")
print(f"    ✓ Best CV score: {grid.best_score_ * 100:.2f}%")

# ── Final Model Evaluation ───────────────────────────────────────
print("\n[6] Final model evaluation...")
model = grid.best_estimator_
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred).tolist()

print(f"    ✓ Tuned test accuracy : {test_acc * 100:.2f}%")
print(f"    ✓ CV score (tuning)   : {grid.best_score_ * 100:.2f}%")
print(f"    ✓ ROC-AUC             : {roc_auc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

# Feature Importance
feat_importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
sorted_fi = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
print("    Feature Importance (Top 5):")
for feat, imp in sorted_fi[:5]:
    print(f"      {feat:10s}: {imp:.4f}")

# ── Save Model ───────────────────────────────────────────────────
print(f"\n[7] Saving model to: {MODEL_PATH}")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"    ✓ Model saved: random_forest_model.pkl")

# Save metrics JSON (used by CI/CD validation)
metrics = {
    "baseline_accuracy": round(baseline_acc * 100, 2),
    "tuned_test_accuracy": round(test_acc * 100, 2),
    "best_cv_score": round(grid.best_score_ * 100, 2),
    "roc_auc": round(roc_auc, 4),
    "best_params": grid.best_params_,
    "feature_importance": feat_importance,
    "confusion_matrix": cm,
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "features": FEATURES,
    "random_state": 323,
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"    ✓ Metrics saved: metrics.json")

print("\n" + "=" * 60)
print("  ✓ TRAINING COMPLETE")
print(f"  Tuned Accuracy : {test_acc * 100:.2f}%")
print(f"  CV Score       : {grid.best_score_ * 100:.2f}%")
print(f"  ROC-AUC        : {roc_auc:.4f}")
print("=" * 60)
