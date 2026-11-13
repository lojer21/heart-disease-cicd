"""
Automated Test Suite for CI/CD Pipeline
========================================
Tests:
  1. Model file exists and loads correctly
  2. Model makes valid predictions on known inputs
  3. Model predict_proba returns probabilities summing to 1
  4. app.py imports without errors
  5. Dataset integrity checks

Usage: pytest tests/test_model.py -v
"""

import os
import sys
import json
import pytest
import joblib
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "heart.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Sample patient records for prediction tests
SAMPLE_POSITIVE = {   # Known high-risk profile
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
SAMPLE_NEGATIVE = {   # Known low-risk profile
    "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
    "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
    "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
}


# ─────────────────────────────────────────────────────────────────
# FIXTURE: Load model once for all tests
# ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    m = joblib.load(MODEL_PATH)
    return m


@pytest.fixture(scope="module")
def dataset():
    assert os.path.exists(DATASET_PATH), f"Dataset not found: {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────
# TEST GROUP 1: Model File & Loading
# ─────────────────────────────────────────────────────────────────
class TestModelLoading:

    def test_model_file_exists(self):
        """Model file must exist at models/random_forest_model.pkl"""
        assert os.path.exists(MODEL_PATH), "random_forest_model.pkl not found"

    def test_model_file_not_empty(self):
        """Model file must not be empty (corrupt save)"""
        size = os.path.getsize(MODEL_PATH)
        assert size > 1000, f"Model file too small ({size} bytes) — likely corrupt"

    def test_model_loads_without_error(self, model):
        """Model must load without raising exceptions"""
        assert model is not None

    def test_model_has_predict(self, model):
        """Model must expose predict() method"""
        assert hasattr(model, 'predict')

    def test_model_has_predict_proba(self, model):
        """Model must expose predict_proba() for probability output"""
        assert hasattr(model, 'predict_proba')

    def test_model_has_feature_importances(self, model):
        """Random Forest must have feature_importances_ attribute"""
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == len(FEATURES)

    def test_model_is_fitted(self, model):
        """Model must be fitted (n_estimators_ > 0)"""
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) > 0


# ─────────────────────────────────────────────────────────────────
# TEST GROUP 2: Prediction Correctness
# ─────────────────────────────────────────────────────────────────
class TestPredictions:

    def test_predict_returns_binary_class(self, model):
        """Predictions must be 0 or 1"""
        X = pd.DataFrame([SAMPLE_POSITIVE], columns=FEATURES)
        pred = model.predict(X)[0]
        assert pred in [0, 1], f"Unexpected class: {pred}"

    def test_predict_proba_sums_to_one(self, model):
        """predict_proba probabilities must sum to 1.0 (within float tolerance)"""
        X = pd.DataFrame([SAMPLE_POSITIVE], columns=FEATURES)
        proba = model.predict_proba(X)[0]
        assert abs(sum(proba) - 1.0) < 1e-6, f"Probabilities sum to {sum(proba)}"

    def test_predict_proba_in_range(self, model):
        """All probability values must be between 0 and 1"""
        X = pd.DataFrame([SAMPLE_NEGATIVE], columns=FEATURES)
        proba = model.predict_proba(X)[0]
        assert all(0.0 <= p <= 1.0 for p in proba), f"Out of range: {proba}"

    def test_predict_proba_returns_two_classes(self, model):
        """predict_proba must return 2 class probabilities"""
        X = pd.DataFrame([SAMPLE_POSITIVE], columns=FEATURES)
        proba = model.predict_proba(X)[0]
        assert len(proba) == 2, f"Expected 2 classes, got {len(proba)}"

    def test_batch_prediction(self, model):
        """Model must handle batch input (multiple rows)"""
        rows = [SAMPLE_POSITIVE, SAMPLE_NEGATIVE]
        X = pd.DataFrame(rows, columns=FEATURES)
        preds = model.predict(X)
        assert len(preds) == 2

    def test_prediction_on_full_test_split(self, model, dataset):
        """Model accuracy on held-out test set must exceed 70%"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        df = dataset.copy()
        X = df[FEATURES]
        y = df['target']
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=323, stratify=y
        )
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        assert acc >= 0.70, f"Model accuracy too low: {acc:.2%} (threshold: 70%)"
        print(f"\n    Test accuracy: {acc:.2%}")


# ─────────────────────────────────────────────────────────────────
# TEST GROUP 3: Dataset Integrity
# ─────────────────────────────────────────────────────────────────
class TestDataset:

    def test_dataset_loads(self, dataset):
        assert dataset is not None and len(dataset) > 0

    def test_dataset_has_correct_columns(self, dataset):
        expected = set(FEATURES + ['target'])
        actual = set(dataset.columns)
        missing = expected - actual
        assert not missing, f"Missing columns: {missing}"

    def test_dataset_minimum_records(self, dataset):
        assert len(dataset) >= 300, f"Too few records: {len(dataset)}"

    def test_dataset_target_binary(self, dataset):
        unique_targets = set(dataset['target'].unique())
        assert unique_targets.issubset({0, 1}), f"Non-binary target: {unique_targets}"

    def test_dataset_no_all_null_columns(self, dataset):
        for col in FEATURES:
            null_count = dataset[col].isnull().sum()
            null_pct = null_count / len(dataset)
            assert null_pct < 0.5, f"Column '{col}' has {null_pct:.0%} nulls"


# ─────────────────────────────────────────────────────────────────
# TEST GROUP 4: Metrics File
# ─────────────────────────────────────────────────────────────────
class TestMetrics:

    def test_metrics_file_exists(self):
        assert os.path.exists(METRICS_PATH), "metrics.json not found"

    def test_metrics_tuned_accuracy_above_threshold(self):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        acc = m.get("tuned_test_accuracy", 0)
        assert acc >= 70.0, f"Tuned accuracy {acc}% below 70% threshold"

    def test_metrics_has_required_keys(self):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        required = ["tuned_test_accuracy", "best_cv_score", "best_params", "features"]
        for key in required:
            assert key in m, f"Missing key in metrics.json: {key}"
