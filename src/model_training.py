"""
Machine Learning Model Training Module
=====================================
Trains the 5 syllabus-specified classification models for next-day direction prediction:
1. Logistic Regression (Unit II - Generalized Linear Model)
2. K-Nearest Neighbors (Unit III - Lazy Learning)
3. Naïve Bayes (Unit III - Probabilistic Classification)
4. Decision Tree Classifier (Unit III - Divide and Conquer Tree Induction)
5. Support Vector Machine (Unit III - Maximum Margin Classifier)

Applies strict chronological splitting and TimeSeriesSplit cross-validation.
"""

import os
import logging
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'Daily_Return',
    'Volatility_10',
    'MA_Ratio_5_20',
    'Close_to_MA5',
    'Close_to_MA20',
    'High_Low_Spread',
    'Open_Close_Return',
    'Return_Lag_1',
    'Return_Lag_2',
    'Return_Lag_3',
    'Price_Momentum_3D',
    'Volume_Ratio',
    'Day_Of_Week'
]


def prepare_datasets(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Split dataset chronologically to strictly respect temporal ordering (no lookahead leakage).
    Fit scaler exclusively on the training set.
    """
    # Drop rows where target is missing (last trading day per ticker)
    labeled_df = df.dropna(subset=['Target_Up']).copy()
    labeled_df['Date'] = pd.to_datetime(labeled_df['Date'])
    labeled_df = labeled_df.sort_values(by='Date').reset_index(drop=True)

    unique_dates = labeled_df['Date'].drop_duplicates().sort_values()
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates.iloc[split_idx]

    train_mask = labeled_df['Date'] < split_date
    test_mask = labeled_df['Date'] >= split_date

    train_data = labeled_df[train_mask]
    test_data = labeled_df[test_mask]

    X_train_raw = train_data[FEATURE_COLUMNS].values
    y_train = train_data['Target_Up'].astype(int).values

    X_test_raw = test_data[FEATURE_COLUMNS].values
    y_test = test_data['Target_Up'].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    logger.info(f"Chronological split date: {split_date.strftime('%Y-%m-%d')}")
    logger.info(f"Train samples: {len(X_train)} (Positive rate: {y_train.mean():.2%})")
    logger.info(f"Test samples:  {len(X_test)} (Positive rate: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, scaler, test_data


def get_syllabus_models() -> Dict[str, Any]:
    """Instantiate the 5 syllabus-mandated classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights='uniform',
            metric='minkowski',
            p=2
        ),
        "Naïve Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion='gini',
            random_state=42
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }


def train_and_cross_validate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Dict[str, Any]
) -> Dict[str, float]:
    """Evaluate training robustness using TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {}

    for name, model in models.items():
        fold_f1 = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_f1.append(f1_score(y_val, preds, zero_division=0))

        cv_scores[name] = float(np.mean(fold_f1))
        logger.info(f"Model [{name}] - 5-Fold TimeSeriesSplit Mean F1: {cv_scores[name]:.4f}")

    return cv_scores


def run_training_pipeline(
    input_path: str = "data/processed/features_engineered_stock_data.csv",
    output_dir: str = "models"
):
    """Execute complete model training and artifact serialization."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    X_train, X_test, y_train, y_test, scaler, test_data = prepare_datasets(df)
    models = get_syllabus_models()

    logger.info("Performing TimeSeriesSplit Cross Validation...")
    cv_scores = train_and_cross_validate_models(X_train, y_train, models)

    # Re-train models on entire training set and evaluate on test set
    test_predictions = {}
    test_probabilities = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        trained_models[name] = model
        test_predictions[name] = preds.tolist()
        test_probabilities[name] = probs.tolist()

    # Save scaler and models
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(trained_models, os.path.join(output_dir, "all_models.joblib"))

    # Save test dataset and predictions
    prediction_payload = {
        "y_true": y_test.tolist(),
        "predictions": test_predictions,
        "probabilities": test_probabilities,
        "cv_scores": cv_scores,
        "feature_names": FEATURE_COLUMNS,
        "test_dates": test_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "test_tickers": test_data['Ticker'].tolist()
    }

    with open(os.path.join(output_dir, "test_predictions.json"), "w") as f:
        json.dump(prediction_payload, f, indent=2)

    logger.info(f"Model training complete. All models saved in {output_dir}")


if __name__ == "__main__":
    run_training_pipeline()
