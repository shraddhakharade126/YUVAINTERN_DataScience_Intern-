"""
Week 3: Python-Based Machine Learning Model Development and Evaluation
Customer Churn Prediction — Reference Implementation

This script operationalizes the plan described in
"Week 3: Python-Based Machine Learning Model Development and Evaluation Plan":
  1. Problem definition (30-day binary churn classification)
  2. Preprocessing (cleaning, scaling, feature engineering)
  3. Model selection & training (Logistic Regression, Random Forest, Gradient Boosting)
  4. Hyperparameter tuning (randomized search, cross-validation)
  5. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix)
  6. Model export for conceptual deployment

Expected input: a CSV of customer records with a binary target column named
`churned` (1 = churned within the 30-day window, 0 = retained) and a date
column `snapshot_date` used for the chronological train/validation/test split.
Replace CONFIG values and the feature lists below to match your real schema.

Usage:
    python churn_model_pipeline.py --data customers.csv --model gboost
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    target_col: str = "churned"
    date_col: str = "snapshot_date"

    # Update these to match your real feature set.
    numeric_features: list[str] = field(default_factory=lambda: [
        "tenure_months",
        "login_frequency_30d",
        "login_frequency_90d",
        "feature_usage_score",
        "payment_delay_count",
        "support_ticket_count_90d",
        "monthly_spend",
    ])
    categorical_features: list[str] = field(default_factory=lambda: [
        "plan_type",
        "acquisition_channel",
    ])

    # Chronological split ratios (train+cv / validation / test)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    n_splits: int = 5
    random_state: int = 42
    n_search_iter: int = 25


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(cfg: Config) -> ColumnTransformer:
    """Builds the preprocessing pipeline: imputation + scaling for numeric
    features, imputation + one-hot encoding for categorical features. Fit
    only on training folds to avoid leakage, per the Week 3 plan."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, cfg.numeric_features),
        ("cat", categorical_pipeline, cfg.categorical_features),
    ])


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds derived features described in Section 4.3 of the plan. All
    calculations must use only information available before the prediction
    cutoff date to prevent leakage."""
    df = df.copy()

    if {"login_frequency_30d", "login_frequency_90d"}.issubset(df.columns):
        df["engagement_trend"] = (
            df["login_frequency_30d"] - df["login_frequency_90d"] / 3
        )

    if "tenure_months" in df.columns:
        df["tenure_band"] = pd.cut(
            df["tenure_months"],
            bins=[-np.inf, 3, 12, np.inf],
            labels=["0-3m", "4-12m", "12m+"],
        ).astype(str)
        if "tenure_band" not in cfg.categorical_features:
            cfg.categorical_features.append("tenure_band")

    if {"monthly_spend", "feature_usage_score"}.issubset(df.columns):
        df["usage_per_dollar"] = df["feature_usage_score"] / df["monthly_spend"].replace(0, np.nan)
        if "usage_per_dollar" not in cfg.numeric_features:
            cfg.numeric_features.append("usage_per_dollar")

    return df


def chronological_split(df: pd.DataFrame, cfg: Config):
    """Splits data by snapshot_date so validation/test always follow
    training in time, consistent with Figure 2 in the plan."""
    df_sorted = df.sort_values(cfg.date_col)
    n = len(df_sorted)
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Models & hyperparameter search spaces (Section 5)
# --------------------------------------------------------------------------- #
def get_model_and_grid(name: str, cfg: Config):
    if name == "logreg":
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=cfg.random_state
        )
        grid = {"clf__C": np.logspace(-3, 2, 20)}
    elif name == "rf":
        model = RandomForestClassifier(
            class_weight="balanced", random_state=cfg.random_state
        )
        grid = {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_leaf": [1, 2, 5, 10],
        }
    elif name == "gboost":
        model = GradientBoostingClassifier(random_state=cfg.random_state)
        grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [2, 3, 4],
            "clf__subsample": [0.6, 0.8, 1.0],
        }
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model, grid


# --------------------------------------------------------------------------- #
# Training & tuning (Section 5.3, 5.4)
# --------------------------------------------------------------------------- #
def train_and_tune(X_train, y_train, cfg: Config, model_name: str):
    preprocessor = build_preprocessor(cfg)
    model, grid = get_model_and_grid(model_name, cfg)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", model),
    ])

    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid,
        n_iter=cfg.n_search_iter,
        scoring="average_precision",  # PR-AUC: appropriate for imbalanced churn data
        cv=cv,
        random_state=cfg.random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


# --------------------------------------------------------------------------- #
# Evaluation (Section 6)
# --------------------------------------------------------------------------- #
def evaluate(model, X, y, split_name: str, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    print(f"\n--- {split_name} set ({len(y)} rows) ---")
    print(classification_report(y, preds, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}  |  PR-AUC: {metrics['pr_auc']:.3f}")
    return metrics


def find_best_threshold(model, X_val, y_val, thresholds=None) -> float:
    """Sweeps thresholds on the validation set and returns the one
    maximizing F1, as a simple stand-in for a full business cost analysis
    (Section 6.2). Replace with a cost-weighted objective when business
    costs for false positives/negatives are known."""
    thresholds = thresholds or np.arange(0.1, 0.9, 0.02)
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Selected threshold = {best_t:.2f} (validation F1 = {best_f1:.3f})")
    return best_t


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a churn prediction model.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV of customer records.")
    parser.add_argument(
        "--model", type=str, default="gboost", choices=["logreg", "rf", "gboost"],
        help="Which candidate model family to train.",
    )
    parser.add_argument("--out-dir", type=str, default="model_output", help="Where to save artifacts.")
    args = parser.parse_args()

    cfg = Config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & engineer features
    df = pd.read_csv(args.data, parse_dates=[cfg.date_col])
    df = engineer_features(df, cfg)

    # 2. Chronological train / validation / test split (Figure 2)
    train_df, val_df, test_df = chronological_split(df, cfg)
    feature_cols = cfg.numeric_features + cfg.categorical_features

    X_train, y_train = train_df[feature_cols], train_df[cfg.target_col]
    X_val, y_val = val_df[feature_cols], val_df[cfg.target_col]
    X_test, y_test = test_df[feature_cols], test_df[cfg.target_col]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Churn rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    # 3. Train & tune the selected model with stratified cross-validation
    search = train_and_tune(X_train, y_train, cfg, args.model)
    print(f"\nBest CV PR-AUC: {search.best_score_:.3f}")
    print(f"Best params: {search.best_params_}")
    best_model = search.best_estimator_

    # 4. Select an operating threshold on the validation set
    threshold = find_best_threshold(best_model, X_val, y_val)

    # 5. Refit on train + validation, then evaluate once on the held-out test set
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    best_model.fit(X_trainval, y_trainval)

    val_metrics = evaluate(best_model, X_val, y_val, "validation", threshold)
    test_metrics = evaluate(best_model, X_test, y_test, "test", threshold)

    # 6. Persist model + metrics for conceptual deployment (Section 7)
    joblib.dump(best_model, out_dir / f"churn_model_{args.model}.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics, "threshold": threshold}, f, indent=2)

    print(f"\nSaved model and metrics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
