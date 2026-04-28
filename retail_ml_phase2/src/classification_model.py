"""
Classification models for retail inventory — stockout, overstock,
and product-speed prediction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

from src.evaluate import (
    classification_metrics,
    plot_confusion_matrix,
)
from src.utils import (
    EXCLUDE_FROM_FEATURES,
    RANDOM_STATE,
    save_model,
    save_plot,
    logger,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _time_split(
    df: pd.DataFrame, days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by time — last *days* become the test set."""
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    return train, test


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns excluding targets."""
    return [
        c
        for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and df[c].dtype in [
            np.float64, np.int64, np.int32, np.float32, np.uint8, int, float,
            "float64", "int64", "int32", "float32", "uint8"
        ]
    ]


# ===================================================================
# 1. Stockout classifier
# ===================================================================

def train_stockout_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost classifier for stockout prediction."""
    logger.info("Starting stockout classification training...")
    
    train, test = _time_split(df)
    feats = _feature_cols(df)

    X_train = train[feats].values
    y_train = train["stockout_flag"].values
    X_test = test[feats].values
    y_test = test["stockout_flag"].values

    # Handle imbalance
    neg = int((y_train == 0).sum())
    pos = max(int((y_train == 1).sum()), 1)
    spw = neg / pos
    logger.info("Stockout Split: Pos=%d, Neg=%d, scale_pos_weight=%.2f", pos, neg, spw)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, preds, probs)
    logger.info("Stockout Metrics: %s", metrics)

    # Save model
    save_model(model, "stockout_classifier.joblib")

    # Confusion matrix
    fig_cm = plot_confusion_matrix(
        y_test,
        preds,
        labels=["No Stockout", "Stockout"],
        title="Stockout Confusion Matrix",
    )
    save_plot(fig_cm, "stockout_confusion.png")

    return metrics


# ===================================================================
# 2. Overstock classifier
# ===================================================================

def train_overstock_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost classifier for overstock prediction."""
    logger.info("Starting overstock classification training...")

    train, test = _time_split(df)
    feats = _feature_cols(df)

    X_train = train[feats].values
    y_train = train["overstock_flag"].values
    X_test = test[feats].values
    y_test = test["overstock_flag"].values

    # Handle imbalance
    neg = int((y_train == 0).sum())
    pos = max(int((y_train == 1).sum()), 1)
    spw = neg / pos
    logger.info("Overstock Split: Pos=%d, Neg=%d, scale_pos_weight=%.2f", pos, neg, spw)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, preds, probs)
    logger.info("Overstock Metrics: %s", metrics)

    # Save model
    save_model(model, "overstock_classifier.joblib")

    # Confusion matrix
    fig_cm = plot_confusion_matrix(
        y_test,
        preds,
        labels=["No Overstock", "Overstock"],
        title="Overstock Confusion Matrix",
    )
    save_plot(fig_cm, "overstock_confusion.png")

    return metrics


# ===================================================================
# 3. Product speed classifier (multiclass)
# ===================================================================

def train_product_speed_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost multiclass classifier for product speed."""
    logger.info("Starting product speed classification training...")

    train, test = _time_split(df)
    feats = _feature_cols(df)

    X_train = train[feats].values
    y_train = train["product_speed"].values
    X_test = test[feats].values
    y_test = test["product_speed"].values

    logger.info(
        "Product Speed distribution (train): slow=%d, medium=%d, fast=%d",
        int((y_train == 0).sum()),
        int((y_train == 1).sum()),
        int((y_train == 2).sum()),
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    
    macro_f = round(f1_score(y_test, preds, average="macro", zero_division=0), 4)
    logger.info("Product Speed Macro F1 = %.4f", macro_f)

    # Save model
    save_model(model, "product_speed_classifier.joblib")

    # Confusion matrix
    fig_cm = plot_confusion_matrix(
        y_test,
        preds,
        labels=["slow", "medium", "fast"],
        title="Product Speed Confusion Matrix",
    )
    save_plot(fig_cm, "speed_confusion.png")

    return {"macro_f1": macro_f}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.feature_engineering import build_features
    from src.utils import get_data_path, setup_logging

    setup_logging()
    raw = pd.read_csv(get_data_path())
    df = build_features(raw)
    train_stockout_classifier(df)
    train_overstock_classifier(df)
    train_product_speed_classifier(df)
