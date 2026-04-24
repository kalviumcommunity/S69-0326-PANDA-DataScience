"""
Classification models for retail inventory — stockout, overstock,
and product-speed prediction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.evaluate import (
    classification_metrics,
    plot_confusion_matrix,
)
from src.utils import (
    EXCLUDE_FROM_FEATURES,
    RANDOM_STATE,
    ensure_dirs,
    get_model_path,
    get_plot_path,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _time_split(
    df: pd.DataFrame, days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by time — last *days* become the test set.

    Args:
        df: Engineered DataFrame with a 'Date' column.
        days: Number of trailing days for test.

    Returns:
        Tuple of (train_df, test_df).
    """
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    return train, test


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns excluding targets.

    Args:
        df: Engineered DataFrame.

    Returns:
        list[str]: Feature column names.
    """
    return [
        c
        for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, np.uint8, int, float]
    ]


# ===================================================================
# 1. Stockout classifier
# ===================================================================

def train_stockout_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost classifier for stockout prediction.

    Handles class imbalance via scale_pos_weight.

    Args:
        df: Engineered DataFrame with 'stockout_flag' column.

    Returns:
        dict: Classification metrics dictionary.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  STOCKOUT CLASSIFICATION")
    print("=" * 60)

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
    print(f"[stockout] Train pos={pos:,}  neg={neg:,}  scale_pos_weight={spw:.2f}")

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
    print(f"\n[stockout] Metrics: {metrics}")
    print("\n" + classification_report(y_test, preds, zero_division=0))

    path = str(get_model_path("stockout_classifier.joblib"))
    joblib.dump(model, path)
    print(f"[stockout] Model saved → {path}")

    return metrics


# ===================================================================
# 2. Overstock classifier
# ===================================================================

def train_overstock_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost classifier for overstock prediction.

    Class is roughly balanced (~50 %), so no scale_pos_weight needed.

    Args:
        df: Engineered DataFrame with 'overstock_flag' column.

    Returns:
        dict: Classification metrics dictionary.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  OVERSTOCK CLASSIFICATION")
    print("=" * 60)

    train, test = _time_split(df)
    feats = _feature_cols(df)

    X_train = train[feats].values
    y_train = train["overstock_flag"].values
    X_test = test[feats].values
    y_test = test["overstock_flag"].values

    print(f"[overstock] Train pos={int((y_train==1).sum()):,}  neg={int((y_train==0).sum()):,}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, preds, probs)
    print(f"\n[overstock] Metrics: {metrics}")
    print("\n" + classification_report(y_test, preds, zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(
        y_test,
        preds,
        labels=["No Overstock", "Overstock"],
        save_path=str(get_plot_path("overstock_confusion.png")),
        title="Overstock Confusion Matrix",
    )

    path = str(get_model_path("overstock_classifier.joblib"))
    joblib.dump(model, path)
    print(f"[overstock] Model saved → {path}")

    return metrics


# ===================================================================
# 3. Product speed classifier (multiclass)
# ===================================================================

def train_product_speed_classifier(df: pd.DataFrame) -> Dict[str, float]:
    """Train an XGBoost multiclass classifier for product speed.

    Labels: 0=slow, 1=medium, 2=fast.

    Args:
        df: Engineered DataFrame with 'product_speed' column.

    Returns:
        dict: {'macro_f1': float} plus per-class report.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  PRODUCT SPEED CLASSIFICATION")
    print("=" * 60)

    train, test = _time_split(df)
    feats = _feature_cols(df)

    X_train = train[feats].values
    y_train = train["product_speed"].values
    X_test = test[feats].values
    y_test = test["product_speed"].values

    print(
        f"[speed] Class distribution (train): "
        f"slow={int((y_train==0).sum()):,}  "
        f"medium={int((y_train==1).sum()):,}  "
        f"fast={int((y_train==2).sum()):,}"
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
    from sklearn.metrics import f1_score

    macro_f1 = round(f1_score(y_test, preds, average="macro", zero_division=0), 4)
    print(f"\n[speed] Macro F1 = {macro_f1}")
    print("\n" + classification_report(y_test, preds, target_names=["slow", "medium", "fast"], zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(
        y_test,
        preds,
        labels=["slow", "medium", "fast"],
        save_path=str(get_plot_path("speed_confusion.png")),
        title="Product Speed Confusion Matrix",
    )

    path = str(get_model_path("product_speed_classifier.joblib"))
    joblib.dump(model, path)
    print(f"[speed] Model saved → {path}")

    return {"macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.feature_engineering import build_features
    from src.utils import get_data_path

    raw = pd.read_csv(get_data_path())
    df = build_features(raw)
    train_stockout_classifier(df)
    train_overstock_classifier(df)
    train_product_speed_classifier(df)
