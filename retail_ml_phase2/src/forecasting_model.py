"""
Demand forecasting models for retail inventory.

Trains Linear Regression, XGBoost, and LightGBM regressors
with a time-based train/test split, evaluates them, and
saves the best model plus diagnostic plots.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.evaluate import (
    plot_actual_vs_predicted,
    plot_shap_summary,
    regression_metrics,
)
from src.utils import (
    EXCLUDE_FROM_FEATURES,
    RANDOM_STATE,
    get_model_path,
    get_plot_path,
    ensure_dirs,
)


def _time_split(
    df: pd.DataFrame, days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train/test using the last *days* as test.

    Args:
        df: Engineered DataFrame with a 'Date' column.
        days: Number of trailing days for the test set.

    Returns:
        Tuple of (train_df, test_df).
    """
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    print(f"[split] Train: {len(train):,} rows  |  Test: {len(test):,} rows  |  Cutoff: {cutoff.date()}")
    return train, test


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature column names, excluding targets and non-numeric cols.

    Args:
        df: Engineered DataFrame.

    Returns:
        list[str]: Feature column names.
    """
    return [
        c
        for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, np.uint8, int, float]
    ]


def train_forecast_model(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Train 3 regression models, evaluate, save best model + plots.

    Args:
        df: Engineered DataFrame (output of build_features).

    Returns:
        dict: Model name → metrics dict.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  DEMAND FORECASTING")
    print("=" * 60)

    train, test = _time_split(df)
    feature_cols = _get_feature_cols(df)
    print(f"[forecast] Using {len(feature_cols)} features")

    X_train = train[feature_cols].values
    y_train = train["Units Sold"].values
    X_test = test[feature_cols].values
    y_test = test["Units Sold"].values

    # ----- Define models -----
    models = {
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n[forecast] Training {name} …")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        results[name] = metrics
        trained_models[name] = (model, preds)
        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MAPE={metrics['MAPE']}%")

    # ----- Comparison table -----
    print("\n" + "-" * 50)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['MAPE']:>7.2f}%")
    print("-" * 50)

    # ----- Best model (lowest RMSE) -----
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model, best_preds = trained_models[best_name]
    print(f"\n[forecast] Best model → {best_name} (RMSE={results[best_name]['RMSE']})")

    model_path = str(get_model_path("best_forecast_model.joblib"))
    joblib.dump(best_model, model_path)
    print(f"[forecast] Model saved → {model_path}")

    # ----- Actual vs Predicted plot -----
    plot_actual_vs_predicted(
        y_test,
        best_preds,
        title=f"Forecast: Actual vs Predicted ({best_name})",
        save_path=str(get_plot_path("forecast_vs_actual.png")),
    )

    # ----- SHAP for XGBoost -----
    xgb_model = trained_models["XGBoost"][0]
    plot_shap_summary(
        xgb_model,
        X_test,
        feature_names=feature_cols,
        save_path=str(get_plot_path("shap_feature_importance.png")),
        max_display=15,
    )

    return results


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
    train_forecast_model(df)
