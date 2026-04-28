"""
Feature engineering pipeline for retail inventory ML Phase 2.

Transforms the raw retail_store_inventory.csv into a fully-engineered
DataFrame ready for modelling (forecasting, classification, clustering).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("retail_ml.features")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps and return a model-ready DataFrame.

    Pipeline steps:
      1. Clean — drop duplicates, impute missing, parse Date
      2. Time features — day_of_week, month, week_of_year, quarter, is_weekend, is_month_end
      3. Lag features — lag_1, lag_7, lag_14, lag_30 (grouped by Store ID + Product ID)
      4. Rolling features — rolling_mean_7, rolling_mean_14, rolling_mean_30
      5. Business features — sell_through_rate, stock_to_sales_ratio, etc.
      6. Target flags — stockout_flag, overstock_flag
      7. Product speed — quantile-based label (slow / medium / fast)
      8. Encoding — label-encode IDs/Region, one-hot encode Category
      9. Drop NaN rows produced by lag/rolling

    Args:
        df: Raw DataFrame loaded from retail_store_inventory.csv.

    Returns:
        pd.DataFrame: Fully-engineered DataFrame with NaN rows removed.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # STEP 1 — Cleaning
    # ------------------------------------------------------------------
    cols_to_drop: List[str] = ["Seasonality", "Weather Condition", "Holiday/Promotion"]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    logger.info("Step 1 — Cleaning done: %s rows remain", f"{len(df):,}")

    # ------------------------------------------------------------------
    # STEP 2 — Temporal features
    # ------------------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    logger.info("Step 2 — Temporal features added")

    # ------------------------------------------------------------------
    # STEP 3 — Lag features (grouped by Store ID + Product ID)
    # ------------------------------------------------------------------
    df.sort_values(["Store ID", "Product ID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    group = df.groupby(["Store ID", "Product ID"])["Units Sold"]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = group.shift(lag)
    logger.info("Step 3 — Lag features: lag_1, lag_7, lag_14, lag_30")

    # ------------------------------------------------------------------
    # STEP 4 — Rolling mean features
    # ------------------------------------------------------------------
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            group.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    logger.info("Step 4 — Rolling features: rolling_mean_7, rolling_mean_14, rolling_mean_30")

    # ------------------------------------------------------------------
    # STEP 5 — Business features
    # ------------------------------------------------------------------
    df["sell_through_rate"] = (df["Units Sold"] / df["Inventory Level"]).clip(0, 1)
    df["stock_to_sales_ratio"] = df["Inventory Level"] / (df["Units Sold"] + 1)
    df["days_of_supply"] = df["Inventory Level"] / (df["rolling_mean_7"] + 1)
    df["price_vs_competitor"] = df["Price"] - df["Competitor Pricing"]
    df["discount_flag"] = (df["Discount"] > 0).astype(int)
    logger.info("Step 5 — Business features computed (5 features)")

    # ------------------------------------------------------------------
    # STEP 6 — Target flags
    # ------------------------------------------------------------------
    # Stockout: units sold >= inventory level (demand met or exceeded supply)
    df["stockout_flag"] = (df["Units Sold"] >= df["Inventory Level"]).astype(int)
    # Overstock: inventory significantly exceeds demand (threshold: 2x units sold)
    df["overstock_flag"] = (df["Inventory Level"] > df["Units Sold"] * 2).astype(int)
    logger.info(
        "Step 6 — Target flags: stockout=%s, overstock=%s",
        f"{df['stockout_flag'].sum():,}",
        f"{df['overstock_flag'].sum():,}",
    )

    # ------------------------------------------------------------------
    # STEP 7 — Product speed label (quantile-based)
    # ------------------------------------------------------------------
    q25 = df["rolling_mean_30"].quantile(0.25)
    q75 = df["rolling_mean_30"].quantile(0.75)
    df["product_speed"] = 1  # medium
    df.loc[df["rolling_mean_30"] > q75, "product_speed"] = 2   # fast
    df.loc[df["rolling_mean_30"] < q25, "product_speed"] = 0   # slow
    logger.info(
        "Step 7 — Product speed: fast=%s, medium=%s, slow=%s",
        int((df["product_speed"] == 2).sum()),
        int((df["product_speed"] == 1).sum()),
        int((df["product_speed"] == 0).sum()),
    )

    # ------------------------------------------------------------------
    # STEP 8 — Encoding
    # ------------------------------------------------------------------
    for col in ["Store ID", "Product ID", "Region"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=["Category"], prefix="cat", dtype=int)
    logger.info("Step 8 — Encoding complete, shape=%s", df.shape)

    # ------------------------------------------------------------------
    # STEP 9 — Drop NaN rows from lags / rolling
    # ------------------------------------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Step 9 — Dropped %s NaN rows, final shape=%s", f"{before - len(df):,}", df.shape)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils import get_data_path, setup_logging

    setup_logging()
    raw = pd.read_csv(get_data_path())
    engineered = build_features(raw)
    logger.info("Final DataFrame: %s rows × %s cols", f"{engineered.shape[0]:,}", engineered.shape[1])
