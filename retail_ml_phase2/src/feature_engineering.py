"""
Feature engineering pipeline for retail inventory ML Phase 2.

Transforms the raw retail_store_inventory.csv into a fully-engineered
DataFrame ready for modelling (forecasting, classification, clustering).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps and return a model-ready DataFrame.

    Performs Phase 1 cleaning, temporal extraction, lag/rolling features,
    business metrics, target flag creation, product speed labelling,
    categorical encoding, and NaN cleanup.

    Args:
        df: Raw DataFrame loaded from retail_store_inventory.csv.

    Returns:
        pd.DataFrame: Fully-engineered DataFrame with NaN rows removed.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # STEP 1 — Re-apply Phase 1 cleaning
    # ------------------------------------------------------------------
    cols_to_drop: List[str] = ["Seasonality", "Weather Condition", "Holiday/Promotion"]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    print(f"[Step 1] Phase 1 cleaning done — {len(df):,} rows remain")

    # ------------------------------------------------------------------
    # STEP 2 — Temporal features
    # ------------------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    print(f"[Step 2] Temporal features added: day_of_week … is_month_end")

    # ------------------------------------------------------------------
    # STEP 3 — Lag features (sort first)
    # ------------------------------------------------------------------
    df.sort_values(["Store ID", "Product ID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    group = df.groupby(["Store ID", "Product ID"])["Units Sold"]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = group.shift(lag)
    print(f"[Step 3] Lag features created: lag_1, lag_7, lag_14, lag_30")

    # ------------------------------------------------------------------
    # STEP 4 — Rolling mean features
    # ------------------------------------------------------------------
    for window in [7, 14, 30]:
        df[f"rolling_{window}"] = (
            group.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    print(f"[Step 4] Rolling features created: rolling_7, rolling_14, rolling_30")

    # ------------------------------------------------------------------
    # STEP 5 — Business metrics
    # ------------------------------------------------------------------
    df["sell_through_rate"] = (df["Units Sold"] / df["Inventory Level"]).clip(0, 1)
    df["stock_to_sales_ratio"] = df["Inventory Level"] / (df["Units Sold"] + 1)
    df["days_of_supply"] = df["Inventory Level"] / (df["rolling_7"] + 1)
    df["demand_forecast_error"] = df["Demand Forecast"] - df["Units Sold"]
    df["price_vs_competitor"] = df["Price"] - df["Competitor Pricing"]
    df["discount_flag"] = (df["Discount"] > 0).astype(int)
    print(f"[Step 5] Business metrics computed (6 features)")

    # ------------------------------------------------------------------
    # STEP 6 — Target flag re-creation
    # ------------------------------------------------------------------
    df["stockout_flag"] = (df["Units Sold"] >= df["Inventory Level"]).astype(int)
    df["overstock_flag"] = (df["Inventory Level"] > df["Units Sold"] * 2).astype(int)
    print(
        f"[Step 6] Target flags — stockout_flag={df['stockout_flag'].sum():,}  "
        f"overstock_flag={df['overstock_flag'].sum():,}"
    )

    # ------------------------------------------------------------------
    # STEP 7 — Product speed label
    # ------------------------------------------------------------------
    q25 = df["rolling_30"].quantile(0.25)
    q75 = df["rolling_30"].quantile(0.75)
    df["product_speed"] = 1  # medium
    df.loc[df["rolling_30"] > q75, "product_speed"] = 2  # fast
    df.loc[df["rolling_30"] < q25, "product_speed"] = 0  # slow
    print(
        f"[Step 7] Product speed labels — "
        f"fast={int((df['product_speed']==2).sum()):,}  "
        f"medium={int((df['product_speed']==1).sum()):,}  "
        f"slow={int((df['product_speed']==0).sum()):,}"
    )

    # ------------------------------------------------------------------
    # STEP 8 — Encoding
    # ------------------------------------------------------------------
    for col in ["Store ID", "Product ID", "Region"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=["Category"], prefix="cat", dtype=int)
    print(f"[Step 8] Encoding complete — shape {df.shape}")

    # ------------------------------------------------------------------
    # STEP 9 — Drop NaN rows from lags / rolling
    # ------------------------------------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[Step 9] Dropped {before - len(df):,} NaN rows — final shape {df.shape}")

    return df


# ---------------------------------------------------------------------------
# CLI entry-point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import get_data_path

    raw = pd.read_csv(get_data_path())
    engineered = build_features(raw)
    print(f"\nFinal DataFrame: {engineered.shape[0]:,} rows × {engineered.shape[1]} cols")
    print(engineered.head())
