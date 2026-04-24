"""
Streamlit dashboard for Retail ML Phase 2.

Four tabs:
  1. Demand Forecast — actual vs predicted with filters
  2. Risk Alerts — stockout / overstock flagged products
  3. Product Segments — cluster scatter + membership table
  4. Model Performance — metrics table + SHAP plot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import build_features
from src.utils import get_data_path, get_model_path, get_plot_path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailLens ML Dashboard",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark gradient header */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    .metric-card {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .risk-stockout { background: rgba(239,68,68,0.15); border-left: 4px solid #ef4444; padding: 0.5rem; border-radius: 6px; }
    .risk-overstock { background: rgba(245,158,11,0.15); border-left: 4px solid #f59e0b; padding: 0.5rem; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & engineering features …")
def load_engineered_data() -> pd.DataFrame:
    """Load raw CSV and run feature engineering pipeline.

    Returns:
        pd.DataFrame: Fully engineered DataFrame.
    """
    raw = pd.read_csv(get_data_path())
    return build_features(raw)


@st.cache_resource(show_spinner="Loading models …")
def load_model(name: str):
    """Load a joblib model from the outputs/models directory.

    Args:
        name: Model filename (e.g. 'best_forecast_model.joblib').

    Returns:
        Trained model object, or None if file not found.
    """
    path = get_model_path(name)
    if path.exists():
        return joblib.load(path)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 RetailLens ML")
st.sidebar.markdown("**Phase 2 — ML Dashboard**")

# Load data
df = load_engineered_data()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Demand Forecast", "🚨 Risk Alerts", "🧩 Product Segments", "🏆 Model Performance"]
)

# ===================================================================
# TAB 1 — Demand Forecast
# ===================================================================
with tab1:
    st.header("Demand Forecast")

    col_f1, col_f2, col_f3 = st.sidebar.columns(1) if False else (st.sidebar, st.sidebar, st.sidebar)

    store_ids = sorted(df["Store ID"].unique())
    product_ids = sorted(df["Product ID"].unique())

    sel_store = st.sidebar.selectbox("Store ID", store_ids, key="fc_store")
    sel_product = st.sidebar.selectbox("Product ID", product_ids, key="fc_product")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="fc_dates",
    )

    # Filter
    mask = (df["Store ID"] == sel_store) & (df["Product ID"] == sel_product)
    if len(date_range) == 2:
        mask &= (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
    sub = df[mask].sort_values("Date").copy()

    if sub.empty:
        st.warning("No data for the selected filters.")
    else:
        model = load_model("best_forecast_model.joblib")
        if model is not None:
            from src.utils import EXCLUDE_FROM_FEATURES
            feat_cols = [
                c for c in df.columns
                if c not in EXCLUDE_FROM_FEATURES
                and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, np.uint8]
            ]
            X_sub = sub[feat_cols].values
            sub["predicted"] = model.predict(X_sub)

            # Line chart
            chart_data = sub[["Date", "Units Sold", "predicted"]].set_index("Date")
            st.line_chart(chart_data, use_container_width=True)

            # Next 7-day prediction table
            st.subheader("Last 7 data-points prediction")
            last7 = sub.tail(7)[["Date", "Units Sold", "predicted"]].reset_index(drop=True)
            last7.columns = ["Date", "Actual", "Predicted"]
            st.dataframe(last7, use_container_width=True)
        else:
            st.info("Run the ML pipeline first to generate `best_forecast_model.joblib`.")
            st.line_chart(sub.set_index("Date")["Units Sold"], use_container_width=True)

# ===================================================================
# TAB 2 — Risk Alerts
# ===================================================================
with tab2:
    st.header("🚨 Risk Alerts")

    alerts = df[(df["stockout_flag"] == 1) | (df["overstock_flag"] == 1)].copy()
    alerts["Risk"] = "—"
    alerts.loc[alerts["stockout_flag"] == 1, "Risk"] = "🔴 Stockout"
    alerts.loc[alerts["overstock_flag"] == 1, "Risk"] = "🟠 Overstock"
    # If both
    both = (alerts["stockout_flag"] == 1) & (alerts["overstock_flag"] == 1)
    alerts.loc[both, "Risk"] = "🔴 Stockout"

    alerts["Severity"] = alerts["sell_through_rate"].apply(
        lambda r: "HIGH" if r >= 0.9 else ("MEDIUM" if r >= 0.5 else "LOW")
    )

    display_cols = ["Date", "Store ID", "Product ID", "Risk", "Severity", "Units Sold", "Inventory Level"]
    avail_cols = [c for c in display_cols if c in alerts.columns]
    st.dataframe(
        alerts[avail_cols].sort_values("Date", ascending=False).head(500),
        use_container_width=True,
        height=600,
    )

    col1, col2 = st.columns(2)
    col1.metric("Stockout Events", f"{int(df['stockout_flag'].sum()):,}")
    col2.metric("Overstock Events", f"{int(df['overstock_flag'].sum()):,}")

# ===================================================================
# TAB 3 — Product Segments
# ===================================================================
with tab3:
    st.header("🧩 Product Segments (Clusters)")

    cluster_plot = get_plot_path("product_clusters.png")
    if cluster_plot.exists():
        st.image(str(cluster_plot), caption="Product Clusters (k=3)", use_container_width=True)
    else:
        st.info("Run `cluster_products()` to generate the cluster plot.")

    # Rebuild lightweight cluster table from data
    agg = df.groupby("Product ID").agg(
        avg_units_sold=("Units Sold", "mean"),
        avg_inventory=("Inventory Level", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
    ).reset_index()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feat_c = ["avg_units_sold", "avg_inventory", "avg_sell_through"]
    X_c = StandardScaler().fit_transform(agg[feat_c])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_c)
    agg["Cluster"] = km.labels_

    # Label heuristic
    centroids_df = pd.DataFrame(km.cluster_centers_, columns=feat_c)
    rank = centroids_df["avg_units_sold"].argsort().values
    label_map = {}
    label_map[rank[0]] = "Slow Movers"
    label_map[rank[2]] = "Fast Movers"
    label_map[rank[1]] = "Overstock Risk"
    agg["Segment"] = agg["Cluster"].map(label_map)

    st.dataframe(agg, use_container_width=True, height=500)

# ===================================================================
# TAB 4 — Model Performance
# ===================================================================
with tab4:
    st.header("🏆 Model Performance Summary")

    # Try to display stored metrics — fallback to placeholder
    metrics_data = {
        "Model": [
            "Forecast (best)",
            "Stockout Classifier",
            "Overstock Classifier",
            "Product Speed Classifier",
        ],
        "Primary Metric": ["RMSE", "F1", "F1", "Macro F1"],
        "Value": ["—", "—", "—", "—"],
    }

    # If models exist, recompute quick metrics
    forecast_model = load_model("best_forecast_model.joblib")
    if forecast_model is not None:
        from src.utils import EXCLUDE_FROM_FEATURES
        feat_cols = [
            c for c in df.columns
            if c not in EXCLUDE_FROM_FEATURES
            and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, np.uint8]
        ]
        cutoff = df["Date"].max() - pd.Timedelta(days=30)
        test = df[df["Date"] >= cutoff]
        if not test.empty:
            preds = forecast_model.predict(test[feat_cols].values)
            y_true = test["Units Sold"].values
            rmse = round(np.sqrt(np.mean((y_true - preds) ** 2)), 2)
            mae = round(np.mean(np.abs(y_true - preds)), 2)
            metrics_data["Value"][0] = f"{rmse}"
            st.markdown(f"**Forecast RMSE:** {rmse}  |  **MAE:** {mae}")

    st.table(pd.DataFrame(metrics_data))

    # SHAP plot
    shap_path = get_plot_path("shap_feature_importance.png")
    if shap_path.exists():
        st.subheader("SHAP Feature Importance (XGBoost)")
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("Run the forecasting pipeline to generate the SHAP plot.")
