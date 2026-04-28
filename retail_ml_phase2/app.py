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

import joblib
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
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    .stTable {
        background: rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & engineering features...")
def load_engineered_data() -> pd.DataFrame:
    """Load raw CSV and run feature engineering pipeline."""
    raw = pd.read_csv(get_data_path())
    return build_features(raw)


def load_raw_data() -> pd.DataFrame:
    """Load raw CSV for display in alerts."""
    return pd.read_csv(get_data_path())


@st.cache_resource
def load_model_file(name: str):
    """Load a joblib model."""
    path = get_model_path(name)
    if path.exists():
        return joblib.load(path)
    return None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 RetailLens ML")
st.sidebar.markdown("**Phase 2 — ML Dashboard**")

try:
    df = load_engineered_data()
    st.sidebar.success("Data loaded & engineered")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

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
    
    col_s1, col_s2 = st.columns(2)
    
    store_ids = sorted(df["Store ID"].unique())
    product_ids = sorted(df["Product ID"].unique())
    
    sel_store = st.sidebar.selectbox("Filter by Store ID", store_ids, key="fc_store")
    sel_product = st.sidebar.selectbox("Filter by Product ID", product_ids, key="fc_product")
    
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
        model = load_model_file("best_forecast_model.joblib")
        feature_cols = load_model_file("forecasting_features.joblib")
        
        if model and feature_cols:
            X_sub = sub[feature_cols].values
            sub["Predicted Units Sold"] = model.predict(X_sub)
            
            # Comparison Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sub["Date"], sub["Units Sold"], label="Actual", marker='o', alpha=0.7)
            ax.plot(sub["Date"], sub["Predicted Units Sold"], label="Predicted", linestyle='--', marker='x', alpha=0.7)
            ax.set_title(f"Store {sel_store}, Product {sel_product}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("Forecast vs Actual Table")
            st.dataframe(sub[["Date", "Units Sold", "Predicted Units Sold"]].tail(10), use_container_width=True)
        else:
            st.warning("Forecasting model or feature list missing. Showing historical data only.")
            st.line_chart(sub.set_index("Date")["Units Sold"])

# ===================================================================
# TAB 2 — Risk Alerts
# ===================================================================
with tab2:
    st.header("🚨 Risk Alerts")
    
    alerts = df[(df["stockout_flag"] == 1) | (df["overstock_flag"] == 1)].copy()
    
    if alerts.empty:
        st.success("No stockout or overstock risks detected!")
    else:
        def highlight_risks(row):
            if row.stockout_flag == 1:
                return ['background-color: #ef4444; color: white'] * len(row)
            elif row.overstock_flag == 1:
                return ['background-color: #f59e0b; color: black'] * len(row)
            return [''] * len(row)

        display_df = alerts[["Date", "Store ID", "Product ID", "Units Sold", "Inventory Level", "stockout_flag", "overstock_flag"]].copy()
        display_df = display_df.sort_values("Date", ascending=False)
        
        st.write("Red = Stockout Risk | Orange = Overstock Risk")
        st.dataframe(display_df.style.apply(highlight_risks, axis=1), use_container_width=True, height=500)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Stockouts", int(df["stockout_flag"].sum()))
        col2.metric("Total Overstocks", int(df["overstock_flag"].sum()))

# ===================================================================
# TAB 3 — Product Segments
# ===================================================================
with tab3:
    st.header("🧩 Product Segments")
    
    plot_path = get_plot_path("product_clusters.png")
    if plot_path.exists():
        st.image(str(plot_path), caption="Product Clustering (K-Means)", use_container_width=True)
    else:
        st.info("Cluster plot missing. Run clustering model to generate.")

    # Show segment distribution
    if "cluster_label" in df.columns:
        st.subheader("Segment Distribution")
        dist = df.groupby("Product ID")["cluster_label"].first().value_counts()
        st.bar_chart(dist)
    else:
        # Fallback aggregate table
        st.info("Detailed clustering data not available in main dataframe. Recomputing summary...")
        agg = df.groupby("Product ID").agg({
            "Units Sold": "mean",
            "Inventory Level": "mean",
            "sell_through_rate": "mean"
        }).reset_index()
        st.dataframe(agg.head(20), use_container_width=True)

# ===================================================================
# TAB 4 — Model Performance
# ===================================================================
with tab4:
    st.header("🏆 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forecasting (RMSE/MAE)")
        # In a real app we'd load saved metrics. Here we check plots.
        avp_plot = get_plot_path("forecast_vs_actual.png")
        if avp_plot.exists():
            st.image(str(avp_plot), use_container_width=True)
        else:
            st.warning("Performance plots missing.")
            
    with col2:
        st.subheader("Feature Importance (SHAP)")
        shap_plot = get_plot_path("shap_feature_importance.png")
        if shap_plot.exists():
            st.image(str(shap_plot), use_container_width=True)
        else:
            st.warning("SHAP plot missing.")
            
    st.divider()
    st.subheader("Classification Confusion Matrices")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("Stockout")
        sc_plot = get_plot_path("stockout_confusion.png")
        if sc_plot.exists():
            st.image(str(sc_plot), use_container_width=True)
        else:
            st.info("Missing")
            
    with c2:
        st.write("Overstock")
        oc_plot = get_plot_path("overstock_confusion.png")
        if oc_plot.exists():
            st.image(str(oc_plot), use_container_width=True)
        else:
            st.info("Missing")
            
    with c3:
        st.write("Product Speed")
        pc_plot = get_plot_path("speed_confusion.png")
        if pc_plot.exists():
            st.image(str(pc_plot), use_container_width=True)
        else:
            st.info("Missing")
