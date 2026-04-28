"""
Product clustering module for retail inventory.

Aggregates per-product statistics, runs K-Means with elbow analysis,
and produces interpretable cluster labels.
"""

from __future__ import annotations

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_STATE, save_plot, logger


def cluster_products(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-product features, run K-Means, and return labelled summary.

    Steps:
      1. Aggregate by Product ID (mean of continuous, sum of flag columns).
      2. Scale features with StandardScaler.
      3. Run K-Means for k = 2, 3, 4 and select best k via elbow method.
      4. Map cluster IDs to human-readable labels based on centroids.
      5. Save a scatter plot of Units Sold vs Inventory Level coloured by cluster.

    Args:
        df: Engineered DataFrame (output of build_features).

    Returns:
        pd.DataFrame: Per-product summary with cluster assignments.
    """
    logger.info("Starting product clustering...")

    # ------------------------------------------------------------------
    # 1. Aggregate by Product ID
    # ------------------------------------------------------------------
    agg = df.groupby("Product ID").agg(
        avg_units_sold=("Units Sold", "mean"),
        avg_inventory=("Inventory Level", "mean"),
        sell_through_rate=("sell_through_rate", "mean"),
        stock_to_sales_ratio=("stock_to_sales_ratio", "mean"),
        discount_flag=("discount_flag", "mean"),
        price_vs_competitor=("price_vs_competitor", "mean"),
        stockout_count=("stockout_flag", "sum"),
        overstock_count=("overstock_flag", "sum"),
    ).reset_index()
    logger.info("Aggregated data for %d products", len(agg))

    # ------------------------------------------------------------------
    # 2. Scale
    # ------------------------------------------------------------------
    feature_cols = [c for c in agg.columns if c != "Product ID"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[feature_cols])

    # ------------------------------------------------------------------
    # 3. Elbow curve & Auto-selection
    # ------------------------------------------------------------------
    inertias: Dict[int, float] = {}
    k_range = [2, 3, 4]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_

    # Plot elbow curve
    fig_elbow, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()), "o-", linewidth=2, color="#6366f1")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xticks(k_range)
    save_plot(fig_elbow, "elbow_curve.png")

    # Simple auto-selection (pick k where inertia drop slows down most)
    # Since we only have 2, 3, 4, we compare the drop from 2->3 vs 3->4
    if len(k_range) >= 3:
        drop1 = inertias[2] - inertias[3]
        drop2 = inertias[3] - inertias[4]
        best_k = 3 if drop1 > drop2 else 4
    else:
        best_k = 3
    
    logger.info("Auto-selected best k = %d", best_k)

    # ------------------------------------------------------------------
    # 4. Fit best K
    # ------------------------------------------------------------------
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    agg["cluster"] = km_best.fit_predict(X_scaled)

    # ------------------------------------------------------------------
    # 5. Map cluster labels
    # ------------------------------------------------------------------
    centroids = pd.DataFrame(
        scaler.inverse_transform(km_best.cluster_centers_),
        columns=feature_cols,
    )
    centroids["cluster"] = range(best_k)

    # Heuristic labeling based on units sold
    rank_by_sold = centroids.sort_values("avg_units_sold")
    label_map: Dict[int, str] = {}
    label_map[int(rank_by_sold.iloc[0]["cluster"])] = "Slow Movers"
    label_map[int(rank_by_sold.iloc[-1]["cluster"])] = "Fast Movers"
    
    remaining = [c for c in range(best_k) if c not in label_map]
    for c in remaining:
        label_map[c] = "Overstock Risk"

    agg["cluster_label"] = agg["cluster"].map(label_map)
    logger.info("Cluster labels: %s", label_map)

    # ------------------------------------------------------------------
    # 6. Scatter plot
    # ------------------------------------------------------------------
    fig_scatter, ax = plt.subplots(figsize=(8, 6))
    colours = {"Fast Movers": "#22c55e", "Slow Movers": "#ef4444", "Overstock Risk": "#f59e0b"}
    for label, grp in agg.groupby("cluster_label"):
        ax.scatter(
            grp["avg_units_sold"],
            grp["avg_inventory"],
            label=label,
            alpha=0.7,
            s=60,
            color=colours.get(str(label), "#6366f1"),
        )
    ax.set_xlabel("Avg Units Sold")
    ax.set_ylabel("Avg Inventory Level")
    ax.set_title(f"Product Clusters (k={best_k})")
    ax.legend()
    save_plot(fig_scatter, "product_clusters.png")

    return agg


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.feature_engineering import build_features
    from src.utils import get_data_path, setup_logging

    setup_logging()
    raw = pd.read_csv(get_data_path())
    df = build_features(raw)
    summary = cluster_products(df)
    logger.info("Clustering Summary Head:\n%s", summary.head(10))
