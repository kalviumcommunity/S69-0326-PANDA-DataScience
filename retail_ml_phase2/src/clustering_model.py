"""
Product clustering module for retail inventory.

Aggregates per-product statistics, runs K-Means with elbow analysis,
and produces interpretable cluster labels.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_STATE, ensure_dirs, get_plot_path


def cluster_products(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-product features, run K-Means, and return labelled summary.

    Steps:
      1. Aggregate by Product ID (mean of continuous, sum of flag columns).
      2. Scale features with StandardScaler.
      3. Run K-Means for k = 2, 3, 4 and plot elbow curve.
      4. Use k = 3 as the best number of clusters.
      5. Map cluster IDs to human-readable labels based on centroids.
      6. Save a scatter plot of Units Sold vs Inventory Level coloured by cluster.

    Args:
        df: Engineered DataFrame (output of build_features).

    Returns:
        pd.DataFrame: Per-product summary with cluster assignments.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  PRODUCT CLUSTERING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Aggregate by Product ID
    # ------------------------------------------------------------------
    agg = df.groupby("Product ID").agg(
        units_sold_mean=("Units Sold", "mean"),
        inventory_level_mean=("Inventory Level", "mean"),
        sell_through_rate_mean=("sell_through_rate", "mean"),
        stock_to_sales_ratio_mean=("stock_to_sales_ratio", "mean"),
        discount_flag_mean=("discount_flag", "mean"),
        price_vs_competitor_mean=("price_vs_competitor", "mean"),
        stockout_flag_sum=("stockout_flag", "sum"),
        overstock_flag_sum=("overstock_flag", "sum"),
    ).reset_index()
    print(f"[cluster] Aggregated {len(agg)} products  |  {agg.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Scale
    # ------------------------------------------------------------------
    feature_cols = [c for c in agg.columns if c != "Product ID"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[feature_cols])

    # ------------------------------------------------------------------
    # 3. Elbow curve
    # ------------------------------------------------------------------
    inertias: Dict[int, float] = {}
    k_range = [2, 3, 4]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()), "o-", linewidth=2, color="#6366f1")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xticks(k_range)
    fig.tight_layout()
    elbow_path = str(get_plot_path("elbow_curve.png"))
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    print(f"[cluster] Elbow plot saved → {elbow_path}")

    # ------------------------------------------------------------------
    # 4. Fit best K = 3
    # ------------------------------------------------------------------
    best_k = 3
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    agg["cluster"] = km_best.fit_predict(X_scaled)
    print(f"[cluster] K-Means fit with k={best_k}")

    # ------------------------------------------------------------------
    # 5. Map cluster labels
    # ------------------------------------------------------------------
    centroids = pd.DataFrame(
        scaler.inverse_transform(km_best.cluster_centers_),
        columns=feature_cols,
    )
    centroids["cluster"] = range(best_k)

    # Heuristic: highest units_sold_mean → fast mover,
    #            highest overstock_flag_sum → overstock risk,
    #            lowest units_sold_mean → slow mover
    rank_by_sold = centroids.sort_values("units_sold_mean")
    label_map: Dict[int, str] = {}
    label_map[int(rank_by_sold.iloc[0]["cluster"])] = "Slow Movers"
    label_map[int(rank_by_sold.iloc[-1]["cluster"])] = "Fast Movers"
    remaining = [c for c in range(best_k) if c not in label_map]
    for c in remaining:
        label_map[c] = "Overstock Risk"

    agg["cluster_label"] = agg["cluster"].map(label_map)
    print(f"[cluster] Label mapping: {label_map}")
    print(agg["cluster_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 6. Scatter plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = {"Fast Movers": "#22c55e", "Slow Movers": "#ef4444", "Overstock Risk": "#f59e0b"}
    for label, grp in agg.groupby("cluster_label"):
        ax.scatter(
            grp["units_sold_mean"],
            grp["inventory_level_mean"],
            label=label,
            alpha=0.7,
            s=60,
            color=colours.get(str(label), "#6366f1"),
        )
    ax.set_xlabel("Avg Units Sold")
    ax.set_ylabel("Avg Inventory Level")
    ax.set_title("Product Clusters (k=3)")
    ax.legend()
    fig.tight_layout()
    scatter_path = str(get_plot_path("product_clusters.png"))
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[cluster] Scatter plot saved → {scatter_path}")

    return agg


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
    summary = cluster_products(df)
    print(summary.head(10))
