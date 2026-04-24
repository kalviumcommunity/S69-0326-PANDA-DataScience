"""
Utility helpers for the retail ML Phase 2 pipeline.

Provides path resolution, directory creation, and shared constants
used across all pipeline modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the retail_ml_phase2 project root.

    Returns:
        Path: Project root directory.
    """
    return Path(__file__).resolve().parent.parent


def ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    root = get_project_root()
    (root / "outputs" / "models").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    print(f"[utils] Output directories verified at {root / 'outputs'}")


def get_data_path(filename: str = "retail_store_inventory.csv") -> Path:
    """Return the absolute path to a file in the data/ directory.

    Args:
        filename: Name of the data file.

    Returns:
        Path: Absolute path to the data file.
    """
    return get_project_root() / "data" / filename


def get_model_path(filename: str) -> Path:
    """Return the absolute path for saving a model in outputs/models/.

    Args:
        filename: Model file name (e.g. 'best_forecast_model.joblib').

    Returns:
        Path: Absolute path to the model file.
    """
    return get_project_root() / "outputs" / "models" / filename


def get_plot_path(filename: str) -> Path:
    """Return the absolute path for saving a plot in outputs/plots/.

    Args:
        filename: Plot file name (e.g. 'forecast_vs_actual.png').

    Returns:
        Path: Absolute path to the plot file.
    """
    return get_project_root() / "outputs" / "plots" / filename


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

COLS_TO_DROP: List[str] = ["Seasonality", "Weather Condition", "Holiday/Promotion"]

TARGET_FORECAST: str = "Units Sold"

EXCLUDE_FROM_FEATURES: List[str] = [
    "Date",
    "Units Sold",
    "stockout_flag",
    "overstock_flag",
    "product_speed",
    "Demand Forecast",
]

RANDOM_STATE: int = 42
