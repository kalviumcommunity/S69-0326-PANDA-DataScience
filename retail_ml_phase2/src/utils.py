"""
Utility helpers for the retail ML Phase 2 pipeline.

Provides path resolution, directory creation, model/plot persistence,
logging setup, and shared constants used across all pipeline modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Module-level logger
logger = logging.getLogger("retail_ml")


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
    logger.info("Output directories verified at %s", root / "outputs")


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
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: object, filename: str) -> Path:
    """Serialize a trained model to outputs/models/ via joblib.

    Args:
        model: Trained scikit-learn / XGBoost / LightGBM model.
        filename: Target filename (e.g. 'best_forecast_model.joblib').

    Returns:
        Path: Absolute path where the model was saved.
    """
    ensure_dirs()
    path = get_model_path(filename)
    joblib.dump(model, str(path))
    logger.info("Model saved → %s", path)
    return path


def load_model(filename: str) -> Optional[object]:
    """Load a joblib model from outputs/models/.

    Args:
        filename: Model filename.

    Returns:
        Trained model object, or None if the file does not exist.
    """
    path = get_model_path(filename)
    if path.exists():
        model = joblib.load(str(path))
        logger.info("Model loaded ← %s", path)
        return model
    logger.warning("Model not found: %s", path)
    return None


def save_plot(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to outputs/plots/ and close it.

    Args:
        fig: Matplotlib Figure object.
        filename: Target filename (e.g. 'elbow_curve.png').
        dpi: Resolution in dots per inch.

    Returns:
        Path: Absolute path where the plot was saved.
    """
    ensure_dirs()
    path = get_plot_path(filename)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    return path


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
