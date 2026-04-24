"""
Evaluation helper functions for regression, classification,
SHAP analysis, and visualisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for regression predictions.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.

    Returns:
        dict: Dictionary with keys 'MAE', 'RMSE', 'MAPE'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1, and AUC-ROC.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        dict: Dictionary of metric name → value.
    """
    metrics: Dict[str, float] = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["AUC-ROC"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["AUC-ROC"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of actual vs predicted values with identity line.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#4f46e5")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[evaluate] Plot saved → {save_path}")
    plt.close(fig)


def plot_shap_summary(
    model: object,
    X_test: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> None:
    """Generate and optionally save a SHAP beeswarm summary plot.

    Args:
        model: Trained tree-based model (XGBoost, LightGBM, etc.).
        X_test: Test feature matrix (numpy array or DataFrame).
        feature_names: List of feature names matching columns.
        save_path: If provided, save figure to this path.
        max_display: Number of top features to display.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] SHAP plot saved → {save_path}")
    plt.close("all")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class label names for axis ticks.
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[evaluate] Confusion matrix saved → {save_path}")
    plt.close(fig)
