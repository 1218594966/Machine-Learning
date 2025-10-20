"""Model evaluation utilities."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return common evaluation metrics for classification models."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            # Raised when probabilities cannot be computed (e.g. binary labels missing)
            pass

    return metrics


def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    labels = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "labels": labels.tolist(),
    }
