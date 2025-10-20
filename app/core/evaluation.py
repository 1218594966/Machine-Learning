"""Model evaluation utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return common evaluation metrics for classification models."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {"confusion_matrix": matrix.tolist(), "classification_report": report}
