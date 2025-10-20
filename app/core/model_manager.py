"""Model management utilities for training, persistence, and inference."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from .evaluation import (
    confusion_matrix_report,
    evaluate_classification,
    evaluate_regression,
)
from .preprocess import PreprocessResult

ModelAlgorithm = Literal["random_forest", "xgboost"]
ModelMode = Literal["classification", "regression"]


@dataclass
class ModelArtifact:
    name: str
    mode: ModelMode
    algorithm: ModelAlgorithm
    model_path: Path
    preprocessor_path: Path
    metrics: Dict[str, float]
    extra: Dict[str, object]


class ModelManager:
    """High level API for training, saving, and loading models."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers for file paths and naming
    # ------------------------------------------------------------------
    def _model_directory(self, name: str) -> Path:
        return self.base_dir / name

    def _model_file(self, name: str) -> Path:
        return self._model_directory(name) / "model.joblib"

    def _preprocessor_file(self, name: str) -> Path:
        return self._model_directory(name) / "preprocessor.joblib"

    def _metadata_file(self, name: str) -> Path:
        return self._model_directory(name) / "metadata.json"

    def _normalise_name(self, name: str) -> str:
        safe = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name]
        normalised = "".join(safe).strip("_") or "model"
        return normalised.lower()

    def generate_model_name(self, dataset_name: str, mode: ModelMode) -> str:
        """Generate a unique model name using the dataset and mode."""

        base = f"{self._normalise_name(dataset_name)}_{mode}"
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        candidate = f"{base}_{timestamp}"
        counter = 1
        while self._model_directory(candidate).exists():
            candidate = f"{base}_{timestamp}_{counter}"
            counter += 1
        return candidate

    # ------------------------------------------------------------------
    # Training and persistence
    # ------------------------------------------------------------------
    def train_model(
        self,
        name: str,
        preprocess_result: PreprocessResult,
        mode: ModelMode,
        algorithm: ModelAlgorithm,
        **model_params,
    ) -> ModelArtifact:
        """Train a model using the supplied preprocessing artefacts."""

        model = self._build_estimator(
            preprocess_result=preprocess_result,
            mode=mode,
            algorithm=algorithm,
            **model_params,
        )

        start_time = time.perf_counter()
        model.fit(preprocess_result.x_train, preprocess_result.y_train)
        training_time = time.perf_counter() - start_time

        predictions = model.predict(preprocess_result.x_test)

        if mode == "classification":
            y_proba: Optional[np.ndarray] = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(preprocess_result.x_test)
                except Exception:  # noqa: BLE001 - fallback when unavailable
                    y_proba = None
            metrics = evaluate_classification(
                preprocess_result.y_test, predictions, y_proba=y_proba
            )
            detailed_report = confusion_matrix_report(
                preprocess_result.y_test, predictions
            )
            class_labels = detailed_report.get("labels")
        else:
            metrics = evaluate_regression(preprocess_result.y_test, predictions)
            detailed_report = {}
            class_labels = None

        model_dir = self._model_directory(name)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._model_file(name))
        joblib.dump(preprocess_result.pipeline, self._preprocessor_file(name))

        metadata: Dict[str, object] = {
            "name": name,
            "mode": mode,
            "algorithm": algorithm,
            "metrics": metrics,
            "report": detailed_report,
            "feature_names": preprocess_result.feature_names,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "training_time_seconds": training_time,
            "stratified_split": preprocess_result.stratify_used,
        }
        if class_labels is not None:
            metadata["class_labels"] = class_labels

        self._metadata_file(name).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        self.save_training_data(name, preprocess_result.x_train)

        return ModelArtifact(
            name=name,
            mode=mode,
            algorithm=algorithm,
            model_path=self._model_file(name),
            preprocessor_path=self._preprocessor_file(name),
            metrics=metrics,
            extra={"report": detailed_report},
        )

    def _build_estimator(
        self,
        *,
        preprocess_result: PreprocessResult,
        mode: ModelMode,
        algorithm: ModelAlgorithm,
        **model_params,
    ):
        if mode == "classification":
            if algorithm == "random_forest":
                default_params = {"n_estimators": 200, "random_state": 42}
                default_params.update(model_params)
                return RandomForestClassifier(**default_params)

            if algorithm == "xgboost":
                default_params = {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "eval_metric": "mlogloss",
                }
                num_classes = int(np.unique(preprocess_result.y_train).shape[0])
                if num_classes <= 2:
                    default_params.setdefault("objective", "binary:logistic")
                else:
                    default_params.setdefault("objective", "multi:softprob")
                    default_params.setdefault("num_class", num_classes)
                default_params.update(model_params)
                return XGBClassifier(**default_params)

        else:  # regression
            if algorithm == "random_forest":
                default_params = {"n_estimators": 200, "random_state": 42}
                default_params.update(model_params)
                return RandomForestRegressor(**default_params)

            if algorithm == "xgboost":
                default_params = {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                }
                default_params.update(model_params)
                return XGBRegressor(**default_params)

        raise ValueError(f"Unsupported algorithm '{algorithm}' for mode '{mode}'")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, object]]:
        models: List[Dict[str, object]] = []
        if not self.base_dir.exists():
            return models
        for model_dir in sorted(self.base_dir.iterdir()):
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                metadata.setdefault("name", model_dir.name)
                models.append(metadata)
        return models

    def load_model(self, name: str) -> ModelArtifact:
        model_path = self._model_file(name)
        preprocessor_path = self._preprocessor_file(name)
        metadata_path = self._metadata_file(name)
        if not all(path.exists() for path in (model_path, preprocessor_path, metadata_path)):
            raise FileNotFoundError(f"Model '{name}' not found")

        metadata = json.loads(metadata_path.read_text())
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return ModelArtifact(
            name=name,
            mode=metadata.get("mode", "classification"),
            algorithm=metadata.get("algorithm", "random_forest"),
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metrics=metadata.get("metrics", {}),
            extra={
                "report": metadata.get("report"),
                "model": model,
                "preprocessor": preprocessor,
                "metadata": metadata,
            },
        )

    def predict(self, name: str, data: np.ndarray) -> np.ndarray:
        artifact = self.load_model(name)
        model = artifact.extra["model"]
        predictions = model.predict(data)
        return predictions

    def predict_with_proba(self, name: str, data: np.ndarray) -> Dict[str, object]:
        """Return predictions and probabilities when available."""

        artifact = self.load_model(name)
        model = artifact.extra["model"]
        metadata = artifact.extra.get("metadata", {})
        mode = metadata.get("mode", "classification")
        predictions = model.predict(data)
        result: Dict[str, object] = {"predictions": predictions}

        if mode == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(data)
                result["probabilities"] = proba
            except Exception:  # noqa: BLE001
                pass

        return result

    def transform_inputs(
        self, name: str, features: Union[Dict[str, object], List[Dict[str, object]]]
    ) -> np.ndarray:
        artifact = self.load_model(name)
        preprocessor = artifact.extra["preprocessor"]
        metadata = artifact.extra.get("metadata", {})
        feature_names: List[str] = metadata.get("feature_names", [])

        rows: List[Dict[str, object]]
        if isinstance(features, dict):
            rows = [features]
        else:
            rows = list(features)

        missing_columns: List[str] = [
            column
            for column in feature_names
            if any(column not in row or row[column] is None for row in rows)
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"缺少必要的特征列: {missing}")

        ordered_values: List[List[object]] = [
            [row.get(col) for col in feature_names] for row in rows
        ]
        transformed = preprocessor.transform(ordered_values)
        return transformed

    def generate_shap_summary(
        self,
        name: str,
        sample_size: int = 200,
    ) -> Dict[str, str]:
        """Generate SHAP summary plot encoded as base64."""

        import base64
        import io

        artifact = self.load_model(name)
        model = artifact.extra["model"]
        preprocessor = artifact.extra["preprocessor"]
        metadata = artifact.extra.get("metadata", {})
        feature_names = metadata.get("feature_names", [])

        training_data_path = self._model_directory(name) / "training.npy"
        if not training_data_path.exists():
            raise FileNotFoundError(
                "Training data not found. Retrain the model to regenerate SHAP artifacts."
            )
        transformed_data = np.load(training_data_path)
        if transformed_data.shape[0] > sample_size:
            transformed_data = transformed_data[:sample_size]

        explainer = shap.Explainer(model, transformed_data)
        shap_values = explainer(transformed_data)
        if feature_names:
            shap_values.feature_names = feature_names

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return {"image_base64": encoded, "feature_names": feature_names}

    def save_training_data(self, name: str, data: np.ndarray) -> None:
        model_dir = self._model_directory(name)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "training.npy", np.asarray(data))

    def delete_model(self, name: str) -> None:
        """Remove a persisted model and its metadata."""

        model_dir = self._model_directory(name)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def model_overview(self, name: str) -> Dict[str, object]:
        """Return metadata and evaluation artefacts for a given model."""

        metadata_path = self._metadata_file(name)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model '{name}' not found")
        metadata = json.loads(metadata_path.read_text())
        metadata.setdefault("name", name)
        return metadata

    def reset_storage(self) -> None:
        """Remove all trained models and metadata."""

        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
