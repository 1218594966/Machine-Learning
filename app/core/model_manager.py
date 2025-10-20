"""Model training and persistence layer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import joblib
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from .evaluation import confusion_matrix_report, evaluate_classification
from .preprocess import PreprocessResult

ModelType = Literal["random_forest", "xgboost"]


@dataclass
class ModelArtifact:
    name: str
    model_type: ModelType
    model_path: Path
    preprocessor_path: Path
    metrics: Dict[str, float]
    extra: Dict[str, object]


class ModelManager:
    """High level API for training, saving, and loading models."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_directory(self, name: str) -> Path:
        return self.base_dir / name

    def _model_file(self, name: str) -> Path:
        return self._model_directory(name) / "model.joblib"

    def _preprocessor_file(self, name: str) -> Path:
        return self._model_directory(name) / "preprocessor.joblib"

    def _metadata_file(self, name: str) -> Path:
        return self._model_directory(name) / "metadata.json"

    def train_model(
        self,
        name: str,
        preprocess_result: PreprocessResult,
        model_type: ModelType,
        **model_params,
    ) -> ModelArtifact:
        """Train a classification model and persist it to disk."""
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "xgboost":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
            }
            default_params.update(model_params)
            model = XGBClassifier(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(preprocess_result.x_train, preprocess_result.y_train)
        predictions = model.predict(preprocess_result.x_test)
        metrics = evaluate_classification(preprocess_result.y_test, predictions)
        detailed_report = confusion_matrix_report(preprocess_result.y_test, predictions)

        model_dir = self._model_directory(name)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._model_file(name))
        joblib.dump(preprocess_result.pipeline, self._preprocessor_file(name))
        metadata = {
            "model_type": model_type,
            "metrics": metrics,
            "report": detailed_report,
            "feature_names": preprocess_result.feature_names,
        }
        self._metadata_file(name).write_text(json.dumps(metadata, indent=2))
        self.save_training_data(name, preprocess_result.x_train)

        return ModelArtifact(
            name=name,
            model_type=model_type,
            model_path=self._model_file(name),
            preprocessor_path=self._preprocessor_file(name),
            metrics=metrics,
            extra=detailed_report,
        )

    def list_models(self) -> Dict[str, Dict[str, object]]:
        models = {}
        for model_dir in self.base_dir.iterdir():
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                models[model_dir.name] = metadata
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
            model_type=metadata["model_type"],
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metrics=metadata["metrics"],
            extra={"report": metadata.get("report"), "model": model, "preprocessor": preprocessor},
        )

    def predict(self, name: str, data: np.ndarray) -> np.ndarray:
        artifact = self.load_model(name)
        model = artifact.extra["model"]
        predictions = model.predict(data)
        return predictions

    def transform_inputs(self, name: str, features: Dict[str, object]) -> np.ndarray:
        artifact = self.load_model(name)
        preprocessor = artifact.extra["preprocessor"]
        metadata = json.loads(self._metadata_file(name).read_text())
        feature_names = metadata.get("feature_names", [])
        ordered_values = [[features.get(col) for col in feature_names]]
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
        metadata = json.loads(self._metadata_file(name).read_text())
        feature_names = metadata.get("feature_names", [])

        # load training data to compute shap values
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
        np.save(model_dir / "training.npy", data)
