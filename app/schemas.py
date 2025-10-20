"""Pydantic schemas shared across API endpoints."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    name: str
    rows: int


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., description="Registered dataset name")
    target_column: str = Field(..., description="Column used as label")
    model_name: str = Field(..., description="Name used to persist the model")
    model_type: str = Field(..., description="Model identifier (random_forest/xgboost)")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, description="Random state for reproducibility")
    model_params: Optional[Dict[str, object]] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    model_name: str
    features: Dict[str, object]


class SHAPRequest(BaseModel):
    model_name: str
    sample_size: int = Field(200, ge=10, le=1000)
