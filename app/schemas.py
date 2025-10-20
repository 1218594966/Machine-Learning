"""Pydantic schemas shared across API endpoints."""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, validator


ModelTypeLiteral = Literal["random_forest", "xgboost"]


class DatasetSummary(BaseModel):
    name: str
    rows: int
    columns: int
    target_suggestions: List[str] = Field(default_factory=list)


class ColumnDetail(BaseModel):
    name: str
    dtype: str
    missing: int
    unique_values: int


class DatasetProfile(BaseModel):
    name: str
    rows: int
    columns: int
    column_details: List[ColumnDetail]
    preview: List[Dict[str, object]]


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., description="Registered dataset name")
    target_column: str = Field(..., description="Column used as label")
    model_name: str = Field(..., description="Name used to persist the model")
    model_type: ModelTypeLiteral = Field(..., description="Model identifier")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, description="Random state for reproducibility")
    model_params: Optional[Dict[str, object]] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    model_name: str
    features: Union[Dict[str, object], List[Dict[str, object]]]

    @validator("features")
    def validate_features(cls, value: Union[Dict[str, object], List[Dict[str, object]]]):
        if isinstance(value, list) and not value:
            raise ValueError("特征列表不能为空")
        if isinstance(value, dict) and not value:
            raise ValueError("特征字段不能为空")
        return value


class SHAPRequest(BaseModel):
    model_name: str
    sample_size: int = Field(200, ge=10, le=1000)
