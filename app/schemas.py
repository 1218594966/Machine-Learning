"""Pydantic schemas shared across API endpoints."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


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


class FeatureEngineeringOptions(BaseModel):
    numeric_scaling: Literal["none", "standardize", "normalize"] = "none"
    categorical_encoding: Literal["one_hot", "none"] = "one_hot"
    numeric_imputation: Literal[
        "none", "mean", "median", "most_frequent", "constant"
    ] = "most_frequent"
    categorical_imputation: Literal["none", "most_frequent", "constant"] = (
        "most_frequent"
    )
    constant_fill_value: Optional[str] = None

    @validator("constant_fill_value", always=True)
    def validate_constant_fill(cls, value: Optional[str], values: Dict[str, str]):
        requires_constant = (
            values.get("numeric_imputation") == "constant"
            or values.get("categorical_imputation") == "constant"
        )
        if requires_constant and (value is None or str(value).strip() == ""):
            raise ValueError("常量填充值不能为空")
        return value


class PreprocessPreviewRequest(BaseModel):
    dataset_name: str
    feature_columns: List[str]
    feature_engineering: FeatureEngineeringOptions = Field(
        default_factory=FeatureEngineeringOptions
    )
    sample_size: int = Field(10, ge=1, le=200)

    @validator("feature_columns")
    def validate_features(cls, value: List[str]):
        if not value:
            raise ValueError("请至少选择一个特征列")
        return value


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., description="Registered dataset name")
    target_column: str = Field(..., description="Column used as label")
    feature_columns: List[str] = Field(..., description="Columns used as features")
    mode: Literal["classification", "regression"]
    algorithm: Literal["random_forest", "xgboost"]
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, description="Random state for reproducibility")
    feature_engineering: FeatureEngineeringOptions = Field(
        default_factory=FeatureEngineeringOptions
    )
    model_params: Optional[Dict[str, object]] = Field(default_factory=dict)

    @validator("feature_columns")
    def check_feature_columns(cls, value: List[str]):
        if not value:
            raise ValueError("请至少选择一个特征列用于训练")
        return value


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
