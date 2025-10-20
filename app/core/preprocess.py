"""Utilities for preprocessing data prior to model training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from ..utils.json_utils import dataframe_to_records, sanitize_for_json


@dataclass
class FeatureEngineeringConfig:
    """Configuration describing how to transform numeric and categorical data."""

    numeric_scaling: str = "none"
    categorical_encoding: str = "one_hot"
    numeric_imputation: str = "most_frequent"
    categorical_imputation: str = "most_frequent"
    constant_fill_value: Optional[str] = None

    @classmethod
    def from_options(
        cls, options: Union["FeatureEngineeringConfig", dict]
    ) -> "FeatureEngineeringConfig":
        if isinstance(options, cls):
            return options
        if hasattr(options, "dict"):
            data = options.dict()  # type: ignore[arg-type]
        elif isinstance(options, dict):
            data = options
        else:
            raise TypeError("Unsupported feature engineering options payload")
        return cls(**data)

    def numeric_constant(self) -> Optional[float]:
        if self.numeric_imputation != "constant":
            return None
        if self.constant_fill_value is None:
            raise ValueError("请填写数值缺失值的常量填充值")
        try:
            return float(self.constant_fill_value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("数值常量填充值必须为数字") from exc

    def categorical_constant(self) -> Optional[str]:
        if self.categorical_imputation != "constant":
            return None
        if self.constant_fill_value is None:
            raise ValueError("请填写类别缺失值的常量填充值")
        return str(self.constant_fill_value)


@dataclass
class PreprocessResult:
    """Container returned after preprocessing a dataset."""

    pipeline: Pipeline
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    stratify_used: bool


def _clean_feature_names(raw_names: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for name in raw_names:
        if "__" in name:
            cleaned.append(name.split("__", 1)[1])
        else:
            cleaned.append(name)
    return cleaned


def _get_feature_names(pipeline: Pipeline, fallback: Iterable[str]) -> List[str]:
    candidates: Optional[Sequence[str]] = None
    try:
        candidates = pipeline.get_feature_names_out()
    except AttributeError:
        try:
            preprocessor = pipeline.named_steps.get("preprocessor")
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                candidates = preprocessor.get_feature_names_out()
        except Exception:  # pragma: no cover - defensive
            candidates = None

    if candidates is None:
        return list(fallback)
    return _clean_feature_names(list(candidates))


def build_preprocess_pipeline(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    options: FeatureEngineeringConfig,
) -> Tuple[Pipeline, List[str]]:
    """Create a preprocessing pipeline based on the selected columns and options."""

    if not feature_columns:
        raise ValueError("请至少选择一个特征列")

    features = df.loc[:, feature_columns]
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_steps: List[Tuple[str, object]] = []
        if options.numeric_imputation != "none":
            strategy = options.numeric_imputation if options.numeric_imputation != "constant" else "constant"
            numeric_steps.append(
                (
                    "imputer",
                    SimpleImputer(strategy=strategy, fill_value=options.numeric_constant()),
                )
            )
        if options.numeric_scaling == "standardize":
            numeric_steps.append(("scaler", StandardScaler()))
        elif options.numeric_scaling == "normalize":
            numeric_steps.append(("scaler", MinMaxScaler()))

        transformer = (
            "numeric",
            Pipeline(steps=numeric_steps) if numeric_steps else "passthrough",
            numeric_cols,
        )
        transformers.append(transformer)

    if categorical_cols:
        categorical_steps: List[Tuple[str, object]] = []
        if options.categorical_imputation != "none":
            strategy = (
                options.categorical_imputation
                if options.categorical_imputation != "constant"
                else "constant"
            )
            categorical_steps.append(
                (
                    "imputer",
                    SimpleImputer(
                        strategy=strategy,
                        fill_value=options.categorical_constant() or "缺失",
                    ),
                )
            )

        if options.categorical_encoding == "one_hot":
            categorical_steps.append(
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                )
            )
        else:
            categorical_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                )
            )

        transformers.append(("categorical", Pipeline(steps=categorical_steps), categorical_cols))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    pipeline = Pipeline([("preprocessor", column_transformer)])
    feature_names = list(feature_columns)
    return pipeline, feature_names


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
    options: FeatureEngineeringConfig,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    mode: str = "classification",
) -> PreprocessResult:
    """Split and transform the dataset, returning a PreprocessResult."""

    missing_columns = [
        column
        for column in list(feature_columns) + [target_column]
        if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"以下列在数据集中不存在: {missing}")

    pipeline, original_feature_names = build_preprocess_pipeline(
        df, feature_columns, options
    )
    features = df.loc[:, feature_columns]
    target = df[target_column]

    stratify_target: Optional[pd.Series] = None
    if mode == "classification":
        unique_count = target.nunique(dropna=True)
        if unique_count > 1 and unique_count <= max(50, len(target) // 2):
            stratify_target = target

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
        stratified = stratify_target is not None
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        stratified = False

    pipeline.fit(x_train)
    x_train_transformed = np.asarray(pipeline.transform(x_train))
    x_test_transformed = np.asarray(pipeline.transform(x_test))
    feature_names = _get_feature_names(pipeline, original_feature_names)

    return PreprocessResult(
        pipeline=pipeline,
        x_train=x_train_transformed,
        x_test=x_test_transformed,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        stratify_used=stratified,
    )


def preview_transformation(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    options: FeatureEngineeringConfig,
    sample_size: int = 10,
) -> Dict[str, object]:
    """Return a preview of the transformed dataset for display purposes."""

    if not feature_columns:
        raise ValueError("请至少选择一个特征列")

    pipeline, original_feature_names = build_preprocess_pipeline(
        df, feature_columns, options
    )
    sample = df.loc[:, feature_columns].head(sample_size)
    pipeline.fit(sample)
    transformed = pipeline.transform(sample)
    feature_names = _get_feature_names(pipeline, original_feature_names)

    preview_df = pd.DataFrame(transformed, columns=feature_names).round(6)
    preview_records = dataframe_to_records(preview_df)

    return sanitize_for_json({"feature_names": feature_names, "rows": preview_records})
