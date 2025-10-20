"""Utilities for preprocessing data prior to model training."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PreprocessResult:
    """Container returned after preprocessing a dataset."""

    def __init__(
        self,
        pipeline: Pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
        feature_names: List[str],
    ) -> None:
        self.pipeline = pipeline
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names


def build_preprocess_pipeline(df: pd.DataFrame, target_column: str) -> Tuple[Pipeline, List[str]]:
    """Create a preprocessing pipeline capable of handling numeric/categorical features."""
    features = df.drop(columns=[target_column])
    categorical_cols = features.select_dtypes(include=["object", "category", "bool"]).columns
    numeric_cols = features.select_dtypes(include=["number"]).columns

    transformers = []
    if len(categorical_cols) > 0:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_cols),
            )
        )
    if len(numeric_cols) > 0:
        transformers.append(("numeric", StandardScaler(), list(numeric_cols)))

    column_transformer = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([("preprocessor", column_transformer)])
    feature_names = list(features.columns)
    return pipeline, feature_names


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessResult:
    """Split and transform the dataset, returning a PreprocessResult."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    pipeline, feature_names = build_preprocess_pipeline(df, target_column)
    features = df.drop(columns=[target_column])
    target = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    pipeline.fit(x_train)
    x_train_transformed = pipeline.transform(x_train)
    x_test_transformed = pipeline.transform(x_test)

    return PreprocessResult(
        pipeline=pipeline,
        x_train=x_train_transformed,
        x_test=x_test_transformed,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
