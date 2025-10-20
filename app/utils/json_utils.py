"""Utilities for converting Python objects into JSON-safe structures."""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _convert_scalar(value: Any) -> Any:
    """Convert a scalar value into a JSON serialisable representation."""

    if value is None:
        return None

    if value is pd.NaT:
        return None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return value.isoformat()

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def sanitize_for_json(data: Any) -> Any:
    """Recursively sanitise *data* so that it can be dumped as JSON."""

    if isinstance(data, pd.Series):
        return sanitize_for_json(data.tolist())

    if isinstance(data, pd.DataFrame):
        return sanitize_for_json(data.to_dict(orient="records"))

    if isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())

    if isinstance(data, dict):
        return {str(key): sanitize_for_json(value) for key, value in data.items()}

    if isinstance(data, (list, tuple, set)):
        return [sanitize_for_json(item) for item in data]

    return _convert_scalar(data)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame preview into JSON serialisable records."""

    safe_df = df.replace([np.inf, -np.inf], np.nan)
    safe_df = safe_df.where(pd.notnull(safe_df), None)
    records = safe_df.to_dict(orient="records")
    return [sanitize_for_json(record) for record in records]

