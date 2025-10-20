"""Data management utilities for handling dataset uploads and storage."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..utils.json_utils import dataframe_to_records, sanitize_for_json


class DatasetManager:
    """Manager responsible for persisting uploaded datasets."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}

    def save_dataset(self, file_path: Path, name: Optional[str] = None) -> str:
        """Persist an uploaded dataset to disk in CSV format."""

        df = self._read_dataframe(file_path)
        base_name = self._normalise_name(name or file_path.stem)
        dataset_name = base_name
        target_path = self.base_dir / f"{dataset_name}.csv"

        counter = 1
        while target_path.exists():
            dataset_name = f"{base_name}_{counter}"
            target_path = self.base_dir / f"{dataset_name}.csv"
            counter += 1

        df.to_csv(target_path, index=False)
        self._cache[dataset_name] = df
        return dataset_name

    def _read_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load an uploaded file into a DataFrame supporting CSV/Excel."""

        suffix = file_path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(file_path)
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(file_path)
        raise ValueError("仅支持 CSV 或 Excel 格式的文件")

    @staticmethod
    def _normalise_name(name: str) -> str:
        """Sanitise dataset names for safe storage on disk."""

        safe = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip()]
        normalised = "".join(safe).strip("_") or "dataset"
        return normalised.lower()

    def list_datasets(self) -> List[Dict[str, object]]:
        """Return a list of dataset summaries with basic statistics."""

        summaries: List[Dict[str, object]] = []
        for csv_file in sorted(self.base_dir.glob("*.csv")):
            name = csv_file.stem
            df = self.load_dataset(name)
            summaries.append(
                {
                    "name": name,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "target_suggestions": self._infer_target_columns(df),
                }
            )
        return summaries

    def load_dataset(self, name: str, refresh: bool = False) -> pd.DataFrame:
        """Load a dataset into memory."""
        if not refresh and name in self._cache:
            return self._cache[name]

        file_path = self.base_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' does not exist")

        df = pd.read_csv(file_path)
        self._cache[name] = df
        return df

    def reset_storage(self) -> None:
        """Remove all persisted datasets and clear cache."""

        for csv_file in self.base_dir.glob("*.csv"):
            csv_file.unlink()
        self._cache.clear()

    def dataset_profile(self, name: str, preview_rows: int = 5) -> Dict[str, object]:
        """Generate an at-a-glance profile for a dataset."""

        df = self.load_dataset(name)
        profile = {
            "name": name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_details": [
                {
                    "name": column,
                    "dtype": str(dtype),
                    "missing": int(df[column].isna().sum()),
                    "unique_values": int(df[column].nunique(dropna=True)),
                }
                for column, dtype in df.dtypes.items()
            ],
            "preview": dataframe_to_records(df.head(preview_rows)),
        }
        return sanitize_for_json(profile)

    def _infer_target_columns(self, df: pd.DataFrame, max_unique: int = 20) -> List[str]:
        """Suggest potential classification targets based on cardinality."""

        candidates: Iterable[str] = (
            col
            for col in df.columns
            if df[col].dtype == "object"
            or df[col].dtype.name.startswith("category")
            or df[col].nunique(dropna=True) <= max_unique
        )
        return sorted(set(candidates))

    def remove_dataset(self, name: str) -> None:
        """Delete a dataset from disk and cache."""
        file_path = self.base_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' does not exist")
        file_path.unlink()
        self._cache.pop(name, None)
