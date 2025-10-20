"""Data management utilities for handling dataset uploads and storage."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class DatasetManager:
    """Manager responsible for persisting uploaded datasets."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}

    def save_dataset(self, file_path: Path, name: Optional[str] = None) -> str:
        """Save a CSV dataset to disk and return its registered name."""
        dataset_name = name or file_path.stem
        target_path = self.base_dir / f"{dataset_name}.csv"
        target_path.write_bytes(file_path.read_bytes())
        # clear cache to guarantee data is reloaded from disk later
        self._cache.pop(dataset_name, None)
        return dataset_name

    def list_datasets(self) -> Dict[str, int]:
        """Return a mapping of dataset names to number of rows."""
        datasets = {}
        for csv_file in self.base_dir.glob("*.csv"):
            name = csv_file.stem
            df = self.load_dataset(name)
            datasets[name] = len(df)
        return datasets

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

    def remove_dataset(self, name: str) -> None:
        """Delete a dataset from disk and cache."""
        file_path = self.base_dir / f"{name}.csv"
        if file_path.exists():
            file_path.unlink()
        self._cache.pop(name, None)
