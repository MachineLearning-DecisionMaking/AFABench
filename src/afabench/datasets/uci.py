from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, override

import polars as pl
import torch

from afabench.datasets.tabular_csv import TabularCSVDataset, prepared_csv_bytes


class UCIDataset(TabularCSVDataset, ABC):
    """A tabular dataset from the UCI repository, prepared as the frame `ucimlrepo` returns; its feature shape is the file's."""

    config_keys: ClassVar[tuple[str, ...]] = ("path",)
    source_config_key: ClassVar[str] = "path"

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size(self.features.shape[1:])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(self, path: str, sha256: str | None = None):
        self.path: str = path
        super().__init__(Path(path), sha256)

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        return prepared_csv_bytes(cls._fetch_uci())

    @classmethod
    @abstractmethod
    def _fetch_uci(cls) -> pl.DataFrame:
        """Return the UCI repository's data as the frame `_read_source` reads, with the label last."""
