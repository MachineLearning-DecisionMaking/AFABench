from pathlib import Path
from typing import ClassVar, final, override

import polars as pl
import torch

from afabench.datasets.miniboone import MiniBooNEDataset
from afabench.datasets.tabular_csv import TabularCSVDataset


@final
class MiniBooNESubsetDataset(TabularCSVDataset):
    """
    MiniBooNE restricted to the 10 features of the L2M paper (arXiv:2510.12624).

    The features are those the authors' loader keeps (`src/data/miniboone.py`
    in reAIM-Lab/Learning-To-Measure), in source order. It reads MiniBooNE's
    source file (ADR 0010, amended) and keeps MiniBooNE's instances, dropping
    the same placeholder and outlier rows.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)
    SHA256: ClassVar[str] = MiniBooNEDataset.SHA256

    # Zero-based positions of source columns "Feature 1", "Feature 2",
    # "Feature 3", "Feature 4", "Feature 12", "Feature 17", "Feature 23",
    # "Feature 25", "Feature 27" and "Feature 32".
    FEATURE_COLUMNS: ClassVar[tuple[int, ...]] = (
        0,
        1,
        2,
        3,
        11,
        16,
        22,
        24,
        26,
        31,
    )

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([len(self.FEATURE_COLUMNS)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(
        self,
        root: str = "data/miniboone/miniboone.csv",
        sha256: str | None = None,
    ):
        self.root = root
        super().__init__(Path(root), sha256)

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        return MiniBooNEDataset.make_prepared_source()

    @override
    def _read_source(self) -> pl.DataFrame:
        frame = MiniBooNEDataset.drop_invalid_rows(super()._read_source())
        return frame.select(
            frame.columns[0],
            *(frame.columns[i + 1] for i in self.FEATURE_COLUMNS),
            frame.columns[-1],
        )
