import io
import zipfile
from pathlib import Path
from typing import ClassVar, final, override

import polars as pl
import torch

from afabench.datasets import tabular_csv
from afabench.datasets.tabular_csv import TabularCSVDataset, prepared_csv_bytes


@final
class MiniBooNEDataset(TabularCSVDataset):
    """
    MiniBooNE particle identification events, labelled 1 for signal (electron neutrinos) and 0 for background.

    The source is UCI's `MiniBooNE_PID.txt` as a CSV with the label last,
    in the provider's order: its first line counts the signal and the
    background events, which follow in that order.

    Rows holding the undocumented -999 placeholder (468 in the UCI file, all
    -999 in every feature) and the four rows whose Feature 20 is off by
    orders of magnitude (-17256.7 to 1.6e7, against [-5.2, 7.5] elsewhere)
    are dropped (issue #129); `get_source_rows` maps instances back to the
    file.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)
    URL: ClassVar[str] = (
        "https://archive.ics.uci.edu/static/public/199/"
        "miniboone+particle+identification.zip"
    )
    RAW_FILE: ClassVar[str] = "MiniBooNE_PID.txt"
    # Also the file the GDFS repository holds as datasets/miniboone.csv
    SHA256: ClassVar[str] = (
        "e8f5a61b08ac12304086a0326603dabb3400d57c691d4630c37e69492effb0ea"
    )

    PLACEHOLDER: ClassVar[float] = -999.0
    # Feature 20 is the 20th column; read by position like the base reads
    # the label
    OUTLIER_COLUMN: ClassVar[int] = 19
    OUTLIER_BOUND: ClassVar[float] = 1000.0

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([50])

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
        archive = zipfile.ZipFile(io.BytesIO(tabular_csv.download(cls.URL)))
        with archive.open(cls.RAW_FILE) as raw:
            text = io.TextIOWrapper(raw)
            n_signal, n_background = map(int, text.readline().split())
            # The provider uses variable-width whitespace, not CSV delimiters.
            content = "\n".join(
                ",".join(line.split()) for line in text if line.strip()
            )
        frame = pl.read_csv(
            io.StringIO(content), has_header=False, infer_schema_length=None
        )
        frame.columns = [f"Feature {i + 1}" for i in range(frame.width)]
        frame = frame.with_columns(
            pl.Series("Outcome", [1] * n_signal + [0] * n_background)
        )
        return prepared_csv_bytes(frame)

    @override
    def _read_source(self) -> pl.DataFrame:
        return self.drop_invalid_rows(super()._read_source())

    @classmethod
    def drop_invalid_rows(cls, frame: pl.DataFrame) -> pl.DataFrame:
        """Drop the placeholder and outlier rows, retaining source rows."""
        features = frame.columns[1:-1]
        return frame.filter(
            ~pl.any_horizontal(
                (pl.col(features) == cls.PLACEHOLDER).fill_null(False)
            )
            & (pl.col(features[cls.OUTLIER_COLUMN]).abs() < cls.OUTLIER_BOUND)
        )
