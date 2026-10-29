import hashlib
from abc import ABC
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self, override
from urllib.request import urlopen

import polars as pl
import torch
from torch import Tensor

from afabench.core.types import Features, Label
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot

# Keep the CSV missing-value vocabulary of the pinned prepared sources.
CSV_NULL_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]
SOURCE_ROW = "__source_row"


def read_source_csv(source: Path, *, separator: str = ",") -> pl.DataFrame:
    return pl.read_csv(
        source,
        separator=separator,
        infer_schema_length=None,
        null_values=CSV_NULL_VALUES,
        row_index_name=SOURCE_ROW,
    )


def prepared_csv_bytes(frame: pl.DataFrame, *, separator: str = ",") -> bytes:
    """Serialize ADR 0014's pinned CSV bytes at the pandas boundary only."""
    import pandas as pd  # noqa: PLC0415

    return (
        pd.DataFrame(frame.to_dict(as_series=False))
        .to_csv(sep=separator, index=False)
        .encode()
    )


def encode_categories(column: pl.Expr) -> pl.Expr:
    # Dense ranks use lexical order, matching the source's label encoding.
    return (column.rank(method="dense") - 1).cast(pl.Float64)


def _z_normalize(features: pl.DataFrame) -> pl.DataFrame:
    """Z-normalize with population statistics, keeping constant columns zero."""
    return features.select(
        (pl.col(name) - pl.col(name).mean())
        / pl.when(pl.col(name).std(ddof=0) == 0)
        .then(1.0)
        .otherwise(pl.col(name).std(ddof=0))
        for name in features.columns
    )


class SourceChecksumError(ValueError):
    """A source file is not the one its dataset is defined on."""


class NoSourcePreparationError(NotImplementedError):
    """The dataset's prepared source cannot be rebuilt from its provider's files."""


def download(url: str) -> bytes:
    """Return the content at `url`."""
    with urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def check_sha256(
    content: bytes, sha256: str, origin: object, hint: str = ""
) -> None:
    """Raise `SourceChecksumError` unless `content`, read from `origin`, has SHA-256 `sha256`."""
    actual = hashlib.sha256(content).hexdigest()
    if actual != sha256:
        msg = f"{origin} has SHA-256 {actual}, expected {sha256}{hint}"
        raise SourceChecksumError(msg)


class TabularCSVDataset(InMemoryAFADataset, ABC):
    """
    A real-world tabular dataset read from one CSV file, in the file's row order (ADR 0004).

    The file is the dataset's prepared source (ADR 0014), checked against
    `SHA256` on every generation. It is committed to the repository unless
    `SOURCE_COMMITTED` is false, when generation prepares it if missing.
    `make_prepared_source` is where a subclass makes it from its
    provider's raw files.

    By default the last column holds the class label and the other columns
    the features; `_read_source` is where a subclass reads its file
    differently. Categorical feature columns are label-encoded, missing
    values are filled with the column mean, and every feature is
    z-normalized over the rows kept. The feature names and each instance's
    source row are saved in dataset.pt beside the tensors.
    """

    # SHA-256 of the prepared source the dataset is defined on
    SHA256: ClassVar[str]
    # False when the source's license does not permit committing it
    SOURCE_COMMITTED: ClassVar[bool] = True
    # The constructor argument holding the source's path
    source_config_key: ClassVar[str] = "root"

    feature_names: list[str]
    # Row of each instance in the source file, counting data rows from 0;
    # differs from the generation index once `_read_source` drops rows
    source_rows: Tensor

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, source: Path, sha256: str | None = None):
        # Used only to build the data; the subclass's constructor arguments
        # they come from are the config
        self.source: Path = source
        self.source_sha256: str = sha256 or self.SHA256
        super().__init__()

    @classmethod
    def prepare_source(cls, source: Path, sha256: str | None = None) -> None:
        """
        Write the prepared source to `source`, made from the provider's raw files.

        It must have SHA-256 `sha256`, by default `SHA256`; a file with
        another is not written, so a changed provider file is not kept.
        """
        content = cls.make_prepared_source()
        check_sha256(
            content, sha256 or cls.SHA256, f"Prepared {cls.__name__} source"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        partial = source.with_name(source.name + ".partial")
        partial.write_bytes(content)
        partial.replace(source)

    @classmethod
    def make_prepared_source(cls) -> bytes:
        """Return the prepared source, made from the provider's raw files, without checking it."""
        msg = (
            f"{cls.__name__} has no source preparation; its source is "
            "committed to the repository"
        )
        raise NoSourcePreparationError(msg)

    def _check_source(self) -> None:
        if not self.source.exists():
            if not self.SOURCE_COMMITTED:
                self.prepare_source(self.source, self.source_sha256)
                return
            msg = (
                f"{type(self).__name__} source not found at {self.source}; "
                "it is committed to the repository, or prepare it with "
                "`uv run python scripts/dataset_generation/prepare_source.py "
                "<dataset key>`"
            )
            raise FileNotFoundError(msg)
        check_sha256(
            self.source.read_bytes(),
            self.source_sha256,
            f"{type(self).__name__} source {self.source}",
            "; restore the committed file"
            if self.SOURCE_COMMITTED
            else "; remove it to prepare it again",
        )

    def _read_source(self) -> pl.DataFrame:
        """
        Return source row, features and label (last), preserving source order.

        Subclasses may filter rows but must retain the explicit SOURCE_ROW
        column, which is assigned before any filtering.
        """
        return read_source_csv(self.source)

    @override
    def _build(self) -> tuple[Features, Label]:
        self._check_source()
        frame = self._read_source()
        labels = frame.get_column(frame.columns[-1])
        features_df = frame.select(frame.columns[1:-1])
        features_df = features_df.with_columns(
            encode_categories(pl.col(name))
            for name, dtype in features_df.schema.items()
            if dtype == pl.String
        ).select(pl.all().cast(pl.Float64, strict=False).fill_nan(None))
        features_df = features_df.select(pl.all().fill_null(pl.all().mean()))
        features_df = _z_normalize(features_df)

        features = torch.tensor(features_df.to_numpy(), dtype=torch.float32)
        if torch.isnan(features).any():
            msg = (
                f"{type(self).__name__} has NaN features after filling "
                f"missing values from {self.source}; a column has no values"
            )
            raise ValueError(msg)
        self.feature_names = features_df.columns
        self.source_rows = torch.tensor(
            frame.get_column(SOURCE_ROW).cast(pl.Int64).to_numpy(),
            dtype=torch.long,
        )
        return features, one_hot(
            torch.tensor(labels.to_numpy(), dtype=torch.long),
            self.label_shape[0],
        )

    def get_source_rows(self) -> Tensor:
        """Return the source file row of each instance, in split order."""
        return self.source_rows

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        subset = super().create_subset(indices)
        subset.source_rows = self.source_rows[list(indices)]
        return subset

    @override
    def _saved_data(self) -> dict[str, object]:
        return super()._saved_data() | {
            "feature_names": self.feature_names,
            "source_rows": self.source_rows,
        }

    @override
    def _restore(self, data: Mapping[str, object], file: Path) -> None:
        super()._restore(data, file)
        feature_names = data.get("feature_names")
        if not isinstance(feature_names, list):
            msg = (
                f"Dataset saved at {file} has no feature names, "
                f"got {feature_names!r}"
            )
            raise TypeError(msg)
        self.feature_names = feature_names
        source_rows = data.get("source_rows")
        if not isinstance(source_rows, Tensor) or len(source_rows) != len(
            self.features
        ):
            msg = (
                f"Dataset saved at {file} has no source row per instance, "
                f"got {source_rows!r}"
            )
            raise TypeError(msg)
        self.source_rows = source_rows
