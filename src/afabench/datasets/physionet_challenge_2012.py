import io
import tarfile
from pathlib import Path
from typing import ClassVar, final, override

import polars as pl
import torch

from afabench.datasets import tabular_csv
from afabench.datasets.tabular_csv import (
    TabularCSVDataset,
    check_sha256,
    prepared_csv_bytes,
)


def _byte_stable_parameter_means(frame: pl.DataFrame) -> pl.DataFrame:
    # ADR 0014 pins decimal CSV bytes: Polars' parallel mean changes the
    # last digit for some stays. Keep only this reduction at the boundary.
    import pandas as pd  # noqa: PLC0415

    means = (
        pd.DataFrame(frame.to_dict(as_series=False))
        .groupby("Parameter")["Value"]
        .mean()
    )
    return pl.from_pandas(means.reset_index())


@final
class PhysionetChallenge2012Dataset(TabularCSVDataset):
    """
    ICU stays of the PhysioNet/CinC Challenge 2012, labelled 1 for in-hospital death.

    The source has one row per stay: each parameter's mean over the stay's
    48 hours, from the records of sets A, B and C, in that order and by
    record file name within each set.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)
    BASE_URL: ClassVar[str] = (
        "https://physionet.org/files/challenge-2012/1.0.0/"
    )
    # Version 1.0.0's files, which make the source
    RAW_SHA256: ClassVar[dict[str, str]] = {
        "set-a.tar.gz": (
            "8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920"
        ),
        "set-b.tar.gz": (
            "b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30"
        ),
        "set-c.tar.gz": (
            "a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1"
        ),
        "Outcomes-a.txt": (
            "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2"
        ),
        "Outcomes-b.txt": (
            "1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5"
        ),
        "Outcomes-c.txt": (
            "0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9"
        ),
    }
    SETS: ClassVar[tuple[str, ...]] = ("a", "b", "c")
    SHA256: ClassVar[str] = (
        "daca2ea78d228b93488f68938ab667a31e683adcbe916b77769633bc297aad91"
    )
    # The Challenge's marker of a missing or unknown value
    MISSING: ClassVar[float] = -1.0

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([41])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(
        self,
        root: str = "data/physionet_challenge_2012/physionet.csv",
        sha256: str | None = None,
    ):
        self.root = root
        super().__init__(Path(root), sha256)

    @classmethod
    def _raw_file(cls, name: str) -> bytes:
        content = tabular_csv.download(cls.BASE_URL + name)
        check_sha256(content, cls.RAW_SHA256[name], cls.BASE_URL + name)
        return content

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        stays: list[pl.DataFrame] = []
        deaths: list[float] = []
        for letter in cls.SETS:
            outcomes = pl.read_csv(
                io.BytesIO(cls._raw_file(f"Outcomes-{letter}.txt"))
            )
            mortality = dict(
                outcomes.select("RecordID", "In-hospital_death").iter_rows()
            )
            records = cls._records(cls._raw_file(f"set-{letter}.tar.gz"))
            for name in sorted(records):
                record_id = int(name.removesuffix(".txt"))
                stays.append(cls._stay_means(records[name]))
                deaths.append(float(mortality[record_id]))
        # Parameters appear in the order the stays first record them.
        frame = pl.concat(stays, how="diagonal").with_columns(
            pl.Series("Outcome", deaths)
        )
        return prepared_csv_bytes(frame)

    @staticmethod
    def _records(archive: bytes) -> dict[str, bytes]:
        """Return each record file of a set's archive by file name, read in one pass."""
        records: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r|gz") as tar:
            for member in tar:
                file = tar.extractfile(member) if member.isfile() else None
                if file is not None and member.name.endswith(".txt"):
                    records[member.name.rsplit("/", 1)[-1]] = file.read()
        return records

    @classmethod
    def _stay_means(cls, record: bytes) -> pl.DataFrame:
        """Return each parameter's mean, the general descriptors included."""
        frame = (
            pl.read_csv(io.BytesIO(record))
            .with_columns(
                pl.col("Value").cast(pl.Float64, strict=False).fill_nan(None)
            )
            .with_columns(
                pl.when(pl.col("Value") == cls.MISSING)
                .then(None)
                .otherwise(pl.col("Value"))
                .alias("Value")
            )
        )
        means = _byte_stable_parameter_means(
            frame.filter(pl.col("Parameter") != "RecordID").select(
                "Parameter", "Value"
            )
        )
        return pl.DataFrame(
            {name: [value] for name, value in means.iter_rows()},
            schema_overrides=dict.fromkeys(means["Parameter"], pl.Float64),
        )
