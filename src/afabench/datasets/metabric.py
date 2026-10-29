"""
METABRIC breast tumours, classified into their molecular subtype from 11 genes.

The features and label are those of the L2M paper (arXiv:2510.12624v2,
Appendix A.7.1); `docs/reference/datasets/metabric.md` describes the source,
its license and how AFABench departs from the paper.
"""

import io
import zipfile
from pathlib import Path
from typing import ClassVar, final, override

import polars as pl
import torch

from afabench.datasets import tabular_csv
from afabench.datasets.tabular_csv import TabularCSVDataset, read_source_csv


class MetabricSourceError(ValueError):
    """The METABRIC source file does not hold the columns and subtypes the dataset is defined on."""


@final
class MetabricDataset(TabularCSVDataset):
    """
    METABRIC restricted to the 11 genes of the L2M paper, labelled with the PAM50 + claudin-low subtype.

    The source is Kaggle's copy of cBioPortal's METABRIC study, unchanged;
    `sha256` is part of the config, so a bundle names the file it was
    generated from. The 6 tumours of subtype NC (not classified) are
    dropped; `get_source_rows` maps instances back to the file.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root", "sha256")

    URL: ClassVar[str] = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "raghadalharbi/breast-cancer-gene-expression-profiles-metabric"
    )
    SOURCE_FILE: ClassVar[str] = "METABRIC_RNA_Mutation.csv"
    # Version 1 of the Kaggle dataset, the only one (2020-05-26)
    SHA256: ClassVar[str] = (
        "d4802594827667a71a75b5ae5c3cad1eb4346222052fcc792361c64abc7e43bc"
    )
    # mRNA z-scores, in the order of both the paper and the source file
    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "ccnb1",
        "cdk1",
        "e2f2",
        "e2f7",
        "stat5b",
        "notch1",
        "rbpj",
        "bcl2",
        "egfr",
        "erbb2",
        "erbb3",
    )
    LABEL_COLUMN: ClassVar[str] = "pam50_+_claudin-low_subtype"
    # Luminal A, Luminal B, HER2-enriched, basal-like, normal-like and
    # claudin-low, in the paper's order
    CLASSES: ClassVar[tuple[str, ...]] = (
        "LumA",
        "LumB",
        "Her2",
        "Basal",
        "Normal",
        "claudin-low",
    )
    NOT_CLASSIFIED: ClassVar[str] = "NC"

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([len(self.FEATURE_COLUMNS)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([len(self.CLASSES)])

    def __init__(
        self,
        root: str = "data/metabric/METABRIC_RNA_Mutation.csv",
        sha256: str = SHA256,
    ):
        self.root = root
        self.sha256 = sha256
        super().__init__(Path(root), sha256)

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        archive = zipfile.ZipFile(io.BytesIO(tabular_csv.download(cls.URL)))
        if cls.SOURCE_FILE not in archive.namelist():
            msg = (
                f"The archive downloaded from {cls.URL} has no "
                f"{cls.SOURCE_FILE}, only {archive.namelist()}"
            )
            raise MetabricSourceError(msg)
        return archive.read(cls.SOURCE_FILE)

    @override
    def _read_source(self) -> pl.DataFrame:
        frame = read_source_csv(self.source)
        columns = [*self.FEATURE_COLUMNS, self.LABEL_COLUMN]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            msg = f"METABRIC source {self.source} has no columns {missing}"
            raise MetabricSourceError(msg)
        frame = frame.filter(
            (pl.col(self.LABEL_COLUMN) != self.NOT_CLASSIFIED)
            | pl.col(self.LABEL_COLUMN).is_null()
        )
        with_missing = [
            c for c in columns if frame.get_column(c).null_count() > 0
        ]
        if with_missing:
            msg = (
                f"METABRIC source {self.source} has missing values in "
                f"{with_missing}"
            )
            raise MetabricSourceError(msg)
        unknown = sorted(
            set(frame.get_column(self.LABEL_COLUMN).to_list())
            - set(self.CLASSES),
            key=str,
        )
        if unknown:
            msg = (
                f"METABRIC source {self.source} has subtypes {unknown}, "
                f"expected {[*self.CLASSES, self.NOT_CLASSIFIED]}"
            )
            raise MetabricSourceError(msg)
        return frame.select(tabular_csv.SOURCE_ROW, *columns).with_columns(
            pl.col(self.LABEL_COLUMN).replace_strict(
                {name: index for index, name in enumerate(self.CLASSES)}
            )
        )
