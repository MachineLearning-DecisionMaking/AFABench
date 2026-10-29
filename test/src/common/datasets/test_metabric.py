"""METABRIC keeps the L2M paper's 11 genes and six subtypes of a pinned source file."""

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets import MetabricDataset, tabular_csv
from afabench.datasets.metabric import MetabricSourceError
from afabench.datasets.tabular_csv import SourceChecksumError
from afabench.testing.provenance import placeholder_provenance

PAPER_GENES = [
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
]
SUBTYPE = "pam50_+_claudin-low_subtype"
# Row 2 is not classified, so instances 0..6 are rows 0, 1, 3..7
SUBTYPES = [
    "Normal",
    "LumA",
    "NC",
    "claudin-low",
    "Basal",
    "Her2",
    "LumB",
    "LumA",
]


def source_frame() -> pl.DataFrame:
    """Return a small source in the Kaggle file's layout, its genes in reverse order."""
    rng = np.random.default_rng(0)
    n_rows = len(SUBTYPES)
    frame = pl.DataFrame({"patient_id": range(n_rows)})
    frame = frame.with_columns(
        pl.Series("age_at_diagnosis", rng.uniform(30, 90, n_rows))
    )
    frame = frame.with_columns(pl.Series(SUBTYPE, SUBTYPES))
    for gene in [*reversed(PAPER_GENES), "abcb1"]:
        frame = frame.with_columns(pl.Series(gene, rng.normal(size=n_rows)))
    frame = frame.with_columns(pl.lit("0").alias("tp53_mut"))
    return frame


def write_source(path: Path, frame: pl.DataFrame) -> str:
    """Write `frame` as the source; return its SHA-256."""
    frame.write_csv(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metabric(tmp_path: Path, frame: pl.DataFrame) -> MetabricDataset:
    path = tmp_path / "METABRIC_RNA_Mutation.csv"
    return MetabricDataset(root=str(path), sha256=write_source(path, frame))


def test_features_are_the_paper_genes_in_the_paper_order(
    tmp_path: Path,
) -> None:
    frame = source_frame()

    dataset = metabric(tmp_path, frame)

    assert dataset.feature_names == PAPER_GENES
    kept = frame.filter(pl.col(SUBTYPE) != "NC")
    # Normalization keeps each gene's ordering across instances
    for column, gene in enumerate(PAPER_GENES):
        assert dataset.features[:, column].argsort().tolist() == (
            kept[gene].to_numpy().argsort().tolist()
        )


def test_subtypes_are_encoded_in_the_paper_order(tmp_path: Path) -> None:
    dataset = metabric(tmp_path, source_frame())

    # LumA, LumB, Her2, Basal, Normal, claudin-low
    assert dataset.labels.argmax(-1).tolist() == [4, 0, 5, 3, 2, 1, 0]
    assert dataset.label_shape == torch.Size([6])


def test_not_classified_tumours_are_dropped_keeping_source_rows(
    tmp_path: Path,
) -> None:
    dataset = metabric(tmp_path, source_frame())

    assert len(dataset) == 7
    assert dataset.get_source_rows().tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert dataset.get_generation_indices().tolist() == list(range(7))


def test_generation_is_repeatable(tmp_path: Path) -> None:
    first = metabric(tmp_path, source_frame())
    second = MetabricDataset(root=first.root, sha256=first.sha256)

    assert torch.equal(first.features, second.features)
    assert torch.equal(first.labels, second.labels)


def test_bundle_round_trip_keeps_instances_and_config(tmp_path: Path) -> None:
    dataset = metabric(tmp_path, source_frame()).create_subset([5, 0, 2])
    bundle_path = tmp_path / "metabric.bundle"

    save_bundle(
        dataset, bundle_path, metadata={}, provenance=placeholder_provenance()
    )
    loaded = load_bundle(bundle_path).obj

    assert isinstance(loaded, MetabricDataset)
    assert torch.equal(loaded.features, dataset.features)
    assert torch.equal(loaded.labels, dataset.labels)
    assert loaded.get_source_rows().tolist() == [6, 0, 3]
    assert loaded.feature_names == PAPER_GENES
    assert loaded.sha256 == dataset.sha256


def test_source_with_another_checksum_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "METABRIC_RNA_Mutation.csv"
    write_source(path, source_frame())

    with pytest.raises(
        SourceChecksumError, match=r"SHA-256 [0-9a-f]{64}, expected 0{64}"
    ):
        MetabricDataset(root=str(path), sha256="0" * 64)


@pytest.mark.parametrize("column", ["egfr", SUBTYPE])
def test_source_without_a_needed_column_is_rejected(
    tmp_path: Path, column: str
) -> None:
    frame = source_frame().drop(column)

    with pytest.raises(
        MetabricSourceError, match=rf"no columns \['{re.escape(column)}'"
    ):
        metabric(tmp_path, frame)


@pytest.mark.parametrize("column", ["erbb2", SUBTYPE])
def test_missing_value_in_a_kept_tumour_is_rejected(
    tmp_path: Path, column: str
) -> None:
    frame = source_frame()
    frame[4, column] = None

    with pytest.raises(
        MetabricSourceError, match=rf"missing values.*{re.escape(column)}"
    ):
        metabric(tmp_path, frame)


def test_unknown_subtype_is_rejected(tmp_path: Path) -> None:
    frame = source_frame()
    frame[5, SUBTYPE] = "Her2-enriched"

    with pytest.raises(MetabricSourceError, match=r"\['Her2-enriched'\]"):
        metabric(tmp_path, frame)


def serve_archive(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, bytes]
) -> list[str]:
    """Make the download return a zip of `members`; return the URLs requested."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    requested: list[str] = []

    def fake_download(url: str) -> bytes:
        requested.append(url)
        return buffer.getvalue()

    monkeypatch.setattr(tabular_csv, "download", fake_download)
    return requested


def test_source_is_prepared_from_the_kaggle_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = source_frame().write_csv().encode()
    requested = serve_archive(
        monkeypatch, {"METABRIC_RNA_Mutation.csv": content}
    )
    path = tmp_path / "metabric" / "METABRIC_RNA_Mutation.csv"

    MetabricDataset.prepare_source(path, hashlib.sha256(content).hexdigest())

    assert requested == [MetabricDataset.URL]
    assert path.read_bytes() == content


def test_archive_without_the_source_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_archive(monkeypatch, {"other.csv": b"a,b\n"})

    with pytest.raises(MetabricSourceError, match=r"no METABRIC_RNA.*other"):
        MetabricDataset.prepare_source(tmp_path / "METABRIC_RNA_Mutation.csv")


@pytest.mark.optional
def test_real_source_has_the_paper_cohort() -> None:
    """Reads the committed source in data/metabric/."""
    dataset = MetabricDataset()

    assert len(dataset) == 1898
    assert dataset.feature_names == PAPER_GENES
    counts = dataset.labels.sum(0).int().tolist()
    # LumA, LumB, Her2, Basal, Normal, claudin-low
    assert counts == [679, 461, 220, 199, 140, 199]
