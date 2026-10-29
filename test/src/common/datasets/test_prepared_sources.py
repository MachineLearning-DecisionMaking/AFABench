"""Tabular datasets read a prepared source pinned by SHA-256, which source preparation rebuilds from the provider's raw files (ADR 0014)."""

import hashlib
import io
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from afabench.datasets import (
    ACTG175Dataset,
    CKDDataset,
    DiabetesDataset,
    MiniBooNEDataset,
    MiniBooNESubsetDataset,
    PhysionetChallenge2012Dataset,
    tabular_csv,
)
from afabench.datasets.tabular_csv import (
    NoSourcePreparationError,
    SourceChecksumError,
)
from scripts.dataset_generation.prepare_source import REPO_ROOT, source_of


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def uci_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "target": [0, 1, 0, 1, 1, 0]}
    )


def serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    """Make the download return `files[name]` for a URL ending in `name`."""
    monkeypatch.setattr(
        tabular_csv, "download", lambda url: files[url.rsplit("/", 1)[-1]]
    )


def test_missing_committed_source_names_its_preparation(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"prepare_source\.py"):
        CKDDataset(path=str(tmp_path / "ckd.csv"))


def test_changed_committed_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ckd.csv"
    uci_frame().write_csv(path)

    with pytest.raises(
        SourceChecksumError, match=r"expected 0{64}; restore the committed"
    ):
        CKDDataset(path=str(path), sha256="0" * 64)


def test_source_kept_out_of_the_repository_is_prepared_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ACTG175Dataset, "_fetch_uci", classmethod(lambda _: uci_frame())
    )
    content = uci_frame().write_csv().encode()
    path = tmp_path / "actg" / "actg.csv"

    dataset = ACTG175Dataset(path=str(path), sha256=sha256(content))

    assert path.read_bytes() == content
    assert len(dataset) == 6


def test_prepared_source_with_another_checksum_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CKDDataset, "_fetch_uci", classmethod(lambda _: uci_frame())
    )

    with pytest.raises(SourceChecksumError, match=r"expected 0{64}"):
        CKDDataset.prepare_source(tmp_path / "ckd.csv", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_source_without_preparation_says_so(tmp_path: Path) -> None:
    with pytest.raises(NoSourcePreparationError, match="committed"):
        DiabetesDataset.prepare_source(tmp_path / "diabetes.csv")


def miniboone_archive() -> bytes:
    """Return a UCI-style archive of 2 signal and 3 background events, each feature its event's number."""
    lines = [" 2 3"]
    lines += [
        "  " + "  ".join(f"{event:.6E}" for _ in range(50))
        for event in range(5)
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("MiniBooNE_PID.txt", "\n".join(lines) + "\n")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "dataset_class", [MiniBooNEDataset, MiniBooNESubsetDataset]
)
def test_miniboone_source_lists_signal_then_background_events(
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[MiniBooNEDataset | MiniBooNESubsetDataset],
) -> None:
    serve(
        monkeypatch,
        {MiniBooNEDataset.URL.rsplit("/", 1)[-1]: miniboone_archive()},
    )

    frame = pl.read_csv(io.BytesIO(dataset_class.make_prepared_source()))

    assert list(frame.columns) == [
        *(f"Feature {i}" for i in range(1, 51)),
        "Outcome",
    ]
    assert frame["Feature 50"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frame["Outcome"].to_list() == [1, 1, 0, 0, 0]


def record(record_id: int, rows: Sequence[tuple[str, str, float]]) -> bytes:
    lines = ["Time,Parameter,Value", f"00:00,RecordID,{record_id}"]
    lines += [f"{time},{name},{value}" for time, name, value in rows]
    return ("\n".join(lines) + "\n").encode()


def physionet_files() -> dict[str, bytes]:
    """Return raw Challenge files of two stays per set, set b's listed out of order in its archive."""
    stays = {
        "a": {
            "132539": [("00:00", "Age", 54), ("01:00", "HR", 70)],
            "132540": [("00:00", "Age", 76), ("02:00", "HR", 80)],
        },
        "b": {
            "142999": [("00:00", "Age", 40), ("03:00", "HR", -1)],
            "142671": [("00:00", "Age", 60), ("05:00", "HR", 90)],
        },
        "c": {
            "152871": [("00:00", "Age", 30), ("01:00", "HR", 60)],
            "152872": [
                ("00:00", "Age", 20),
                ("01:00", "HR", 61),
                ("02:00", "HR", 63),
            ],
        },
    }
    deaths = {"132539": 0, "132540": 1, "142999": 0}
    deaths |= {"142671": 1, "152871": 1, "152872": 0}
    files: dict[str, bytes] = {}
    for letter, records in stays.items():
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for record_id, rows in records.items():
                content = record(int(record_id), rows)
                info = tarfile.TarInfo(f"set-{letter}/{record_id}.txt")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        files[f"set-{letter}.tar.gz"] = buffer.getvalue()
        outcome_lines = ["RecordID,SAPS-I,In-hospital_death"]
        outcome_lines += [f"{rid},1,{deaths[rid]}" for rid in records]
        files[f"Outcomes-{letter}.txt"] = "\n".join(outcome_lines).encode()
    return files


def test_physionet_source_has_each_stays_parameter_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = physionet_files()
    serve(monkeypatch, files)
    monkeypatch.setattr(
        PhysionetChallenge2012Dataset,
        "RAW_SHA256",
        {name: sha256(content) for name, content in files.items()},
    )

    frame = pl.read_csv(
        io.BytesIO(PhysionetChallenge2012Dataset.make_prepared_source())
    )

    assert list(frame.columns) == ["Age", "HR", "Outcome"]
    # Sets a, b and c in order, by record file name within each set
    assert frame["Age"].to_list() == [54, 76, 60, 40, 30, 20]
    # -1 marks a missing value, so stay 142999 has no HR
    assert (
        frame["HR"].is_null().to_list() == [False] * 3 + [True] + [False] * 2
    )
    assert frame["HR"].drop_nulls().to_list() == [70, 80, 90, 60, 62]
    assert frame["Outcome"].to_list() == [0, 1, 1, 0, 1, 0]


def test_physionet_raw_file_with_another_checksum_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = physionet_files()
    serve(monkeypatch, files)
    expected = {name: sha256(content) for name, content in files.items()}
    expected["set-b.tar.gz"] = "0" * 64
    monkeypatch.setattr(PhysionetChallenge2012Dataset, "RAW_SHA256", expected)

    with pytest.raises(SourceChecksumError, match=r"set-b\.tar\.gz"):
        PhysionetChallenge2012Dataset.prepare_source(tmp_path / "p.csv")


@pytest.mark.parametrize(
    ("dataset_key", "dataset_class", "folder"),
    [
        ("actg", ACTG175Dataset, "actg"),
        ("ckd", CKDDataset, "ckd"),
        ("diabetes", DiabetesDataset, "diabetes"),
        ("miniboone", MiniBooNEDataset, "miniboone"),
        # Reads its parent's source (ADR 0010)
        ("miniboone_subset", MiniBooNESubsetDataset, "miniboone"),
        (
            "physionet_challenge_2012",
            PhysionetChallenge2012Dataset,
            "physionet_challenge_2012",
        ),
    ],
)
def test_preparation_writes_where_generation_reads(
    dataset_key: str, dataset_class: type, folder: str
) -> None:
    found_class, path, _ = source_of(dataset_key)

    assert found_class is dataset_class
    assert path.parent == REPO_ROOT / "data" / folder


def test_metabric_preparation_uses_the_configs_checksum() -> None:
    _, _, configured = source_of("metabric")

    assert configured is not None
    assert len(configured) == 64


def test_dataset_without_a_tabular_source_has_no_preparation() -> None:
    with pytest.raises(TypeError, match="no tabular source"):
        source_of("cube")


def test_miniboone_prepared_csv_bytes_are_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(
        monkeypatch,
        {MiniBooNEDataset.URL.rsplit("/", 1)[-1]: miniboone_archive()},
    )
    header = ",".join([*(f"Feature {i}" for i in range(1, 51)), "Outcome"])
    rows = [
        ",".join([*([f"{event}.0"] * 50), str(int(event < 2))])
        for event in range(5)
    ]

    assert (
        MiniBooNEDataset.make_prepared_source()
        == ("\n".join([header, *rows]) + "\n").encode()
    )


def test_physionet_prepared_csv_bytes_are_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = physionet_files()
    serve(monkeypatch, files)
    monkeypatch.setattr(
        PhysionetChallenge2012Dataset,
        "RAW_SHA256",
        {name: sha256(content) for name, content in files.items()},
    )

    assert PhysionetChallenge2012Dataset.make_prepared_source() == (
        b"Age,HR,Outcome\n54.0,70.0,0.0\n76.0,80.0,1.0\n"
        b"60.0,90.0,1.0\n40.0,,0.0\n30.0,60.0,1.0\n20.0,62.0,0.0\n"
    )


def test_physionet_parameter_mean_preserves_pinned_decimal_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("00:00", "HeartRate", 90.0)] * 1000
    for index, value in zip((30, 400, 900), (33.4, 31.3, 29.1), strict=True):
        rows[index] = ("00:00", "HCT", value)
    content = record(135365, rows)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("set-a/135365.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    files = {
        "set-a.tar.gz": buffer.getvalue(),
        "Outcomes-a.txt": b"RecordID,In-hospital_death\n135365,0\n",
    }
    serve(monkeypatch, files)
    monkeypatch.setattr(PhysionetChallenge2012Dataset, "SETS", ("a",))
    monkeypatch.setattr(
        PhysionetChallenge2012Dataset,
        "RAW_SHA256",
        {name: sha256(raw) for name, raw in files.items()},
    )

    assert PhysionetChallenge2012Dataset.make_prepared_source() == (
        b"HCT,HeartRate,Outcome\n31.266666666666666,90.0,0.0\n"
    )
