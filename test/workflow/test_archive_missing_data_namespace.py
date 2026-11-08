import shutil
import subprocess
from pathlib import Path

ARCHIVER = (
    Path(__file__).parents[2]
    / "scripts"
    / "workflow"
    / "archive_missing_data_namespace.sh"
)


def test_archive_namespace_is_exact_and_removal_is_explicit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    archive_dir = tmp_path / "archives"
    first = output / "trained" / "study_v2"
    second = output / "summary" / "val" / "study_v2"
    unrelated = output / "trained" / "other_study"
    for directory in (first, second, unrelated):
        directory.mkdir(parents=True)
    (first / "model.pt").write_bytes(b"model")
    (second / "summary.csv").write_text("accuracy\n0.7\n")
    (unrelated / "model.pt").write_bytes(b"keep")

    command = [
        str(ARCHIVER),
        "--namespace",
        "study_v2",
        "--root",
        str(output),
        "--archive-dir",
        str(archive_dir),
        "--remove-source",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True
    )

    archive = archive_dir / "missing-data-study_v2.tar.zst"
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".manifest.txt").is_file()
    assert "entries=4" in result.stdout
    assert not first.exists()
    assert (second / "summary.csv").is_file()
    assert (unrelated / "model.pt").read_bytes() == b"keep"

    tar = shutil.which("gtar") or shutil.which("tar")
    assert tar is not None
    listing = subprocess.run(
        [tar, "--list", "--zstd", f"--file={archive}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert listing == [
        "./summary/val/study_v2/",
        "./summary/val/study_v2/summary.csv",
        "./trained/study_v2/",
        "./trained/study_v2/model.pt",
    ]


def test_archive_namespace_retains_source_by_default(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = output / "eval" / "val" / "study_v2"
    source.mkdir(parents=True)
    (source / "eval_data.parquet").write_bytes(b"evaluation")

    subprocess.run(
        [
            str(ARCHIVER),
            "--namespace",
            "study_v2",
            "--root",
            str(output),
            "--archive-dir",
            str(tmp_path / "archives"),
        ],
        check=True,
    )

    assert (source / "eval_data.parquet").read_bytes() == b"evaluation"
