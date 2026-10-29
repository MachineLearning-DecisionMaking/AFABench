"""Snapshot round trip through the real orchestration's output root."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from afabench.release.manifest import (
    ClassifierVariant,
    ExecutionMode,
    read_release_manifest,
)
from afabench.release.snapshot import restore_snapshot, save_snapshot
from scripts.release.snapshot import app
from test.workflow.submission_harness import WorkflowHarness


@pytest.mark.pipeline
def test_restored_snapshot_leaves_nothing_for_a_dry_run_to_schedule(
    tmp_path: Path,
) -> None:
    completed = WorkflowHarness(tmp_path / "completed")
    run = completed.run("--executor", "local", target="all_eval_methods")
    assert run.returncode == 0, run.stdout + run.stderr

    snapshot_dir = tmp_path / "snapshot"
    save_snapshot(tmp_path / "completed/output/smoke", snapshot_dir)

    fresh = WorkflowHarness(tmp_path / "fresh")
    restore_snapshot(snapshot_dir, tmp_path / "fresh/output/smoke")

    result = fresh.run(
        "--dry-run", "--executor", "local", target="all_eval_methods"
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Nothing to be done" in output


# Writes a bundle with a provenance record, as `save_bundle` does, so the
# release manifest can index it. Shared by the stubs below.
RECORDED_BUNDLE = """\
import json
from pathlib import Path

from afabench.core.bundle_system.bundle import compute_content_hash
from afabench.core.provenance import DatasetIdentity, capture_provenance

CUBE = DatasetIdentity(dataset_key="cube", dataset_realization_index=0)


def record(stage, **fields):
    return capture_provenance(
        stage=stage,
        resolved_config={},
        seed=0,
        smoke_test=True,
        device="cpu",
        dataset_identity=CUBE,
        **fields,
    )


def write_bundle(path, provenance):
    path = Path(path)
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "data/path.txt").write_text(str(path))
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": "1.1.0",
                "class_name": "Stub",
                "class_version": None,
                "metadata": {},
                "provenance": provenance.to_json_dict(),
                "content_hash": compute_content_hash(path / "data"),
            }
        )
    )
"""

# Stands in for training: writes a method bundle with its record.
FAKE_TRAINER = """\
import sys

from recorded_bundle import record, write_bundle

args = dict(arg.split("=", 1) for arg in sys.argv[1:])
write_bundle(
    args["save_path"], record("training", method_name=args["method_name"])
)
"""

# Stands in for the evaluator: writes a one-episode raw table with external
# predictions only, its identity columns and its record.
FAKE_EVALUATOR = """\
import sys
from pathlib import Path

import polars as pl

from afabench.core.bundle_system.bundle import bundle_provenance
from afabench.evaluation.provenance import save_evaluation_table
from afabench.evaluation.schemas import identity_column
from recorded_bundle import record

args = dict(arg.split("=", 1) for arg in sys.argv[1:])
method = bundle_provenance(Path(args["method_bundle_path"])).method_name
out = Path(args["save_path"])
out.parent.mkdir(parents=True, exist_ok=True)
frame = pl.DataFrame(
    {
        "episode_id": [0],
        "generation_index": [0],
        "split_index": [0],
        "step": [0],
        "action_performed": [0],
        "builtin_predicted_probabilities": pl.Series([None], dtype=pl.List(pl.Float64)),
        "external_predicted_probabilities": [[0.25, 0.75]],
        "true_class": [1],
        "accumulated_cost": [0.0],
        "forced_stop": [False],
    }
)
for name, value in {
    "afa_method": method,
    "dataset": "cube",
    "dataset_realization_index": 0,
    "eval_split": "test",
    "initializer": "cold",
    "train_seed": 0,
    "train_hard_budget": 1.0,
    "train_soft_budget_param": None,
    "eval_seed": 0,
    "eval_hard_budget": 1.0,
    "eval_soft_budget_param": None,
}.items():
    frame = frame.with_columns(identity_column(name, value, len(frame)))
save_evaluation_table(
    frame, out, provenance=record("evaluation", method_name=method)
)
"""

# Records the harness's prerequisite bundles, which it creates empty;
# datasets first, so the classifier stays newer than its inputs.
RECORD_PREREQUISITES = """\
from pathlib import Path

from recorded_bundle import record, write_bundle

for stage, folder in [
    ("dataset_generation", "datasets"),
    ("classifier_training", "trained_classifiers"),
]:
    for path in sorted(Path("output/smoke", folder).rglob("*.bundle")):
        write_bundle(path, record(stage))
"""


@pytest.mark.pipeline
def test_smoke_snapshot_round_trips_its_smoke_manifest(
    tmp_path: Path,
) -> None:
    completed = WorkflowHarness(tmp_path / "completed")
    scripts = tmp_path / "completed/scripts"
    for path, stub in [
        ("recorded_bundle.py", RECORDED_BUNDLE),
        ("train_method/alpha.py", FAKE_TRAINER),
        ("train_method/beta.py", FAKE_TRAINER),
        ("eval/eval_afa_method.py", FAKE_EVALUATOR),
    ]:
        (scripts / path).write_text(stub)
    # The stubs import the shared module from their own folder's parent.
    for folder in ["train_method", "eval"]:
        (scripts / folder / "recorded_bundle.py").symlink_to(
            scripts / "recorded_bundle.py"
        )
    prerequisites = subprocess.run(
        [sys.executable, "-c", RECORD_PREREQUISITES],
        cwd=tmp_path / "completed",
        env={**os.environ, "PYTHONPATH": str(scripts)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert prerequisites.returncode == 0, prerequisites.stderr
    run = completed.run("--executor", "local", target="all_eval_methods")
    assert run.returncode == 0, run.stdout + run.stderr
    # This round trip needs the dataset bundles too, not a package that
    # omits them for lack of a redistribution review.
    review = tmp_path / "completed/conf/release/dataset_redistribution.yaml"
    review.parent.mkdir(parents=True)
    review.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "cube": {
                        "status": "permitted",
                        "bundle_policy": "include",
                        "license": "MIT",
                        "source": "generated",
                        "reviewed_by": "test maintainer",
                        "notes": None,
                    }
                }
            }
        )
    )
    snapshot_dir = tmp_path / "snapshot"
    runner = CliRunner()

    save = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(tmp_path / "completed/output/smoke"),
            "--checkout",
            str(tmp_path / "completed"),
            "--release-id",
            "smoke-check",
        ],
    )
    assert save.exit_code == 0, save.output
    fresh = WorkflowHarness(tmp_path / "fresh")
    restore = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(tmp_path / "fresh/output/smoke"),
        ],
    )
    assert restore.exit_code == 0, restore.output

    manifest = read_release_manifest(
        tmp_path / "fresh/output/smoke/release_manifest.json"
    )
    assert manifest.execution_mode is ExecutionMode.SMOKE
    assert manifest.evaluations
    assert all(evaluation.raw_path for evaluation in manifest.evaluations)
    assert manifest.coverage.methods == ["alpha", "beta"]
    assert manifest.coverage.classifier_variants == [
        ClassifierVariant.EXTERNAL
    ]
    result = fresh.run(
        "--dry-run", "--executor", "local", target="all_eval_methods"
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Nothing to be done" in output
