"""
Repository adopter and results-only user workflows on a real smoke run.

A published workspace runs the real pipeline on CUBE with `random_dummy`
and `gdfs` (see `test_native_bundle_restore`), saves it as a release and
lays it out on a fake release host; `publish` refuses smoke-test outputs
(ADR 0012). A separate fork adds `gdfs_adopter`, a configured variant of
GDFS, downloads the published methods' chains and runs its own pipeline
profile through the launcher. That profile lists the published methods in
`methods` next to the variant and adds `cube_without_noise`, a dataset the
release lacks, so the run reuses the release as a cache: it produces only
the variant's work and everything on the new dataset. The comparison is a
workflow demonstration on smoke runs, not a scientific result, and its
plots say so.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from click.testing import Result
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from afabench.release.manifest import ExecutionMode, read_release_manifest
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport
from test.workflow.test_native_bundle_restore import (
    PROFILE_CONFIGFILES,
    REPO_ROOT,
    SMOKE_SELECTION,
    snakemake,
    workspace,
)
from workflow.run import snakemake_arguments

RELEASE_ID = "smoke-adopter-journey"
REPO_ID = "afabench-test/releases"
TAG = "initializer-cold"
CAPTION = "Workflow demonstration from smoke runs, not scientific results"
PUBLISHED_METHODS = ["random_dummy", "gdfs"]
NEW_DATASET = "cube_without_noise"
# The adopter's own configuration: the published run's settings, the
# variant and the new dataset, and its own method set.
ADOPTER_CONFIG: dict[str, Any] = {
    "datasets": ["cube", NEW_DATASET],
    "dataset_realization_indices": [0],
    "methods": [*PUBLISHED_METHODS, "gdfs_adopter"],
    "eval_hard_budgets": {"cube": [2], NEW_DATASET: [2]},
    "soft_budget_params": {
        "random_dummy": {"cube": [[0.3, None]], NEW_DATASET: [[0.3, None]]},
        "gdfs": {"cube": [], NEW_DATASET: []},
        "gdfs_adopter": {"default": []},
    },
    "method_options": {
        "gdfs_adopter": {
            "pretrained_model_name": "gdfs",
            "train_script_name": "gdfs",
            "method_specific_params": ["lr=0.0005"],
            "use_max_hard_budget_when_training_soft_budget": True,
            "eval_batch_size": {"default": 32},
        }
    },
    "method_sets": {
        "adopter_comparison": [*PUBLISHED_METHODS, "gdfs_adopter"]
    },
}
# Without the shipped method sets: the adopter names their own.
ADOPTER_PROFILE = {
    "snakefile": "workflow/snakefiles/orchestration/pipeline.smk",
    "configfile": [
        *(
            path.relative_to(REPO_ROOT).as_posix()
            for path in PROFILE_CONFIGFILES
            if "method_sets" not in path.parts
        ),
        "workflow/conf/adopter.yaml",
    ],
}
LAUNCHER_ARGUMENTS = [
    "adopter",
    "--jobs",
    "4",
    "smoke_test=true",
    "use_wandb=false",
    "execution_file=workflow/profiles/execution/cpu.yaml",
]
EVAL_PERF = f"eval_split-test/{TAG}/eval_perf"
COMPARISON = "method_set-adopter_comparison"
# The jobs of each pipeline stage the adopter's run must schedule, by
# method and dataset; nothing else of those stages.
PRODUCED_JOBS = {
    ("dataset_generation", None, NEW_DATASET),
    ("train_classifier", None, NEW_DATASET),
    ("pretrain_model", None, NEW_DATASET),
    *(
        (rule, method, dataset)
        for rule in ["train_method", "eval_method", "transform_eval_data"]
        for method, dataset in [
            ("gdfs_adopter", "cube"),
            ("gdfs_adopter", NEW_DATASET),
            ("gdfs", NEW_DATASET),
            ("random_dummy", NEW_DATASET),
        ]
    ),
}


def fork(root: Path) -> Path:
    """
    Make a checkout-shaped fork that owns its config, as a fork does.

    The display configuration names the variant, the comparison's plots
    carry the workflow-demonstration caption, and the adopter's pipeline
    profile runs the comparison.
    """
    workspace(root)
    for folder in ["conf", "workflow"]:
        (root / folder).unlink()
        shutil.copytree(
            REPO_ROOT / folder,
            root / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    display_path = root / "conf/scripts/plotting/common/default.yaml"
    display = yaml.safe_load(display_path.read_text())
    display["method_name_mapping"]["gdfs_adopter"] = "GDFS (adopter)"
    display["method_policy_family_mapping"]["gdfs_adopter"] = "gdfs"
    display["caption"] = CAPTION
    display_path.write_text(yaml.safe_dump(display))
    (root / "workflow/conf/adopter.yaml").write_text(
        yaml.safe_dump(ADOPTER_CONFIG)
    )
    profile = root / "workflow/profiles/pipeline/adopter/config.yaml"
    profile.parent.mkdir()
    profile.write_text(yaml.safe_dump(ADOPTER_PROFILE))
    return root


def launch(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Snakemake as `workflow/run.py` does in the fork."""
    # No Hugging Face token, and no service: the pipeline reads local files.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("HF_") and name != "SNAKEMAKE_PROFILE"
    }
    environment |= {
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(root / "no-hf-home"),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            *snakemake_arguments(
                [*LAUNCHER_ARGUMENTS, *arguments], checkout=root
            ),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )


def planned_jobs(output: str) -> Counter[tuple[str, str | None, str | None]]:
    """Count a listed dry run's jobs by rule, method and dataset."""
    jobs: Counter[tuple[str, str | None, str | None]] = Counter()
    for rule, block in re.findall(
        r"^rule (\w+):\n((?:    .*\n)*)", output, flags=re.MULTILINE
    ):
        found = re.search(r"^    wildcards: (.*)$", block, flags=re.MULTILINE)
        wildcards = dict(
            item.split("=", 1)
            for item in (found[1] if found else "").split(", ")
            if "=" in item
        )
        jobs[rule, wildcards.get("method"), wildcards.get("dataset")] += 1
    return jobs


def invoke(transport: FakeReleaseTransport, *args: str) -> Result:
    return CliRunner().invoke(
        app, [*args, "--repo-id", REPO_ID], obj=lambda _repo_id: transport
    )


def files_and_mtimes(root: Path) -> dict[str, float]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime
        for path in root.rglob("*")
        if path.is_file()
    }


class Journey:
    published: Path
    forked: Path
    transport: FakeReleaseTransport
    plan: subprocess.CompletedProcess[str]
    run: subprocess.CompletedProcess[str]
    downloaded: dict[str, float]


@pytest.fixture(scope="module")
def journey(tmp_path_factory: pytest.TempPathFactory) -> Journey:
    tmp_path = tmp_path_factory.mktemp("adopter-journey")
    result = Journey()

    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(yaml.safe_dump(SMOKE_SELECTION))
    baseline_configfiles = [*PROFILE_CONFIGFILES, baseline]
    result.published = workspace(tmp_path / "published")
    run = snakemake(result.published, baseline_configfiles, "all")
    assert run.returncode == 0, run.stdout + run.stderr

    package = tmp_path / "package"
    saved = CliRunner().invoke(
        app,
        [
            "save",
            str(package),
            "--source-root",
            str(result.published / "output/smoke"),
            *(
                argument
                for path in baseline_configfiles
                for argument in ["--configfile", str(path)]
            ),
            "--release-id",
            RELEASE_ID,
            "--checkout",
            str(REPO_ROOT),
        ],
    )
    assert saved.exit_code == 0, saved.output
    result.transport = FakeReleaseTransport()
    result.transport.host_locally(package)

    result.forked = fork(tmp_path / "fork")
    downloaded = invoke(
        result.transport,
        "download",
        "--destination-root",
        str(result.forked / "output/smoke"),
        *(
            argument
            for method in PUBLISHED_METHODS
            for argument in ["--method", method]
        ),
    )
    assert downloaded.exit_code == 0, downloaded.output
    assert "Missing from release" not in downloaded.output
    result.downloaded = files_and_mtimes(result.forked / "output/smoke")

    result.plan = launch(result.forked, "-n", "list_jobs=true")
    result.run = launch(result.forked)
    return result


@pytest.mark.pipeline
def test_fork_downloads_the_published_methods_whole_chains(
    journey: Journey,
) -> None:
    downloaded = set(journey.downloaded)

    for folder in [
        "datasets/cube/0/train.bundle",
        "datasets/cube/0/test.bundle",
        f"trained_classifiers/{TAG}/dataset-cube+realization_index-0.bundle",
        f"pretrained_models/{TAG}/gdfs/",
        *(f"trained_methods/{TAG}/{method}/" for method in PUBLISHED_METHODS),
        *(
            f"{tables}/eval_split-test/{TAG}/{method}/"
            for tables in ["eval_results", "eval_results_transformed"]
            for method in PUBLISHED_METHODS
        ),
    ]:
        assert any(path.startswith(folder) for path in downloaded), folder
    assert not [path for path in downloaded if path.startswith("plots/")]
    manifest = read_release_manifest(
        journey.forked / "output/smoke/release_manifest.json"
    )
    assert manifest.release_id == RELEASE_ID
    assert manifest.execution_mode is ExecutionMode.SMOKE


@pytest.mark.pipeline
def test_the_run_produces_only_the_adopters_method_and_the_new_dataset(
    journey: Journey,
) -> None:
    planned = journey.plan.stdout + journey.plan.stderr
    executed = journey.run.stdout + journey.run.stderr

    assert journey.plan.returncode == 0, planned
    jobs = planned_jobs(planned)
    produced = {
        job
        for job in jobs
        if job[0]
        in {
            "dataset_generation",
            "train_classifier",
            "train_classifier_for_method",
            "pretrain_model",
            "train_method",
            "eval_method",
            "transform_eval_data",
        }
    }
    assert produced == PRODUCED_JOBS, planned
    assert journey.run.returncode == 0, executed
    finished = Counter(
        # The launcher's terminal names each finished job's rule.
        re.findall(r"\[[^]]*\] Finished (\w+)", executed)
    )
    assert finished == Counter(job[0] for job in jobs.elements()), executed


@pytest.mark.pipeline
def test_the_run_leaves_every_downloaded_file_untouched(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    after = files_and_mtimes(journey.forked / "output/smoke")

    assert {path: after[path] for path in journey.downloaded} == (
        journey.downloaded
    )
    trained = next(
        (journey.forked / f"output/smoke/trained_methods/{TAG}").glob(
            "gdfs_adopter/dataset-cube+*/**/method.bundle"
        )
    )
    contract = json.dumps(
        json.loads((trained / "manifest.json").read_text())["metadata"][
            "contract"
        ]
    )
    for prerequisite in [
        "datasets/cube/0/train.bundle",
        "datasets/cube/0/val.bundle",
        f"trained_classifiers/{TAG}/dataset-cube+realization_index-0.bundle",
        f"pretrained_models/{TAG}/gdfs/dataset-cube+realization_index-0/"
        "pretrain_seed-0/model.bundle",
    ]:
        assert f"output/smoke/{prerequisite}" in contract, prerequisite


@pytest.mark.pipeline
def test_comparison_holds_each_published_row_once_on_every_dataset(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    output = journey.forked / "output/smoke"
    merged = pl.read_parquet(
        output / f"merged_results/{EVAL_PERF}/{COMPARISON}+all.parquet"
    )

    assert set(zip(merged["afa_method"], merged["dataset"], strict=True)) == {
        (method, dataset)
        for method in [*PUBLISHED_METHODS, "gdfs_adopter"]
        for dataset in ["cube", NEW_DATASET]
    }
    published = journey.published / "output/smoke"
    for method in PUBLISHED_METHODS:
        restored = pl.concat(
            pl.read_parquet(path)
            for path in sorted(
                (
                    published / f"eval_results_transformed/eval_split-test/"
                    f"{TAG}/{method}"
                ).rglob("eval_data.parquet")
            )
        )
        on_cube = (merged["afa_method"] == method) & (
            merged["dataset"] == "cube"
        )
        assert on_cube.sum() == len(restored)
    dummy = merged.filter(pl.col("afa_method") == "random_dummy")
    hard_budget = dummy["eval_hard_budget"].is_not_null()
    soft_budget = dummy["train_soft_budget_param"].is_not_null()
    assert hard_budget.any()
    assert soft_budget.any()
    assert not (hard_budget & soft_budget).any()
    for classifier in ["builtin", "external"]:
        split = pl.read_parquet(
            output / f"merged_results/{EVAL_PERF}/"
            f"{COMPARISON}+classifier_type-{classifier}.parquet"
        )
        # Built-in and external predictions are plotted apart.
        assert len(split) == (merged["classifier"] == classifier).sum()


@pytest.mark.pipeline
def test_comparison_plots_show_both_and_are_labelled_demonstrations(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    plot = (
        journey.forked / f"output/smoke/plots/{EVAL_PERF}/"
        f"{COMPARISON}+classifier_type-external/dataset_set-all/"
        "hard_budget_normal.svg"
    ).read_text()

    # Matplotlib keeps each text of an SVG as a comment beside its glyphs.
    for text in ["Random", "GDFS", "GDFS (adopter)", CAPTION]:
        assert f"<!-- {text} -->" in plot, text


@pytest.mark.pipeline
def test_results_only_journey_needs_no_afabench_loading(
    journey: Journey, tmp_path: Path
) -> None:
    destination = tmp_path / "results"
    downloaded = invoke(
        journey.transport,
        "download",
        RELEASE_ID,
        "--destination-root",
        str(destination),
        "--payload-category",
        "transformed_evaluation_table",
        "--payload-category",
        "raw_evaluation_table",
        "--output-category",
        "plots",
    )

    assert downloaded.exit_code == 0, downloaded.output
    manifest = json.loads((destination / "release_manifest.json").read_text())
    for table in manifest["evaluations"]:
        for key in ["raw_path", "transformed_path"]:
            # The host stores each table as an ordinary file, so a plain
            # download of its URL reads the same as the restored copy.
            hosted = journey.transport.files[
                f"releases/{RELEASE_ID}/output/{table[key]}"
            ]
            assert_frame_equal(
                pl.read_parquet(destination / table[key]),
                pl.read_parquet(io.BytesIO(hosted)),
            )
    # Plots open in any viewer: plain PDF and SVG files.
    for suffix, header in [(".pdf", b"%PDF"), (".svg", b"<?xml")]:
        plots = sorted((destination / "plots").rglob(f"*{suffix}"))
        assert plots, suffix
        for plot in plots:
            assert plot.read_bytes().startswith(header), plot
