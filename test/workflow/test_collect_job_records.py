"""The job duration table the aggregation rule collects, and the time plot of it."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from afabench.core.job_duration_table import load_job_duration_table
from test.workflow.submission_harness import REPO_ROOT
from test.workflow.test_cpu_processing_execution import processing_workflow

TAG = "initializer-cold"
REALIZATION = "dataset-cube+realization_index-0"
BUDGETS = "train_seed-0+train_hard_budget-1+train_soft_budget_param-null"
EVALUATION = "eval_seed-0+eval_hard_budget-1+eval_soft_budget_param-null"
ALPHA = f"alpha/{REALIZATION}/pretrain_seed-0/{BUDGETS}"
BETA = f"beta/{REALIZATION}/NO_PRETRAIN/{BUDGETS}"
# The record of each job of processing_workflow's graph that leaves one.
COMPLETED_RECORDS = {
    "datasets/cube/0/dataset_generation.job_record.json",
    f"trained_classifiers/{TAG}/{REALIZATION}.job_record.json",
    f"pretrained_models/{TAG}/shared/{REALIZATION}/pretrain_seed-0/"
    "model.job_record.json",
    *(
        f"trained_methods/{TAG}/{training}/method.job_record.json"
        for training in [ALPHA, BETA]
    ),
    *(
        f"{stage}/eval_split-test/{TAG}/{training}/{EVALUATION}/"
        "eval_data.job_record.json"
        for stage in ["eval_results", "eval_results_transformed"]
        for training in [ALPHA, BETA]
    ),
}
TABLE = "merged_results/job_duration_table.parquet"


@dataclass
class CollectedRun:
    output: Path
    table: pl.DataFrame


@pytest.fixture(scope="module")
def collected_run(tmp_path_factory: pytest.TempPathFactory) -> CollectedRun:
    """Run `all` after a failed attempt, with the real time plot script."""
    root = tmp_path_factory.mktemp("collected")
    workflow = processing_workflow(root)
    shutil.copyfile(
        REPO_ROOT / "scripts/plotting/plot_total_time.py",
        root / "scripts/plotting/plot_total_time.py",
    )
    shutil.copytree(REPO_ROOT / "conf", root / "conf")
    beta = root / "scripts/train_method/beta.py"
    script = beta.read_text()
    beta.write_text("import sys\nsys.exit(3)\n")
    failed = workflow.run("--executor", "local", target="all")
    assert failed.returncode != 0
    beta.write_text(script)

    result = workflow.run("--executor", "local", target="all")

    assert result.returncode == 0, result.stdout + result.stderr
    output = root / "output/smoke"
    return CollectedRun(
        output=output, table=load_job_duration_table(output / TABLE)
    )


def test_the_table_holds_one_row_per_job_that_ran_including_failed_attempts(
    collected_run: CollectedRun,
) -> None:
    table = collected_run.table
    completed = table.filter(pl.col("exit_status") == "completed")
    assert sorted(completed["job_record_path"]) == sorted(COMPLETED_RECORDS)
    failed_attempts = table.filter(pl.col("exit_status") == "failed")
    assert list(failed_attempts["name"]) == ["beta"]
    [failed_path] = list(failed_attempts["job_record_path"])
    assert failed_path.startswith(
        f"failed_job_records/trained_methods/{TAG}/{BETA}/method."
    )
    assert len(table) == len(COMPLETED_RECORDS) + 1
    assert_frame_equal(table, load_job_duration_table(collected_run.output))


def test_the_time_plot_renders_from_the_table(
    collected_run: CollectedRun,
) -> None:
    time_plots = collected_run.output / f"plots/eval_split-test/{TAG}/time"
    for plot in ["average_time", "dataset_time"]:
        for suffix in [".pdf", ".svg"]:
            assert (time_plots / plot).with_suffix(suffix).is_file()
