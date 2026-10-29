"""The job duration table, loaded from loose job records or from Parquet."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from afabench.core.job_duration_table import (
    JobRecordFieldsError,
    UnknownJobRecordVersionError,
    load_job_duration_table,
    write_job_duration_table,
)
from afabench.core.job_record import Allocation, JobIdentity, run_job
from test.job_record_examples import job_record, write_job_record

DATASET_GENERATION = "datasets/cube/0/dataset_generation.job_record.json"
TRAINING = (
    "trained_methods/initializer-cold/alpha/dataset-cube+realization_index-0/"
    "NO_PRETRAIN/train_seed-0+train_hard_budget-1+train_soft_budget_param-null"
)


def run_recorded_job(
    output_root: Path,
    record: str,
    *,
    succeeds: bool,
    identity: JobIdentity,
    allocation: Allocation,
) -> dict[str, Any]:
    """Run a job as the pipeline does, leaving its record under the root."""
    return run_job(
        ["true" if succeeds else "false"],
        identity=identity,
        allocation=allocation,
        smoke_test=False,
        record_path=output_root / record,
        failed_record_path=output_root / "failed_job_records" / record,
    ).to_json_dict()


@dataclass
class RecordedJobs:
    output_root: Path
    dataset_generation: dict[str, Any]
    failed_training_attempts: list[dict[str, Any]]
    completed_training: dict[str, Any]


def record_jobs(tmp_path: Path) -> RecordedJobs:
    """Run a dataset generation job and three attempts of a training job."""
    output_root = tmp_path / "output"
    dataset_generation = run_recorded_job(
        output_root,
        DATASET_GENERATION,
        succeeds=True,
        identity=JobIdentity(
            stage="dataset_generation",
            dataset_key="cube",
            dataset_realization_index=0,
        ),
        allocation=Allocation(
            device="cpu", cpus=None, gpus=0, time_limit_minutes=None
        ),
    )
    training = JobIdentity(
        stage="training",
        name="alpha",
        dataset_key="cube",
        dataset_realization_index=0,
        train_seed=0,
        train_hard_budget=1,
    )
    gpu = Allocation(device="cuda", cpus=8, gpus=1, time_limit_minutes=600)
    attempts = [
        run_recorded_job(
            output_root,
            f"{TRAINING}/method.job_record.json",
            succeeds=succeeds,
            identity=training,
            allocation=gpu,
        )
        for succeeds in [False, False, True]
    ]
    return RecordedJobs(
        output_root, dataset_generation, attempts[:2], attempts[2]
    )


def test_an_output_root_loads_one_row_per_job_record_including_failed(
    tmp_path: Path,
) -> None:
    jobs = record_jobs(tmp_path)

    table = load_job_duration_table(jobs.output_root)

    assert len(table) == 4
    rows = {row.pop("job_record_path"): row for row in table.to_dicts()}
    assert rows[DATASET_GENERATION]["stage"] == "dataset_generation"
    assert rows[DATASET_GENERATION]["cpus"] is None
    assert rows[DATASET_GENERATION]["name"] is None
    assert rows[f"{TRAINING}/method.job_record.json"]["cpus"] == 8
    failed = table.filter(pl.col("exit_status") == "failed")
    assert sorted(failed["job_duration_seconds"]) == sorted(
        attempt["job_duration_seconds"]
        for attempt in jobs.failed_training_attempts
    )
    assert all(
        path.startswith(f"failed_job_records/{TRAINING}/method.")
        for path in failed["job_record_path"]
    )
    for path, record in [
        (DATASET_GENERATION, jobs.dataset_generation),
        (f"{TRAINING}/method.job_record.json", jobs.completed_training),
    ]:
        assert rows[path] == {
            field: value
            for field, value in record.items()
            if field not in {"started_at", "ended_at"}
        } | {
            "started_at": datetime.fromisoformat(record["started_at"]),
            "ended_at": datetime.fromisoformat(record["ended_at"]),
        }


def test_a_table_loads_the_same_rows_as_the_output_root_it_was_built_from(
    tmp_path: Path,
) -> None:
    jobs = record_jobs(tmp_path)
    table_path = tmp_path / "job_duration_table.parquet"

    write_job_duration_table(jobs.output_root, table_path)

    assert_frame_equal(
        load_job_duration_table(table_path),
        load_job_duration_table(jobs.output_root),
    )


def test_a_job_record_of_an_unknown_version_is_rejected(
    tmp_path: Path,
) -> None:
    jobs = record_jobs(tmp_path)
    path = jobs.output_root / DATASET_GENERATION
    path.write_text(
        json.dumps(json.loads(path.read_text()) | {"job_record_version": 2})
    )

    with pytest.raises(UnknownJobRecordVersionError, match=r"version 2.*cube"):
        load_job_duration_table(jobs.output_root)


def test_a_table_holding_an_unknown_job_record_version_is_rejected(
    tmp_path: Path,
) -> None:
    jobs = record_jobs(tmp_path)
    table_path = tmp_path / "job_duration_table.parquet"
    write_job_duration_table(jobs.output_root, table_path)
    table = pl.read_parquet(table_path).with_columns(
        pl.lit(2, dtype=pl.Int64).alias("job_record_version")
    )
    table.write_parquet(table_path)

    with pytest.raises(UnknownJobRecordVersionError, match="version 2"):
        load_job_duration_table(table_path)


def test_a_job_record_missing_a_field_of_its_version_is_rejected(
    tmp_path: Path,
) -> None:
    jobs = record_jobs(tmp_path)
    path = jobs.output_root / DATASET_GENERATION
    record = json.loads(path.read_text())
    del record["cpus"]
    path.write_text(json.dumps(record))

    with pytest.raises(JobRecordFieldsError, match=r"cpus.*cube"):
        load_job_duration_table(jobs.output_root)


def test_empty_output_root_preserves_nullable_parquet_types(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "table.parquet"
    write_job_duration_table(tmp_path, table_path)

    table = load_job_duration_table(table_path)

    assert isinstance(table, pl.DataFrame)
    assert table.is_empty()
    schema = pq.read_schema(table_path)
    assert schema.field("cpus").type == pa.int64()
    assert schema.field("job_duration_seconds").type == pa.float64()
    assert schema.field("smoke_test").type == pa.bool_()
    for column in ["started_at", "ended_at"]:
        assert schema.field(column).type == pa.timestamp("us", tz="UTC")
    assert all(field.nullable for field in schema)


def test_timestamps_are_nullable_utc_microseconds_and_rows_sorted(
    tmp_path: Path,
) -> None:
    for path, started_at in [
        ("z.job_record.json", None),
        ("a.job_record.json", "2026-10-08T14:00:00.123456+02:00"),
    ]:
        write_job_record(
            tmp_path / path,
            job_record(JobIdentity(stage="training"), started_at=started_at),
        )

    table = load_job_duration_table(tmp_path)

    assert table["job_record_path"].to_list() == [
        "a.job_record.json",
        "z.job_record.json",
    ]
    assert table.schema["started_at"] == pl.Datetime("us", "UTC")
    assert table["started_at"].to_list() == [
        datetime(2026, 10, 8, 12, 0, 0, 123456, tzinfo=UTC),
        None,
    ]


def test_unknown_timestamps_and_durations_stay_null_on_round_trip(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    write_job_record(
        output_root / "unknown.job_record.json",
        job_record(
            JobIdentity(stage="training"),
            started_at=None,
            ended_at=None,
            job_duration_seconds=None,
        ),
    )
    table_path = tmp_path / "table.parquet"

    write_job_duration_table(output_root, table_path)

    table = load_job_duration_table(table_path)
    assert table.select("started_at", "ended_at", "job_duration_seconds").row(
        0
    ) == (None, None, None)
    assert table.schema["started_at"] == pl.Datetime("us", "UTC")
