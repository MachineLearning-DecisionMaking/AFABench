"""
The job duration table: every job record of a pipeline's outputs as one row.

Design: `docs/adr/0007-job-records-beside-artifacts.md`; columns:
`docs/reference/job_records.md`. Rows are not aggregated: each job record,
completed, failed or timed out, is one row, so repeated attempts of one job
are separate rows. `load_job_duration_table` reads either an output root of
loose job records or a job duration table written from one, and returns the
same frame for both.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from afabench.core.job_record import (
    JOB_RECORD_SUFFIX,
    JOB_RECORD_VERSION,
)

KNOWN_JOB_RECORD_VERSIONS = {JOB_RECORD_VERSION}

# One column per field of a job record (`JobRecord`). Nullable dtypes, since
# null means unknown.
RECORD_FIELD_DTYPES = pl.Schema(
    {
        "job_record_version": pl.Int64,
        "stage": pl.String,
        "name": pl.String,
        "dataset_key": pl.String,
        "dataset_realization_index": pl.Int64,
        "pretrain_seed": pl.Int64,
        "train_seed": pl.Int64,
        "eval_seed": pl.Int64,
        "train_hard_budget": pl.Int64,
        "train_soft_budget_param": pl.Float64,
        "eval_hard_budget": pl.Int64,
        "eval_soft_budget_param": pl.Float64,
        "eval_batch_size": pl.Int64,
        "started_at": pl.Datetime("us", "UTC"),
        "ended_at": pl.Datetime("us", "UTC"),
        "job_duration_seconds": pl.Float64,
        "exit_status": pl.String,
        "exit_code": pl.Int64,
        "device": pl.String,
        "cpus": pl.Int64,
        "gpus": pl.Int64,
        "time_limit_minutes": pl.Int64,
        "gpu_model": pl.String,
        "cpu_model": pl.String,
        "host": pl.String,
        "slurm_job_id": pl.String,
        "slurm_cluster": pl.String,
        "code_commit": pl.String,
        "smoke_test": pl.Boolean,
    }
)
# The path of the row's job record relative to the output root comes first.
COLUMN_DTYPES = pl.Schema(
    {"job_record_path": pl.String, **RECORD_FIELD_DTYPES}
)


class UnknownJobRecordVersionError(ValueError):
    """A job record, or a table row, has a schema version this code lacks."""


class JobRecordFieldsError(ValueError):
    """A job record, or a table, lacks or adds fields of its version."""


def load_job_duration_table(source: Path) -> pl.DataFrame:
    """
    Load a job duration table, or the job records under an output root.

    `source` is a table's Parquet file or an output root directory. Rows
    are sorted by `job_record_path`.
    """
    if not source.is_dir():
        table = pl.read_parquet(source)
        for version in table["job_record_version"].unique():
            _check_version(version, source)
        _check_fields(table.columns, COLUMN_DTYPES, source)
        return _typed(table.select(COLUMN_DTYPES.names()))
    rows = [
        {
            "job_record_path": path.relative_to(source).as_posix(),
            **_read_job_record(path),
        }
        for path in source.rglob(f"*{JOB_RECORD_SUFFIX}")
    ]
    if not rows:
        return empty_job_duration_table()
    return _typed(
        pl.from_dicts(
            rows,
            schema={
                **COLUMN_DTYPES,
                "started_at": pl.String,
                "ended_at": pl.String,
            },
        )
    )


def empty_job_duration_table() -> pl.DataFrame:
    """Return a job duration table without rows, as of an empty output root."""
    return pl.DataFrame(schema=COLUMN_DTYPES)


def write_job_duration_table(output_root: Path, table_path: Path) -> None:
    """Write the job duration table of an output root's job records."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    load_job_duration_table(output_root).write_parquet(table_path)


def _read_job_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text())
    _check_version(record.get("job_record_version"), path)
    _check_fields(record, RECORD_FIELD_DTYPES, path)
    return record


def _check_version(version: object, source: Path) -> None:
    if version not in KNOWN_JOB_RECORD_VERSIONS:
        message = (
            f"Unknown job record version {version} in {source}; known "
            f"versions: {sorted(KNOWN_JOB_RECORD_VERSIONS)}"
        )
        raise UnknownJobRecordVersionError(message)


def _check_fields(
    present: Iterable[str], expected: Iterable[str], source: Path
) -> None:
    missing = set(expected) - set(present)
    unexpected = set(present) - set(expected)
    if missing or unexpected:
        message = (
            f"Missing job record fields {sorted(missing)} and unexpected "
            f"fields {sorted(unexpected)} in {source}"
        )
        raise JobRecordFieldsError(message)


def _typed(table: pl.DataFrame) -> pl.DataFrame:
    return (
        table.with_columns(
            pl.col(column).str.to_datetime(time_unit="us", time_zone="UTC")
            if table.schema[column] == pl.String
            else pl.col(column).cast(pl.Datetime("us", "UTC"))
            for column in ["started_at", "ended_at"]
        )
        .cast(COLUMN_DTYPES)
        .sort("job_record_path", maintain_order=True)
    )
