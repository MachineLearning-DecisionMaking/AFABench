"""
Report a compute estimate: per-job rows, and totals per group of jobs.

`estimate-compute` prints `format_report` and writes `per_job_table` as
CSV. Totals hold only estimated jobs, as mean and p90 job-hours,
core-hours and GPU-hours; a p90 total adds up each job's p90 job duration.
The minimal wall time follows them, with its longest chain of jobs.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict

import polars as pl

from afabench.compute_estimate.estimate import (
    IDENTITY_COLUMNS,
    POOL_COLUMNS,
    ComputeEstimate,
    FailureHistory,
    JobEstimate,
    MatchLevel,
    minimal_wall_time,
)
from afabench.core.job_duration_table import RECORD_FIELD_DTYPES
from afabench.core.job_record import JobType

JOB_COLUMNS = [
    "rule",
    "wildcards",
    *IDENTITY_COLUMNS,
    "device",
    "cpus",
    "gpus",
    "time_limit_minutes",
]
DURATION_COLUMNS = ["mean_job_duration_seconds", "p90_job_duration_seconds"]
# What totals add up
HOUR_COLUMNS = [
    "mean_job_hours",
    "p90_job_hours",
    "mean_core_hours",
    "p90_core_hours",
    "mean_gpu_hours",
    "p90_gpu_hours",
]
ESTIMATE_COLUMNS = [
    "match_level",
    "matched_job_records",
    *DURATION_COLUMNS,
    *HOUR_COLUMNS,
]
# As in the job duration table, nullable
PER_JOB_DTYPES = pl.Schema(
    {
        "rule": pl.String,
        "wildcards": pl.String,
        **{column: RECORD_FIELD_DTYPES[column] for column in IDENTITY_COLUMNS},
        "device": pl.String,
        "cpus": pl.Int64,
        "gpus": pl.Int64,
        "time_limit_minutes": pl.Int64,
        "match_level": pl.String,
        "matched_job_records": pl.Int64,
        **dict.fromkeys([*DURATION_COLUMNS, *HOUR_COLUMNS], pl.Float64),
        "failure_history": pl.Boolean,
        "jobid": pl.Int64,
        "dependencies": pl.String,
    }
)
GROUPING_COLUMNS = [column for column in JOB_COLUMNS if column != "wildcards"]
DEFAULT_GROUPING = ["stage", "device"]
COUNTED_MATCH_LEVELS: list[MatchLevel] = ["exact", "pooled", "unestimated"]


class UnknownGroupingColumnError(ValueError):
    """Totals were asked to be grouped by a column jobs do not have."""


def per_job_table(estimate: ComputeEstimate) -> pl.DataFrame:
    """
    Return one row per planned job: identity, allocation, estimate.

    `failure_history` flags jobs whose job type failed or timed out before.
    """
    failed_types = {history.job_type for history in estimate.failure_histories}
    return pl.DataFrame(
        [_per_job_row(job, failed_types) for job in estimate.jobs],
        schema=PER_JOB_DTYPES,
    )


def _per_job_row(
    job_estimate: JobEstimate, failed_types: set[JobType]
) -> dict[str, object]:
    job = job_estimate.job
    return {
        "rule": job.rule,
        "wildcards": json.dumps(job.wildcards),
        **(
            dict.fromkeys(IDENTITY_COLUMNS)
            if job.identity is None
            else asdict(job.identity)
        ),
        "device": job.device,
        "cpus": job.cpus,
        "gpus": job.gpus,
        "time_limit_minutes": job.time_limit_minutes,
        **{
            column: getattr(job_estimate, column)
            for column in ESTIMATE_COLUMNS
        },
        "failure_history": job.job_type in failed_types,
        "jobid": job.jobid,
        "dependencies": json.dumps(sorted(job.dependencies)),
    }


def check_grouping(by: Sequence[str]) -> None:
    """Raise `UnknownGroupingColumnError` unless jobs have columns `by`."""
    unknown = [column for column in by if column not in GROUPING_COLUMNS]
    if unknown or not by:
        message = (
            f"Cannot group by {unknown or 'no column'}; jobs have the "
            f"columns {GROUPING_COLUMNS}"
        )
        raise UnknownGroupingColumnError(message)


def group_totals(
    estimate: ComputeEstimate, by: Sequence[str] = DEFAULT_GROUPING
) -> pl.DataFrame:
    """
    Total the estimated jobs' hours, mean and p90, per group of jobs.

    Each group also counts its jobs per match level. Unestimated jobs are
    counted but not in the totals, and jobs of rules without job records
    are left out. p90 totals add up each job's p90.
    """
    check_grouping(by)
    jobs = per_job_table(estimate)
    counted = jobs.filter(pl.col("match_level").is_in(COUNTED_MATCH_LEVELS))
    return (
        counted.group_by(list(by))
        .agg(
            pl.len().cast(pl.Int64).alias("jobs"),
            *(
                (pl.col("match_level") == level)
                .sum()
                .cast(pl.Int64)
                .alias(level)
                for level in COUNTED_MATCH_LEVELS
            ),
            *(_nullable_sum(column) for column in HOUR_COLUMNS),
        )
        .sort(list(by), nulls_last=True)
    )


def _nullable_sum(column: str) -> pl.Expr:
    # Polars sums an all-null group to zero; unknown hours must stay unknown.
    return (
        pl.when(pl.col(column).count() > 0)
        .then(pl.col(column).sum())
        .alias(column)
    )


def format_report(
    estimate: ComputeEstimate,
    *,
    source: str,
    by: Sequence[str] = DEFAULT_GROUPING,
) -> str:
    """Return the report `estimate-compute` prints, naming `source`."""
    jobs = per_job_table(estimate)
    counts = dict(jobs["match_level"].value_counts().iter_rows())
    hardware = estimate.hardware
    lines = [
        f"Compute estimate of {_plural(len(jobs), 'planned job')}: "
        + ", ".join(
            f"{counts.get(level, 0)} {level}" for level in COUNTED_MATCH_LEVELS
        )
        + f", {counts.get('no_job_record', 0)} without job records",
        f"Job durations from {source}: "
        f"{_plural(estimate.matched_job_records, 'job record')} matched; "
        f"SLURM clusters: {_listing(hardware.slurm_clusters)}; "
        f"hosts: {_listing(hardware.hosts)}; "
        f"CPU models: {_listing(hardware.cpu_models)}; "
        f"GPU models: {_listing(hardware.gpu_models)}",
    ]
    refused = estimate.refused_smoke_test_job_records
    if refused:
        lines.append(f"Refused {_plural(refused, 'smoke-test job record')}")
    totals = group_totals(estimate, by)
    overall = totals.select(
        *(
            pl.lit("total" if column == by[0] else "").alias(column)
            for column in by
        ),
        pl.col(["jobs", *COUNTED_MATCH_LEVELS]).sum(),
        *(_nullable_sum(column) for column in HOUR_COLUMNS),
    )
    totals = pl.concat(
        [totals.with_columns(pl.col(list(by)).cast(pl.String)), overall]
    )
    lines += [
        "",
        f"Totals by {' and '.join(by)} in hours, unestimated jobs not "
        "included:",
        _table(totals),
        "",
        *_wall_time_lines(estimate),
        "",
    ]
    unestimated = jobs.filter(pl.col("match_level") == "unestimated")
    if unestimated.is_empty():
        lines.append("Unestimated jobs: none")
    else:
        lines += [
            "Unestimated jobs, not in the totals:",
            _table(
                unestimated.group_by(POOL_COLUMNS)
                .len(name="jobs")
                .sort(POOL_COLUMNS, nulls_last=True)
            ),
        ]
    lines.append("")
    if estimate.failure_histories:
        lines += [
            "Failed or timed out before, so the estimate of these jobs may "
            "be low:",
            *(
                f"  {_failure_warning(history)}"
                for history in estimate.failure_histories
            ),
            "",
        ]
    without_records = (
        jobs.filter(pl.col("match_level") == "no_job_record")
        .group_by("rule", maintain_order=True)
        .len()
    )
    if not without_records.is_empty():
        lines.append(
            "Not estimated, rules without job records: "
            + ", ".join(
                f"{rule} ({count})"
                for rule, count in without_records.iter_rows()
            )
        )
    unknown_cpus = (
        jobs["mean_core_hours"].is_null()
        & jobs["mean_job_hours"].is_not_null()
    )
    if unknown_cpus.any():
        lines.append(
            f"Core-hours leave out {_plural(int(unknown_cpus.sum()), 'job')}"
            " whose CPUs the cluster chooses."
        )
    return "\n".join(lines) + "\n"


def _wall_time_lines(estimate: ComputeEstimate) -> list[str]:
    if all(job.mean_job_duration_seconds is None for job in estimate.jobs):
        return ["Minimal wall time: unknown, no planned job is estimated"]
    mean = minimal_wall_time(estimate, "mean")
    p90 = minimal_wall_time(estimate, "p90")
    lines = [
        "Minimal wall time, every job starting once the jobs it needs are "
        f"done: mean {mean.seconds / 3600:.2f} h, "
        f"p90 {p90.seconds / 3600:.2f} h"
    ]
    if any(job.mean_job_duration_seconds is None for job in estimate.jobs):
        lines.append(
            "  It counts unestimated jobs and jobs of rules without job "
            "records as taking no time."
        )
    hours = [
        "-" if job.mean_job_hours is None else f"{job.mean_job_hours:.2f}"
        for job in mean.chain
    ]
    width = max(map(len, hours))
    lines.append("  Longest chain at mean job durations:")
    for job, value in zip(mean.chain, hours, strict=True):
        unit = "  " if job.mean_job_hours is None else " h"
        lines.append(f"    {value.rjust(width)}{unit}  {_describe_job(job)}")
    return lines


def _describe_job(job_estimate: JobEstimate) -> str:
    """Return e.g. "training alpha on mnist (unestimated)"."""
    job_type = job_estimate.job.job_type
    if job_type is None:
        return f"{job_estimate.job.rule} (no job record)"
    description = f"{job_type.stage} {_names(job_type)}"
    if job_estimate.match_level == "unestimated":
        description += " (unestimated)"
    return description


def _names(job_type: JobType) -> str:
    """Return e.g. "alpha on mnist", leaving out what the job type lacks."""
    return " on ".join(
        value
        for value in [job_type.name, job_type.dataset_key]
        if value is not None
    )


def _failure_warning(history: FailureHistory) -> str:
    """Return e.g. "training alpha on mnist timed out 3 times at 600 min"."""
    job_type = history.job_type
    names = _names(job_type)
    attempts = []
    if history.timed_out:
        limits = history.time_limits_minutes
        at = (
            f"at {' and '.join(map(str, limits))} min"
            if limits
            else "at an unknown time limit"
        )
        attempts.append(f"timed out {_plural(history.timed_out, 'time')} {at}")
    if history.failed:
        attempts.append(f"failed {_plural(history.failed, 'time')}")
    return f"{job_type.stage} {names} {', '.join(attempts)}"


def _table(table: pl.DataFrame) -> str:
    columns = [
        [
            column,
            *(
                "-"
                if value is None
                else f"{value:.2f}"
                if dtype.is_float()
                else str(value)
                for value in table[column]
            ),
        ]
        for column, dtype in table.schema.items()
    ]
    widths = [max(map(len, column)) for column in columns]
    return "\n".join(
        " ".join(
            value.rjust(width)
            for value, width in zip(row, widths, strict=True)
        )
        for row in zip(*columns, strict=True)
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _listing(values: list[str]) -> str:
    return ", ".join(values) or "unknown"
