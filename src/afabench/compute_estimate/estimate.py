"""
Estimate the compute of planned jobs from measured job durations.

The second half of a compute estimate
(`docs/adr/0007-job-records-beside-artifacts.md`): each planned job is
matched to the completed job records of a job duration table, in a fixed
fallback order. An exact match has the same identity and device. Otherwise
the job is pooled with every job of its stage, name, dataset key and
device, whatever their seeds, dataset realizations, hard budgets and
soft-budget parameters. Otherwise it is unestimated. Matching never crosses
device or dataset key, and durations are not normalized across hardware.

Failed and timed-out job records give no job duration. Instead, each
planned job type (stage, name and dataset key) that has them gets a failure
history: its jobs may fail or hit their time limit again, so their estimate
may be low.

The minimal wall time is the longest chain of planned jobs, each depending
on the one before, with every job taking its estimated job duration: how
long the run takes if every job starts as soon as the jobs it depends on
are done.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, fields
from graphlib import TopologicalSorter
from typing import Literal, cast

import polars as pl

from afabench.core.job_record import JobIdentity, JobType
from afabench.pipeline_plan.planning import PlannedJob

# no_job_record: the job's rule writes no job record, so no job duration
# can match it; aggregation and visualization jobs are not computational.
type MatchLevel = Literal["exact", "pooled", "unestimated", "no_job_record"]

IDENTITY_COLUMNS = [field.name for field in fields(JobIdentity)]
EXACT_COLUMNS = [*IDENTITY_COLUMNS, "device"]
JOB_TYPE_COLUMNS = [field.name for field in fields(JobType)]
POOL_COLUMNS = [*JOB_TYPE_COLUMNS, "device"]

# Which of a job's job durations to take
type Statistic = Literal["mean", "p90"]

# The values of a job's identity columns and device, None for null
type Key = tuple[object, ...]


@dataclass(frozen=True, kw_only=True)
class JobEstimate:
    """A planned job and the job durations it was matched to."""

    job: PlannedJob
    match_level: MatchLevel
    matched_job_records: int
    # None when unestimated
    mean_job_duration_seconds: float | None
    p90_job_duration_seconds: float | None

    @property
    def mean_job_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, 1)

    @property
    def p90_job_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, 1)

    # Core-hours and GPU-hours use the planned allocation: the measured
    # job may have run with another one.
    @property
    def mean_core_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, self.job.cpus)

    @property
    def p90_core_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, self.job.cpus)

    @property
    def mean_gpu_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, self.job.gpus)

    @property
    def p90_gpu_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, self.job.gpus)


@dataclass(frozen=True, kw_only=True)
class Hardware:
    """Where matched job durations were measured, distinct and sorted."""

    # The sites, as far as SLURM names them
    slurm_clusters: list[str]
    hosts: list[str]
    cpu_models: list[str]
    gpu_models: list[str]


@dataclass(frozen=True, kw_only=True)
class FailureHistory:
    """The failed and timed-out job records of a planned job type."""

    job_type: JobType
    failed: int
    timed_out: int
    # Distinct known time limits of the timed-out jobs, sorted
    time_limits_minutes: list[int]


@dataclass(frozen=True, kw_only=True)
class ComputeEstimate:
    jobs: list[JobEstimate]
    # Job records some job was matched to
    matched_job_records: int
    hardware: Hardware
    # Job records of smoke tests, never used as job durations
    refused_smoke_test_job_records: int
    # Sorted by job type
    failure_histories: list[FailureHistory]


@dataclass(frozen=True, kw_only=True)
class WallTime:
    """The longest chain of dependent planned jobs, and how long it takes."""

    seconds: float
    # In the order the jobs run
    chain: list[JobEstimate]


def minimal_wall_time(
    estimate: ComputeEstimate, statistic: Statistic
) -> WallTime:
    """
    Return the wall time of the planned jobs, all run as early as possible.

    Each job takes its `statistic` job duration. Unestimated jobs and jobs
    of rules without job records take none, but still chain the jobs
    before them to the jobs after them.
    """
    jobs = {job.job.jobid: job for job in estimate.jobs}
    # The longest chain ending with each job, and how long it takes
    chains: dict[int, tuple[float, list[JobEstimate]]] = {}
    order = TopologicalSorter(
        {jobid: job.job.dependencies for jobid, job in jobs.items()}
    ).static_order()
    for jobid in order:
        job = jobs[jobid]
        seconds, chain = max(
            (chains[dependency] for dependency in job.job.dependencies),
            key=_longest,
            default=(0.0, []),
        )
        duration = (
            job.mean_job_duration_seconds
            if statistic == "mean"
            else job.p90_job_duration_seconds
        )
        chains[jobid] = (seconds + (duration or 0.0), [*chain, job])
    seconds, chain = max(chains.values(), key=_longest, default=(0.0, []))
    return WallTime(seconds=seconds, chain=chain)


def _longest(ending: tuple[float, list[JobEstimate]]) -> tuple[float, int]:
    # Of chains that take as long, the one with more jobs, so that jobs
    # that take no time are not left off its end.
    seconds, chain = ending
    return seconds, len(chain)


def estimate_compute(
    planned_jobs: Sequence[PlannedJob], job_durations: pl.DataFrame
) -> ComputeEstimate:
    """
    Estimate planned jobs from a job duration table's rows.

    `job_durations` has the columns of `load_job_duration_table`.
    """
    # Only completed jobs ran for their whole job duration. Smoke-test jobs
    # did too, but would make the estimate far too low.
    smoke_test = job_durations["smoke_test"].fill_null(False)
    usable = job_durations.filter(
        (pl.col("exit_status") == "completed") & ~smoke_test
    )
    # Positions of the usable job records per key
    exact: defaultdict[Key, list[int]] = defaultdict(list)
    pools: defaultdict[Key, list[int]] = defaultdict(list)
    for position, record in enumerate(
        usable.select(EXACT_COLUMNS).iter_rows(named=True)
    ):
        exact[_key(record, EXACT_COLUMNS)].append(position)
        pools[_key(record, POOL_COLUMNS)].append(position)
    matches = [_match(job, exact, pools) for job in planned_jobs]
    durations = usable["job_duration_seconds"].fill_nan(None)
    matched = usable[
        sorted(
            {position for _, positions in matches for position in positions}
        )
    ]
    return ComputeEstimate(
        jobs=[
            _estimate(job, match_level, durations.gather(positions))
            for job, (match_level, positions) in zip(
                planned_jobs, matches, strict=True
            )
        ],
        matched_job_records=len(matched),
        hardware=Hardware(
            slurm_clusters=_distinct(matched["slurm_cluster"]),
            hosts=_distinct(matched["host"]),
            cpu_models=_distinct(matched["cpu_model"]),
            gpu_models=_distinct(matched["gpu_model"]),
        ),
        refused_smoke_test_job_records=int(smoke_test.sum()),
        failure_histories=_failure_histories(
            planned_jobs,
            job_durations.filter(
                pl.col("exit_status").is_in(["failed", "timeout"])
                & ~smoke_test
            ),
        ),
    )


def _failure_histories(
    planned_jobs: Sequence[PlannedJob], failures: pl.DataFrame
) -> list[FailureHistory]:
    """Return the failure history of each planned job type that has one."""
    planned_types = {
        job.job_type for job in planned_jobs if job.job_type is not None
    }
    timed_out: defaultdict[Key, int] = defaultdict(int)
    failed: defaultdict[Key, int] = defaultdict(int)
    time_limits: defaultdict[Key, set[int]] = defaultdict(set)
    for record in failures.select(
        [*JOB_TYPE_COLUMNS, "exit_status", "time_limit_minutes"]
    ).iter_rows(named=True):
        key = _key(record, JOB_TYPE_COLUMNS)
        if record["exit_status"] == "timeout":
            timed_out[key] += 1
            if isinstance(limit := record["time_limit_minutes"], int):
                time_limits[key].add(limit)
        else:
            failed[key] += 1
    histories = []
    for job_type in sorted(
        planned_types, key=lambda job_type: tuple(map(str, astuple(job_type)))
    ):
        key = astuple(job_type)
        if failed[key] or timed_out[key]:
            histories.append(
                FailureHistory(
                    job_type=job_type,
                    failed=failed[key],
                    timed_out=timed_out[key],
                    time_limits_minutes=sorted(time_limits[key]),
                )
            )
    return histories


def _match(
    job: PlannedJob,
    exact: Mapping[Key, list[int]],
    pools: Mapping[Key, list[int]],
) -> tuple[MatchLevel, list[int]]:
    """Return the match level and the matched job records' positions."""
    if job.identity is None:
        return "no_job_record", []
    values = {**asdict(job.identity), "device": job.device}
    if positions := exact.get(_key(values, EXACT_COLUMNS)):
        return "exact", positions
    if positions := pools.get(_key(values, POOL_COLUMNS)):
        return "pooled", positions
    return "unestimated", []


def _estimate(
    job: PlannedJob, match_level: MatchLevel, durations: pl.Series
) -> JobEstimate:
    return JobEstimate(
        job=job,
        match_level=match_level,
        matched_job_records=len(durations),
        mean_job_duration_seconds=cast("float | None", durations.mean()),
        p90_job_duration_seconds=durations.quantile(
            0.9, interpolation="linear"
        ),
    )


def _hours(seconds: float | None, count: int | None) -> float | None:
    if seconds is None or count is None:
        return None
    return seconds / 3600 * count


def _key(values: Mapping[str, object], columns: list[str]) -> Key:
    return tuple(values[column] for column in columns)


def _distinct(values: pl.Series) -> list[str]:
    return sorted(str(value) for value in values.drop_nulls().unique())
