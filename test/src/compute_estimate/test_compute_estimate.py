"""
The compute estimate of planned jobs from measured job durations (#83).

Job durations come from job records written under an output root and
loaded as the job duration table, as `estimate-compute` loads them.
"""

from collections.abc import Iterable
from dataclasses import replace
from itertools import count
from pathlib import Path

import polars as pl
import pytest

from afabench.compute_estimate.estimate import (
    FailureHistory,
    estimate_compute,
    minimal_wall_time,
)
from afabench.compute_estimate.report import (
    UnknownGroupingColumnError,
    format_report,
    group_totals,
    per_job_table,
)
from afabench.core.job_duration_table import load_job_duration_table
from afabench.core.job_record import ExitStatus, JobIdentity, JobType
from afabench.pipeline_plan.planning import PlannedJob
from test.job_record_examples import job_record, write_job_record

ALPHA_TRAINING = JobIdentity(
    stage="training",
    name="alpha",
    dataset_key="cube",
    dataset_realization_index=0,
    train_seed=0,
    train_hard_budget=1,
)


class JobRecords:
    """An output root to write job records with chosen job durations to."""

    output_root: Path
    count: int

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        output_root.mkdir()
        self.count = 0

    def add(
        self,
        identity: JobIdentity,
        job_duration_seconds: float | None,
        *,
        device: str = "cuda",
        cpus: int | None = 4,
        gpus: int = 1,
        exit_status: ExitStatus = "completed",
        time_limit_minutes: int | None = 600,
        slurm_cluster: str | None = "alvis",
        smoke_test: bool = False,
    ) -> None:
        self.count += 1
        write_job_record(
            self.output_root / f"job{self.count}.job_record.json",
            job_record(
                identity,
                started_at="2026-10-08T12:00:00+00:00",
                ended_at="2026-10-08T13:00:00+00:00",
                job_duration_seconds=job_duration_seconds,
                exit_status=exit_status,
                exit_code=0 if exit_status == "completed" else 1,
                device="cuda" if device == "cuda" else "cpu",
                cpus=cpus,
                gpus=gpus,
                time_limit_minutes=time_limit_minutes,
                gpu_model="NVIDIA A40" if gpus else None,
                cpu_model="AMD EPYC 7742",
                host=f"node{self.count % 2}",
                slurm_job_id=str(self.count),
                slurm_cluster=slurm_cluster,
                code_commit="0123abc",
                smoke_test=smoke_test,
            ),
        )

    def table(self) -> pl.DataFrame:
        return load_job_duration_table(self.output_root)


@pytest.fixture
def records(tmp_path: Path) -> JobRecords:
    return JobRecords(tmp_path / "output")


JOBIDS = count()


def planned(
    identity: JobIdentity | None,
    *,
    rule: str = "train_method",
    device: str = "cuda",
    cpus: int | None = 8,
    gpus: int = 1,
    after: Iterable[PlannedJob] = (),
) -> PlannedJob:
    return PlannedJob(
        rule=rule,
        wildcards={},
        identity=identity,
        device="cuda" if device == "cuda" else "cpu",
        cpus=cpus,
        gpus=gpus,
        jobid=next(JOBIDS),
        status="run",
        dependencies=frozenset(job.jobid for job in after),
    )


def test_a_job_with_a_record_of_the_same_identity_and_device_matches_exactly(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 7200)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 3600)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "exact"
    assert job.matched_job_records == 1
    assert job.mean_job_duration_seconds == 7200
    assert job.p90_job_duration_seconds == 7200


def test_a_job_without_an_exact_match_pools_its_stage_name_dataset_and_device(
    records: JobRecords,
) -> None:
    # Other seeds, dataset realizations, hard budgets and soft-budget
    # parameters of the same method on the same dataset and device
    for seconds, seed in zip(range(10, 101, 10), range(1, 11), strict=True):
        records.add(
            replace(
                ALPHA_TRAINING,
                train_seed=seed,
                dataset_realization_index=seed % 3,
                train_hard_budget=None,
                train_soft_budget_param=seed / 10,
            ),
            seconds,
        )

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "pooled"
    assert job.matched_job_records == 10
    assert job.mean_job_duration_seconds == 55
    # Linear interpolation between the 9th and 10th of 10 durations
    assert job.p90_job_duration_seconds == pytest.approx(91)


@pytest.mark.parametrize(
    ("identity", "device", "gpus"),
    [
        (ALPHA_TRAINING, "cpu", 0),
        (replace(ALPHA_TRAINING, dataset_key="mnist"), "cuda", 1),
    ],
    ids=["other device", "other dataset key"],
)
def test_matching_never_crosses_device_or_dataset_key(
    records: JobRecords, identity: JobIdentity, device: str, gpus: int
) -> None:
    records.add(identity, 3600, device=device, gpus=gpus)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "unestimated"
    assert job.matched_job_records == 0
    assert job.mean_job_duration_seconds is None
    assert job.p90_job_duration_seconds is None


@pytest.mark.parametrize("exit_status", ["failed", "timeout"])
def test_failed_and_timed_out_jobs_give_no_job_durations(
    records: JobRecords, exit_status: ExitStatus
) -> None:
    records.add(ALPHA_TRAINING, 60, exit_status=exit_status)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 3600)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "pooled"
    assert job.mean_job_duration_seconds == 3600


def test_failed_and_timed_out_jobs_warn_once_per_stage_name_and_dataset_key(
    records: JobRecords,
) -> None:
    # Two seeds of alpha timed out at two time limits and one crashed;
    # beta timed out on a CPU, and in a smoke test, which is refused; gamma
    # is not planned.
    records.add(ALPHA_TRAINING, 36000, exit_status="timeout")
    records.add(
        replace(ALPHA_TRAINING, train_seed=1),
        18000,
        exit_status="timeout",
        time_limit_minutes=300,
    )
    records.add(
        replace(ALPHA_TRAINING, train_seed=1), 36000, exit_status="timeout"
    )
    records.add(replace(ALPHA_TRAINING, train_seed=2), 5, exit_status="failed")
    records.add(replace(ALPHA_TRAINING, train_seed=3), 3600)
    beta = replace(ALPHA_TRAINING, name="beta")
    records.add(beta, 600, device="cpu", gpus=0, exit_status="timeout")
    records.add(beta, 1, exit_status="timeout", smoke_test=True)
    gamma = replace(ALPHA_TRAINING, name="gamma")
    records.add(gamma, 600, exit_status="failed")

    estimate = estimate_compute(
        [
            planned(ALPHA_TRAINING),
            planned(replace(ALPHA_TRAINING, train_seed=4)),
            planned(beta),
        ],
        records.table(),
    )

    assert estimate.failure_histories == [
        FailureHistory(
            job_type=JobType(
                stage="training", name="alpha", dataset_key="cube"
            ),
            failed=1,
            timed_out=3,
            time_limits_minutes=[300, 600],
        ),
        FailureHistory(
            job_type=JobType(
                stage="training", name="beta", dataset_key="cube"
            ),
            failed=0,
            timed_out=1,
            time_limits_minutes=[600],
        ),
    ]


def test_smoke_test_job_durations_are_refused(records: JobRecords) -> None:
    records.add(ALPHA_TRAINING, 1, smoke_test=True)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 2, smoke_test=True)

    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    assert [job.match_level for job in estimate.jobs] == ["unestimated"]
    assert estimate.refused_smoke_test_job_records == 2


def test_core_and_gpu_hours_follow_the_planned_allocation_not_the_measured(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600, cpus=4, gpus=1)
    records.add(ALPHA_TRAINING, 10800, cpus=4, gpus=1)

    (job,) = estimate_compute(
        [planned(ALPHA_TRAINING, cpus=8, gpus=2)], records.table()
    ).jobs

    assert (job.mean_job_hours, job.p90_job_hours) == (2, pytest.approx(2.8))
    assert (job.mean_core_hours, job.p90_core_hours) == (
        16,
        pytest.approx(22.4),
    )
    assert (job.mean_gpu_hours, job.p90_gpu_hours) == (4, pytest.approx(5.6))


def test_core_hours_are_unknown_when_the_cluster_chooses_the_cpus(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)

    (job,) = estimate_compute(
        [planned(ALPHA_TRAINING, cpus=None, gpus=1)], records.table()
    ).jobs

    assert job.mean_core_hours is None
    assert job.p90_core_hours is None
    assert job.mean_gpu_hours == 1


def test_jobs_of_rules_without_job_records_are_not_unestimated(
    records: JobRecords,
) -> None:
    aggregation = planned(None, rule="merge_eval_perf", device="cpu", gpus=0)

    (job,) = estimate_compute([aggregation], records.table()).jobs

    assert job.match_level == "no_job_record"
    assert job.mean_job_duration_seconds is None


def two_methods_on_two_datasets(records: JobRecords) -> list[PlannedJob]:
    """Measure alpha and beta training on cube, and plan both on two."""
    records.add(ALPHA_TRAINING, 3600)
    records.add(replace(ALPHA_TRAINING, name="beta"), 7200, device="cpu")
    jobs = []
    for dataset_key in ["cube", "mnist"]:
        jobs.append(
            planned(replace(ALPHA_TRAINING, dataset_key=dataset_key), cpus=2)
        )
        jobs.append(
            planned(
                replace(ALPHA_TRAINING, name="beta", dataset_key=dataset_key),
                device="cpu",
                cpus=4,
                gpus=0,
            )
        )
    jobs.append(planned(None, rule="merge_eval_perf", device="cpu", gpus=0))
    return jobs


def test_totals_group_by_stage_and_device_by_default(
    records: JobRecords,
) -> None:
    estimate = estimate_compute(
        two_methods_on_two_datasets(records), records.table()
    )

    totals = group_totals(estimate)

    assert totals.to_dicts() == [
        {
            "stage": "training",
            "device": "cpu",
            "jobs": 2,
            "exact": 1,
            "pooled": 0,
            "unestimated": 1,
            "mean_job_hours": 2.0,
            "p90_job_hours": 2.0,
            "mean_core_hours": 8.0,
            "p90_core_hours": 8.0,
            "mean_gpu_hours": 0.0,
            "p90_gpu_hours": 0.0,
        },
        {
            "stage": "training",
            "device": "cuda",
            "jobs": 2,
            "exact": 1,
            "pooled": 0,
            "unestimated": 1,
            "mean_job_hours": 1.0,
            "p90_job_hours": 1.0,
            "mean_core_hours": 2.0,
            "p90_core_hours": 2.0,
            "mean_gpu_hours": 1.0,
            "p90_gpu_hours": 1.0,
        },
    ]


def test_totals_regroup_by_other_columns(records: JobRecords) -> None:
    estimate = estimate_compute(
        two_methods_on_two_datasets(records), records.table()
    )

    by_name = group_totals(estimate, ["name"])
    by_dataset = group_totals(estimate, ["dataset_key"])

    assert by_name.select("name", "jobs", "mean_core_hours").to_dicts() == [
        {"name": "alpha", "jobs": 2, "mean_core_hours": 2.0},
        {"name": "beta", "jobs": 2, "mean_core_hours": 8.0},
    ]
    assert by_dataset.select(
        "dataset_key", "exact", "unestimated"
    ).to_dicts() == [
        {"dataset_key": "cube", "exact": 2, "unestimated": 0},
        {"dataset_key": "mnist", "exact": 0, "unestimated": 2},
    ]
    # Unestimated jobs are not in the totals
    assert by_dataset[0, "mean_job_hours"] == 3.0
    assert by_dataset[1, "mean_job_hours"] is None


def test_totals_refuse_an_unknown_grouping_column(
    records: JobRecords,
) -> None:
    estimate = estimate_compute([], records.table())

    with pytest.raises(UnknownGroupingColumnError, match="'method'"):
        group_totals(estimate, ["method"])


def test_the_report_names_the_site_and_hardware_of_the_matched_job_records(
    records: JobRecords,
) -> None:
    records.add(
        replace(ALPHA_TRAINING, name="gamma"),
        60,
        gpus=0,
        slurm_cluster="vera",
    )  # node1
    records.add(ALPHA_TRAINING, 3600, slurm_cluster="alvis")  # node0
    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    report = format_report(estimate, source="v1/job_durations.parquet")

    (source,) = [
        line for line in report.splitlines() if "v1/job_durations" in line
    ]
    assert "1 job record" in source
    assert "SLURM clusters: alvis;" in source
    assert "vera" not in source
    assert "node0" in source
    assert "node1" not in source
    assert "AMD EPYC 7742" in source
    assert "NVIDIA A40" in source


def test_the_report_lists_unestimated_jobs_apart_from_the_totals(
    records: JobRecords,
) -> None:
    estimate = estimate_compute(
        two_methods_on_two_datasets(records), records.table()
    )

    report = format_report(estimate, source="output")

    assert "5 planned jobs: 2 exact, 0 pooled, 2 unestimated" in report
    totals, unestimated = report.split("Unestimated jobs")
    assert "mnist" not in totals
    assert "mnist" in unestimated
    assert "merge_eval_perf" in unestimated


def test_the_report_counts_refused_smoke_test_job_records(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 1, smoke_test=True)
    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    report = format_report(estimate, source="output")

    assert "Refused 1 smoke-test job record" in report


def test_the_report_warns_that_job_types_that_failed_before_may_be_low(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 36000, exit_status="timeout")
    records.add(ALPHA_TRAINING, 18000, exit_status="timeout")
    records.add(
        ALPHA_TRAINING, 3600, exit_status="timeout", time_limit_minutes=60
    )
    records.add(ALPHA_TRAINING, 5, exit_status="failed")
    beta = replace(ALPHA_TRAINING, name="beta")
    records.add(beta, 5, exit_status="failed")
    records.add(beta, 5, exit_status="failed")
    gamma = replace(ALPHA_TRAINING, name="gamma")
    records.add(gamma, 60, exit_status="timeout", time_limit_minutes=None)
    estimate = estimate_compute(
        [planned(ALPHA_TRAINING), planned(beta), planned(gamma)],
        records.table(),
    )

    report = format_report(estimate, source="output")

    _, warnings = report.split("Failed or timed out before")
    assert "may be low" in warnings.splitlines()[0]
    assert warnings.splitlines()[1:4] == [
        "  training alpha on cube timed out 3 times at 60 and 600 min, "
        "failed 1 time",
        "  training beta on cube failed 2 times",
        "  training gamma on cube timed out 1 time at an unknown time limit",
    ]


def test_the_report_has_no_warnings_without_failed_or_timed_out_jobs(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)
    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    report = format_report(estimate, source="output")

    assert "Failed or timed out before" not in report


def test_the_per_job_table_flags_jobs_whose_type_failed_before(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 5, exit_status="failed")
    beta = replace(ALPHA_TRAINING, name="beta")
    records.add(beta, 3600)
    estimate = estimate_compute(
        [
            planned(replace(ALPHA_TRAINING, train_seed=1)),
            planned(beta),
            planned(None, rule="merge_eval_perf", device="cpu", gpus=0),
        ],
        records.table(),
    )

    jobs = per_job_table(estimate)

    assert jobs["failure_history"].to_list() == [True, False, False]


ALPHA_EVALUATION = replace(ALPHA_TRAINING, stage="evaluation")
BETA_TRAINING = replace(ALPHA_TRAINING, name="beta")


def test_minimal_wall_time_is_the_longest_chain_of_dependent_jobs(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)
    records.add(ALPHA_EVALUATION, 1800)
    records.add(BETA_TRAINING, 7200)
    alpha = planned(ALPHA_TRAINING)
    evaluation = planned(ALPHA_EVALUATION, after=[alpha])
    beta = planned(BETA_TRAINING)
    # Independent jobs run in parallel, whichever way they are listed.
    estimate = estimate_compute(
        [evaluation, beta, alpha, planned(ALPHA_TRAINING)], records.table()
    )

    wall_time = minimal_wall_time(estimate, "mean")

    assert wall_time.seconds == 7200
    assert [step.job for step in wall_time.chain] == [beta]
    records.add(ALPHA_EVALUATION, 1800 + 7200)
    longer = minimal_wall_time(
        estimate_compute([evaluation, beta, alpha], records.table()), "mean"
    )
    assert longer.seconds == 3600 + (1800 + 9000) / 2
    assert [step.job for step in longer.chain] == [alpha, evaluation]


def test_minimal_wall_time_at_p90_takes_each_job_s_p90(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)
    records.add(ALPHA_TRAINING, 7200)
    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    assert minimal_wall_time(estimate, "mean").seconds == 5400
    assert minimal_wall_time(estimate, "p90").seconds == 6840


def test_jobs_without_a_job_duration_take_no_time_but_keep_their_chain(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)
    records.add(ALPHA_EVALUATION, 1800)
    unestimated = planned(BETA_TRAINING)
    alpha = planned(ALPHA_TRAINING, after=[unestimated])
    merge = planned(None, rule="merge_eval_perf", gpus=0, after=[alpha])
    evaluation = planned(ALPHA_EVALUATION, after=[merge])
    estimate = estimate_compute(
        [unestimated, alpha, merge, evaluation], records.table()
    )

    wall_time = minimal_wall_time(estimate, "mean")

    assert wall_time.seconds == 3600 + 1800
    assert [step.job for step in wall_time.chain] == [
        unestimated,
        alpha,
        merge,
        evaluation,
    ]


def test_the_report_gives_the_minimal_wall_time_and_its_chain(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)
    records.add(ALPHA_TRAINING, 7200)
    alpha = planned(ALPHA_TRAINING)
    unestimated = planned(BETA_TRAINING, after=[alpha])
    merge = planned(None, rule="merge_eval_perf", gpus=0, after=[unestimated])
    estimate = estimate_compute([alpha, unestimated, merge], records.table())

    report = format_report(estimate, source="output")

    _, wall_time = report.split("Minimal wall time")
    assert wall_time.splitlines()[:5] == [
        ", every job starting once the jobs it needs are done: "
        "mean 1.50 h, p90 1.90 h",
        "  It counts unestimated jobs and jobs of rules without job records "
        "as taking no time.",
        "  Longest chain at mean job durations:",
        "    1.50 h  training alpha on cube",
        "       -    training beta on cube (unestimated)",
    ]
    assert wall_time.splitlines()[5] == (
        "       -    merge_eval_perf (no job record)"
    )


def test_the_report_has_no_minimal_wall_time_without_estimated_jobs(
    records: JobRecords,
) -> None:
    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    report = format_report(estimate, source="output")

    assert "Minimal wall time: unknown, no planned job is estimated" in report


def test_the_per_job_table_holds_each_job_s_dependencies(
    records: JobRecords,
) -> None:
    alpha = planned(ALPHA_TRAINING)
    beta = planned(BETA_TRAINING)
    evaluation = planned(ALPHA_EVALUATION, after=[beta, alpha])
    estimate = estimate_compute([alpha, beta, evaluation], records.table())

    jobs = per_job_table(estimate)

    assert jobs["jobid"].to_list() == [
        alpha.jobid,
        beta.jobid,
        evaluation.jobid,
    ]
    assert jobs["dependencies"].to_list() == [
        "[]",
        "[]",
        f"[{alpha.jobid}, {beta.jobid}]",
    ]


def test_empty_estimate_has_typed_tables_and_unknown_totals(
    records: JobRecords,
) -> None:
    estimate = estimate_compute([], records.table())

    jobs = per_job_table(estimate)
    totals = group_totals(estimate)
    report = format_report(estimate, source="empty")

    assert isinstance(jobs, pl.DataFrame)
    assert jobs.is_empty()
    assert jobs.schema["train_seed"] == pl.Int64
    assert jobs.schema["mean_job_hours"] == pl.Float64
    assert totals.is_empty()
    assert "0 planned jobs: 0 exact, 0 pooled, 0 unestimated" in report
    assert "Minimal wall time: unknown" in report


def test_totals_keep_null_groups_and_all_unknown_hours(
    records: JobRecords,
) -> None:
    generation = JobIdentity(stage="dataset_generation", dataset_key="cube")
    records.add(generation, 3600, device="cpu", cpus=None, gpus=0)
    estimate = estimate_compute(
        [
            planned(generation, device="cpu", cpus=None, gpus=0),
            planned(replace(ALPHA_TRAINING, dataset_key="mnist")),
        ],
        records.table(),
    )

    totals = group_totals(estimate, ["name"])

    assert totals.select(
        "name", "jobs", "mean_job_hours", "mean_core_hours"
    ).to_dicts() == [
        {
            "name": "alpha",
            "jobs": 1,
            "mean_job_hours": None,
            "mean_core_hours": None,
        },
        {
            "name": None,
            "jobs": 1,
            "mean_job_hours": 1.0,
            "mean_core_hours": None,
        },
    ]


def test_matched_completed_unknown_duration_stays_unknown_in_summary_and_report(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, None)

    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    (job,) = estimate.jobs
    assert job.match_level == "exact"
    assert job.matched_job_records == 1
    assert job.mean_job_duration_seconds is None
    assert job.p90_job_duration_seconds is None
    assert (
        per_job_table(estimate)
        .select(
            "mean_job_duration_seconds",
            "p90_job_duration_seconds",
            "mean_job_hours",
            "p90_job_hours",
            "mean_core_hours",
            "p90_core_hours",
            "mean_gpu_hours",
            "p90_gpu_hours",
        )
        .row(0)
        == (None,) * 8
    )
    assert group_totals(estimate)[0, "mean_job_hours"] is None
    report = format_report(estimate, source="unknown")
    assert "1 planned job: 1 exact, 0 pooled, 0 unestimated" in report
    assert "Minimal wall time: unknown, no planned job is estimated" in report
    total = next(
        line
        for line in report.splitlines()
        if line.strip().startswith("total")
    )
    assert total.split() == ["total", "1", "1", "0", "0", *["-"] * 6]
