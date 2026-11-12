# `estimate-compute` command

`uv run workflow/run.py -e` or `uv run just estimate-compute`
(`scripts/compute_estimate/estimate_compute.py`) prints the [compute estimate](../../CONTEXT.md) of a pipeline invocation
([ADR 0007](../adr/0007-job-records-beside-artifacts.md)). Steps:
[estimate the compute of a run](../how-to/estimate_compute.md).

```shell
uv run workflow/run.py -e [--job-durations PATH] [--by COLUMN ...]
    [--output CSV] [--strict] <launcher arguments>
uv run just estimate-compute [--job-durations PATH] [--by COLUMN ...]
    [--output CSV] [--strict] <launcher arguments>
```

## Options

Every argument other than these four is an argument of the
[pipeline launcher](pipeline_launcher_command.md), so the planned jobs, and
the CPUs and GPUs each would request, are those of
`uv run workflow/run.py <launcher arguments>`. Nothing is submitted or run.
Profiles are looked up in the working directory.

| Option | Meaning |
| --- | --- |
| `--job-durations PATH` | A [job duration table](job_records.md#job-duration-table) Parquet file, such as a downloaded release's `output/production/release_job_duration_table.parquet`, or an output root whose job records are read. Default `output/production`, the production output root, which holds no smoke test's job records; if it does not exist, every job is unestimated. Any other missing path is an error. |
| `--by COLUMN` | Group the totals by this job column instead of `stage` and `device`; repeat for several. Any column of the [per-job CSV](#per-job-csv) up to `time_limit_minutes`, except `wildcards`. |
| `--output CSV` | Also write the [per-job CSV](#per-job-csv). |
| `--strict` | Exit with 1 after the report when any planned job is unestimated. |

## Matching

Only completed job records give job durations: failed and timed-out
attempts do not, and smoke-test records are refused and counted in the
report. Failed and timed-out records give [failure warnings](#report)
instead. Each planned job's match level is the first that applies:

| Match level | Job durations used |
| --- | --- |
| `exact` | Records with the same identity (every identity [field](job_records.md#identity)) and device. |
| `pooled` | Records with the same stage, name, dataset key and device: any seeds, dataset realizations, hard budgets, soft-budget parameters and evaluation batch size. |
| `unestimated` | None; the job is not in the totals. |
| `no_job_record` | None; the job's rule writes no job record (aggregation and visualization). Not in the totals, and `--strict` ignores it. |

A job's estimate is the mean and the p90 (linear interpolation) of its
matched durations. Core-hours multiply them by the CPUs the job would
request and GPU-hours by its GPUs, not by the allocation the measured jobs
had. A job whose CPUs are left to the cluster has no core-hours. Durations
are never scaled across hardware.

## Report

Printed to standard output, in this order:

1. The number of planned jobs per match level.
2. The source of the job durations, and the distinct SLURM clusters
   (sites), hosts, CPU models and GPU models of the matched job records.
   For a `release_job_duration_table.parquet` with a release manifest
   beside it, the source names the release id, or says that the
   table is not that release's when its size differs from the manifest's
   `job_duration_table` entry.
3. The number of refused smoke-test job records, if any.
4. The totals per group, and a `total` row: `jobs` and the jobs per match
   level, then `mean_` and `p90_` `job_hours`, `core_hours` and
   `gpu_hours` of the estimated jobs. A p90 total adds up each job's p90.
5. The [minimal wall time](#minimal-wall-time), mean and p90, and the
   longest chain of jobs at mean job durations, each with its hours.
6. Unestimated jobs, counted per stage, name, dataset key and device.
7. Failure warnings, if any: one line per stage, name and dataset key of
   a planned job with `failed` or `timeout` job records (smoke tests
   excluded), on any device, with the number of each and the distinct
   `time_limit_minutes` of the timed-out ones ("an unknown time limit"
   when none was recorded). The estimate of these jobs may be low. A
   warning stays while those job records are in the source, also after a
   later attempt completed.
8. Jobs of rules without job records, counted per rule.
9. How many estimated jobs' core-hours are unknown, if any.

## Minimal wall time

How long the planned jobs take if each starts as soon as the planned jobs it
depends on are done, with no limit on how many run at once: the longest
chain of dependent jobs, each taking its mean (or p90) job duration. Jobs
whose existing outputs the run reuses are not scheduled, so they add
nothing. Unestimated jobs
and jobs of rules without job records take no time, but still link the jobs
before them to those after them, so the wall time is too low when the
chain holds unestimated jobs. Queueing, the site's job limit and the
cluster's free nodes are left out: the real wall time is never shorter.

## Per-job CSV

One row per planned job, sorted by rule and wildcards. Empty cells are
null.

| Column | Meaning |
| --- | --- |
| `rule` | Snakemake rule. |
| `wildcards` | The job's wildcards as a JSON object. |
| `stage` ... `eval_batch_size` | The identity the job's record will have, as in [job records](job_records.md#identity); empty for `no_job_record` jobs. |
| `device`, `cpus`, `gpus` | The planned allocation; `cpus` is empty when the cluster chooses. |
| `time_limit_minutes` | Resolved requested walltime, including [site lookup](requested_walltime.md) and resource overrides; empty when unspecified or not yet resolvable. Not the measured duration used for estimates. |
| `match_level` | See [matching](#matching). |
| `matched_job_records` | How many job durations the estimate uses. |
| `mean_job_duration_seconds`, `p90_job_duration_seconds` | The job's estimated job duration. |
| `mean_job_hours`, `p90_job_hours`, `mean_core_hours`, `p90_core_hours`, `mean_gpu_hours`, `p90_gpu_hours` | The same in hours, times 1, the planned CPUs and the planned GPUs. |
| `failure_history` | `True` when the report has a failure warning for the job's stage, name and dataset key. |
| `jobid` | Snakemake's id of the job in this plan. |
| `dependencies` | The `jobid`s of the planned jobs it needs, as a JSON list. |
