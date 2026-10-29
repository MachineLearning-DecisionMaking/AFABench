# Job records

A [job record](../../CONTEXT.md) is a JSON file that one computational
pipeline job writes beside the artifact it produced. It holds the job's
identity, its job duration and the allocation it ran with
([ADR 0007](../adr/0007-job-records-beside-artifacts.md)). It describes the
run, not the artifact, so it is not part of the artifact's provenance record.
The schema is `afabench.core.job_record.JobRecord`, version 1. The
[job duration table](#job-duration-table) collects every record of an output
root.

## Where records are

Each record is a declared output of its rule, named after the artifact with
`.job_record.json` in place of the artifact's suffix. Paths are relative to
the output root: `output/production/`, or `output/smoke/` for a smoke test
(see [`output_root`](pipeline_configuration.md#output_root)); `<tag>` is `initializer-<initializer>`,
`<training>` the training run's folder
(`<method>/dataset-<key>+realization_index-<k>/<pretrain folder>/train_seed-<s>+train_hard_budget-<b>+train_soft_budget_param-<p>`)
and `<evaluation>` the evaluation's subfolder
(`eval_seed-<s>+eval_hard_budget-<b>+eval_soft_budget_param-<p>`).

| Pipeline stage | Rule | Record |
| --- | --- | --- |
| `dataset_generation` | `dataset_generation` | `datasets/<key>/<k>/dataset_generation.job_record.json`, in the dataset realization's folder, beside the bundles of its splits |
| `classifier_training` | `train_classifier`, `train_classifier_for_method` | `trained_classifiers/<tag>/[method-<method>+]dataset-<key>+realization_index-<k>.job_record.json` |
| `pretraining` | `pretrain_model` | `pretrained_models/<tag>/<pretrained model>/dataset-<key>+realization_index-<k>/pretrain_seed-<s>/model.job_record.json` |
| `training` | `train_method` | `trained_methods/<tag>/<training>/method.job_record.json` |
| `evaluation` | `eval_method` | `eval_results/eval_split-<split>/<tag>/<training>/<evaluation>/eval_data.job_record.json` |
| `transformation` | `transform_eval_data` | `eval_results_transformed/eval_split-<split>/<tag>/<training>/<evaluation>/eval_data.job_record.json` |

No job depends on a record, so a missing record never makes Snakemake rerun
a job whose artifact exists, for example a downloaded shared prerequisite.

### Failed and timed-out jobs

Snakemake deletes a failed job's declared outputs, so a job whose
`exit_status` is `failed` or `timeout` writes its record to the
failed-records directory `failed_job_records/` instead. No rule declares
it. The record's path there is the path in the table above, with an
attempt id inserted before `.job_record.json`: the UTC start time and a
random suffix. Each attempt of a job leaves its own record, for example

```text
failed_job_records/trained_methods/<tag>/<training>/method.20261008T120000123456Z-1a2b3c4d.job_record.json
```

Snakemake does not delete these records, and a later successful attempt
does not remove them.

## How a record is written

A rule runs its script through the wrapper command, which runs the script,
times it and writes the record:

```shell
python -m afabench.core.job_record --record <path> \
    --failed-record <path> --stage <stage> --device <cpu|cuda> \
    [--cpus <n>] [--gpus <n>] [--time-limit-minutes <n>] [--smoke-test] \
    [identity options] -- <script command>
```

`--record` is where a completed job's record goes, `--failed-record` the
path a failed or timed-out job's record is named after. Each identity field
below has an option of the same name with `-` for `_` (`--dataset-key`); an
omitted option records `null`. `workflow/src/job_records.py`
renders the options from the job's wildcards and its final Snakemake
resources, after profile defaults and `--set-resources` overrides.

The wrapper exits with the script's exit code, or 128 plus the number of
the signal that ended the script, so Snakemake sees a failed script as a
failed job. On SIGTERM, which SLURM sends at a job's time limit, it passes
the signal on to the script, kills the script if it has not exited after 10
seconds, records `timeout` and exits with 143 (128 plus SIGTERM). SLURM
kills the wrapper itself after its `KillWait`, 30 seconds by default, so a
cluster with a `KillWait` under 10 seconds can lose a timed-out job's
record.

## Fields

`null` means unknown, or not part of this job's identity, never a default.
Records are flat, so that one record is one row of a table.

| Field | Meaning |
| --- | --- |
| `job_record_version` | Schema version, 1. |

### Identity

| Field | Meaning |
| --- | --- |
| `stage` | Pipeline stage: `dataset_generation`, `classifier_training`, `pretraining`, `training`, `evaluation` or `transformation`. |
| `name` | Method name for training, evaluation and transformation; pretrained-model name for pretraining; the classifier's script name for classifier training; null for dataset generation. |
| `dataset_key` | Dataset key. |
| `dataset_realization_index` | Dataset realization index. |
| `pretrain_seed` | Seed of the pretraining run the job is part of or depends on; null without pretraining. |
| `train_seed` | Seed of the training run; for classifier training, the classifier's seed, which is its dataset realization index. |
| `eval_seed` | Seed of the evaluation. |
| `train_hard_budget`, `eval_hard_budget` | Hard budget (integer) of training and of evaluation; null in the soft-budget setting. |
| `train_soft_budget_param`, `eval_soft_budget_param` | Soft-budget parameter (number) of training and of evaluation; null in the hard-budget setting. |
| `eval_batch_size` | Evaluation batch size; evaluation only. |

### Timing

| Field | Meaning |
| --- | --- |
| `started_at`, `ended_at` | UTC ISO-8601 times the script started and ended. |
| `job_duration_seconds` | Job duration: the script's wall-clock time in seconds, measured with a monotonic clock. |
| `exit_status` | `completed` if the script exited with 0; `timeout` if the wrapper received SIGTERM, which SLURM sends at the time limit, but which `scancel` also sends; `failed` otherwise. Only `completed` records sit beside artifacts; the others are in the failed-records directory. |
| `exit_code` | The script's exit code; negative if a signal ended it, `-15` for SIGTERM. |

### Allocation

| Field | Meaning |
| --- | --- |
| `device` | `cpu` or `cuda`, the device the pipeline resolved for the job. |
| `cpus` | CPUs requested: the `cpus_per_task` resource, or the job's threads without one, as the SLURM executor requests them; null if `cpus_per_task` is negative, which leaves it to the cluster. |
| `gpus` | GPUs requested: the `gpu` resource, or the count of a `gpu[:<model>]:<n>` `gres`; 0 for none. |
| `time_limit_minutes` | The job's time limit in minutes: its `runtime` resource, which the SLURM executor requests with `-t`; null if no profile sets one. |
| `gpu_model` | Names of the GPUs `nvidia-smi` lists in the job, comma-separated; null for jobs without GPUs or where it cannot be queried. |
| `cpu_model` | The CPU's model name from `/proc/cpuinfo`, or the platform's processor name. |
| `host` | Host name the job ran on. |
| `slurm_job_id` | `SLURM_JOB_ID` of the job; null outside SLURM. |
| `slurm_cluster` | `SLURM_CLUSTER_NAME` of the job, the site it ran at; null outside SLURM. |

### Code and mode

| Field | Meaning |
| --- | --- |
| `code_commit` | Commit of the afabench checkout that ran the job; null outside a git work tree. |
| `smoke_test` | Whether the pipeline ran with `smoke_test=true`. |

## Job duration table

The [job duration table](../../CONTEXT.md) holds every job record under an
output root, one row per record with no aggregation: the completed records
beside artifacts and every failed or timed-out attempt in
`failed_job_records/`. The `collect_job_records` rule writes it to
`merged_results/job_duration_table.parquet` with
`scripts/misc/collect_job_records.py`, as part of the `all` target. It runs
after the selected methods' transformations, so after every job of the run,
and on CPU under the site's CPU allocation, like the other aggregation
rules. It reads the output root, not its inputs, so the table also holds
records of earlier invocations, for example of other methods, initializers
or evaluation splits. No record is an input, so a missing record does not
rerun its job. A benchmark release ships its own copy, beside its manifest
rather than in `merged_results/`
([payload category](release_manifest.md#job-duration-table)).

The `plot_time` rule plots the table with
`scripts/plotting/plot_total_time.py`, which calls
`afabench.plotting.total_time`. It keeps the completed jobs of the
selected methods under the run's initializer and evaluation split, so
failed and timed-out attempts are left out: each method's pretraining,
through the pretrained model it trains from, its method-specific
classifier training, training, evaluation and transformation.

`afabench.core.job_duration_table.load_job_duration_table(source)` loads
either the table's Parquet file or an output root of loose job records,
and returns the same Polars DataFrame for both, sorted by
`job_record_path`. The
[`estimate-compute` command](estimate_compute_command.md) loads its job
durations with it. It raises `UnknownJobRecordVersionError` for a job record
version it does not know and `JobRecordFieldsError` for a record or table
whose fields do not match its version.

The first column is `job_record_path`, the path of the row's record relative
to the output root. Records do not name the initializer or evaluation split,
and repeated attempts of a job have the same identity, so the path is what
tells such rows apart. One column per [field](#fields) follows, named and
ordered as there. Every column is nullable; unknown values are `null`
(Python `None`).

| Columns | Type |
| --- | --- |
| `job_record_path`, `stage`, `name`, `dataset_key`, `exit_status`, `device`, `gpu_model`, `cpu_model`, `host`, `slurm_job_id`, `slurm_cluster`, `code_commit` | `String` |
| `job_record_version`, `dataset_realization_index`, `pretrain_seed`, `train_seed`, `eval_seed`, `train_hard_budget`, `eval_hard_budget`, `eval_batch_size`, `exit_code`, `cpus`, `gpus`, `time_limit_minutes` | `Int64` |
| `train_soft_budget_param`, `eval_soft_budget_param`, `job_duration_seconds` | `Float64` |
| `started_at`, `ended_at` | `Datetime(time_unit="us", time_zone="UTC")` |
| `smoke_test` | `Boolean` |
