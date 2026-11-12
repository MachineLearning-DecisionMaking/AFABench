# Pipeline stages and configuration layers

AFABench uses Snakemake to turn a benchmark configuration into dependent
jobs. The launcher, `workflow/run.py`, selects the configuration, previews
the work and starts execution.

## What runs

```text
dataset generation
    ├── classifier training
    └── pretraining (when needed)
             ↓
       method training
             ↓
         evaluation
             ↓
 transformation → aggregation → plots
```

Dependencies vary by method: some need a pretrained model, some train a
built-in classifier, and others need neither. External classifiers and
named pretrained models can be shared across methods. Evaluation applies
the shared acquisition protocol and writes raw acquisition histories;
transformation creates plotting-ready tables, and aggregation and plotting
produce comparisons.

The run expands across selected datasets, dataset realizations, methods
and budget settings. A realization index seeds dataset generation and
training. Hard-budget and soft-budget results are separate comparisons.
Snakemake schedules jobs as their prerequisites become available and reuses
completed outputs when its dependency checks permit it.

A target such as `all_eval_methods` asks for a stage and its prerequisites.
Disabling a stage with `stages` instead requires its outputs to exist and
prevents that stage from running. See [running selected stages](../how-to/run_pipeline.md#run-a-specific-stage).

## Which files control what

| Layer | Location | Controls |
| --- | --- | --- |
| Pipeline profile | `workflow/profiles/pipeline/<name>/config.yaml` | Which workflow and configuration variants to load |
| Benchmark settings | `workflow/conf/` | Datasets, methods, realizations, budgets, classifiers, Unmaskers and plotting groups |
| Execution file | `workflow/profiles/execution/` | CPU/GPU choices by stage, method or pretrained model |
| Site profile | `workflow/profiles/site/<name>/` | SLURM partitions, accounts, resources and environments |
| Script configuration | `conf/scripts/` and `conf/components/` | Individual jobs' Hydra configs, model hyperparameters and shared components |

Pipeline profiles answer **what to run**, execution files **which device to
use**, and site profiles **where to run**. These choices are independent:
the same benchmark settings can run locally or on different clusters.

The launcher accepts `key=value` overrides for one-off changes. For a
repeatable experiment, copy a pipeline profile and the configuration files
you want to change. Script-level Hydra configuration is a separate layer:
the workflow supplies job-specific arguments such as dataset paths, seeds,
budgets and devices to scripts.

## Acquisition configuration

A method chooses an action: `0` stops; action `i + 1` selects selection `i`.
The Unmasker maps that selection to features, and the initializer sets the
starting feature mask. With direct unmasking, one selection reveals one
feature. With image patches, one selection can reveal many pixels: a
28×28 image split into 4×4 patches has 49 selections, not 784.
Budgets count selection cost, not necessarily feature count.

The external classifier evaluates information acquired by different methods
on a shared basis. Built-in classifier results are reported separately.
The [configuration reference](../reference/pipeline_configuration.md) describes
classifier mappings, Unmaskers, training-budget mappings and soft-budget parameters.

## Outputs and reuse

Production outputs default to `output/production/`; smoke tests use
`output/smoke/` and cannot satisfy production jobs. Outputs include dataset
and model bundles, raw and transformed evaluation tables, merged results,
plots and job records.

A [release](../how-to/manage_releases.md) provides existing outputs and a
runnable configuration for reuse or extension. Reuse is not a guarantee of
scientific comparability: changed data, acquisition semantics, classifiers
or hyperparameters can invalidate comparisons even when files still load.
[Snapshots](output_snapshots.md) also omit Snakemake metadata, so restored
outputs require care after code or parameter changes.

For commands, see [run](../how-to/run_pipeline.md),
[customize](../how-to/customize_pipeline.md) or [SLURM](../how-to/run_on_slurm.md).
