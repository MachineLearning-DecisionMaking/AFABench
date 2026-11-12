# Customize the pipeline

Start from a working [pipeline command](run_pipeline.md). Add the overrides
below to that command and preview with `-i` before executing.
For the configuration layers and stages, see the [pipeline overview](../explanation/pipeline.md).

## Select methods, datasets and realizations

```shell
uv run workflow/run.py ... "methods=[gdfs,dime]" \
    "datasets=[cube,mnist]" "dataset_realization_indices=[0,1]" -i
```

Ensure each method has `method_options` and `soft_budget_params` entries,
and each dataset has `conf/components/dataset_key/<key>.yaml`.
Specify the complete realization list; retain existing indices to include
their results. To implement something new, follow [add a method](add_method.md)
or [add a dataset](add_dataset.md).

## Make changes durable

Copy a pipeline profile and the configuration files you need to edit:

```shell
cp -r workflow/profiles/pipeline/all workflow/profiles/pipeline/my_run
cp workflow/conf/soft_budget_params/all.yaml workflow/conf/soft_budget_params/my_run.yaml
# Repoint soft_budget_params/all.yaml to my_run.yaml in my_run/config.yaml.
# Edit workflow/conf/soft_budget_params/my_run.yaml.
uv run workflow/run.py my_run -i
```

Repeat for other configuration groups you change. Check selected methods,
datasets and parameter entries in the preview.
[Configuration groups](../reference/pipeline_configuration.md#configuration-overview).

## Change hard budgets

Override each dataset explicitly, in selection-cost units:

```shell
uv run workflow/run.py ... "eval_hard_budgets={cube: [2, 4], mnist: [10]}" -i
```

Retain old entries to include them in comparisons. Edit explicit dataset
entries, not just `default`. Check the method's training-budget mappings
before running; a new evaluation budget may require new training.
[Budget mappings](../reference/pipeline_configuration.md#method-option-keys-and-validation).

For soft-budget-only evaluation, empty the hard-budget lists:

```shell
uv run workflow/run.py ... "eval_hard_budgets={cube: [], mnist: []}" -i
```

## Change soft-budget parameters

Add the dataset to the method's `soft_budget_datasets`, then supply its
complete list of `[train, eval]` pairs. Set at most one value per pair,
following the method's supported convention:

```shell
uv run workflow/run.py ... \
    "soft_budget_params={jafa: {cube: [[0.001, null], [0.02, null]]}}" -i
```

For hard-budget-only evaluation, add `--no-soft-budget` instead.

## Choose CPUs or GPUs

For all-CPU execution, pass
`execution_file=workflow/profiles/execution/cpu.yaml`.
Otherwise, copy `workflow/profiles/execution/default.yaml` to your own file
and edit the stage defaults and overrides:

```yaml
execution:
  defaults:
    classifier_training: cuda
    pretraining: cpu
    training: cpu
    evaluation: cpu
  methods:
    jafa:
      classifier_training: cpu
      training: cuda
      evaluation: cpu
  pretrained_models:
    pvae: cuda
```

- Use only `cpu` or `cuda`; choose an evaluation device that supports both
  the policy and classifier.
- Set shared classifier training through `defaults.classifier_training`;
  method-level `classifier_training` applies only to method-specific classifiers.
- Name pretrained models by their key in `pretrain_mapping`, not the script
  or requesting method. Do not set method-level `pretraining`.
- Remove `device` from script parameters and invocation-wide overrides.
- Keep partitions, accounts and GPU requests in the [SLURM site profile](run_on_slurm.md),
  not this file. Dataset generation and result processing always use CPU.

Pass your file and inspect each job's Device, CPUs, GPUs and Placement:

```shell
uv run workflow/run.py ... execution_file=execution.yaml -i
```

Check device support and allocation access before removing `-i`.
[Execution precedence](../reference/pipeline_configuration.md#execution).

## Select plot metrics and groups

Choose a headline metric and optionally request additional per-dataset figures:

```shell
uv run workflow/run.py ... --per-dataset-plots \
    "plotting={dataset_metrics: {cube: auroc}}" -i
```

Read figures under `plots/`, including `eval_perf/` and, with the flag,
`eval_perf_per_dataset/`. Edit `method_sets` and `dataset_sets` to group
comparisons; the automatic `all` sets need no configuration.
[Plot options and layout](../reference/evaluation_plots.md).

To rerun plots from existing evaluations, add
`"stages={dataset_generation: false, training: false, evaluation: false}"`.
For restored plots, also append
`-- --forcerun plot_eval_perf plot_eval_perf_per_dataset`;
without `--per-dataset-plots`, force only `plot_eval_perf`.

To extend published results instead of starting fresh, follow the
[release guide](manage_releases.md#extend-a-release).
