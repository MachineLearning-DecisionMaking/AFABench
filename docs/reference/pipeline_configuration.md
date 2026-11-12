# Pipeline configuration

Configuration keys accepted by the [pipeline launcher](pipeline_launcher_command.md)
as `key=value` overrides. For the moving parts, see the
[pipeline overview](../explanation/pipeline.md); for recipes, see
[customize the pipeline](../how-to/customize_pipeline.md).

## Configuration overview

Configuration files are organized into subdirectories under
`workflow/conf/`. Each subdirectory contains multiple named variants
(e.g., `all.yaml`, `kdd26.yaml`). The command above uses the
`workflow/profiles/pipeline/all` profile, which bundles the commonly used
`all.yaml` config files and the pipeline Snakefile.
`workflow/profiles/pipeline/kdd26` bundles the `kdd26.yaml` files. Pipeline
profiles say only what to run; where to run it is a site profile under
`workflow/profiles/site/`, and which jobs use a GPU is the execution file
under `workflow/profiles/execution/`
([Run the pipeline](../how-to/run_pipeline.md)).
Below we discuss the meaning of each configuration group.

## Runtime configuration options

Pipeline profile overrides (`key=value` arguments of the pipeline launcher, which it passes to Snakemake's `--config`) allow you to customize how the pipeline runs without modifying configuration files. These options are:

### `eval_dataset_split`

Specifies which dataset split to use during evaluation.

- **Default:** `test`
- **Valid values:** `train`, `val`, `test`
- **Example:** `eval_dataset_split=val` to evaluate on the validation set instead of the test set

### `dataset_realization_indices`

Specifies which dataset realizations to run. This allows you to run a subset of the experiments. Each index corresponds to a different random seed for dataset generation, model initialization, and training.

- **Default:** `[0,1,2,3,4]`
- **Example:** `dataset_realization_indices=[0,1]` to run two dataset realizations
- **Use case:** Use fewer dataset realizations for faster debugging, more for more robust results
- **Note:** Adding an index later runs only the new dataset realizations' jobs; the existing ones are reused

### `device` (deprecated)

An invocation-wide device for all classifier, pretraining, training and
evaluation jobs. It still works on its own with a deprecation warning, but
cannot be combined with `execution` (see below), and dataset generation,
transformations, aggregation and visualization always run on CPU regardless.

- **Default:** `cpu`
- **Valid values:** `cpu`, `cuda`
- **Instead:** declare `execution` choices in a config file

### `execution`

A mapping that declares per job whether it runs on
`cpu` or `cuda`: pipeline stage defaults for `classifier_training`, `pretraining`,
`training` and `evaluation`, overrides per method and stage, and overrides
per named pretrained model. Unspecified stages run on CPU. The resolved choice is both
the script's `device` argument and, under SLURM, the CPU or GPU allocation.
It is read from the file named by `execution_file`, by default
`workflow/profiles/execution/default.yaml`; `execution/cpu.yaml` runs every
job on CPU. A mapping given inline in a config file takes precedence over the
file, and an explicit deprecated `device` disables the file. Entries for
methods a run does not select are ignored. The format, from
`execution/default.yaml`:

```yaml
execution:
  defaults:                     # per pipeline stage; unspecified ones default to cpu
    classifier_training: cuda   # external and method-specific classifiers
    pretraining: cuda           # named pretrained models
    training: cpu
    evaluation: cpu
  methods:
    jafa:                       # per method: training, evaluation, classifier_training
      training: cuda
      evaluation: cuda
  pretrained_models:   # per named model in pretrain_mapping (none here)
    pvae: cuda
```

Values are exactly `cpu` and `cuda`. Precedence, resolved independently for
every job:

1. `execution.methods.<method>.<training|evaluation|classifier_training>`, or
   `execution.pretrained_models.<named model>` for pretraining.
2. `execution.defaults.<classifier_training|pretraining|training|evaluation>`.
3. `cpu`.

Training and evaluation are independent, so a method can train on GPU and
evaluate on CPU. Evaluation runs the classifier too, so declare its
end-to-end needs. Shared external classifiers use only
`defaults.classifier_training`; a pretrained model shared by several methods
uses its own name, never a requesting method's choice. Dataset generation,
transformations, aggregation and visualization always run on CPU and cannot
be configured. Hardware is never inferred from a method's implementation,
taxonomy or the selected method list, and a `cuda` job never falls back to
CPU. `cuda` both passes `device=cuda` to the script and requests a GPU
allocation; `cpu` passes `device=cpu` and requests none. Partitions and
accounts never belong in these files; the site profile owns them
([SLURM integration](../how-to/run_on_slurm.md)).

### `--jobs` (launcher option)

Controls the number of jobs Snakemake runs in parallel locally. This is not an override but an option of the [pipeline launcher](pipeline_launcher_command.md); a site profile sets it for its cluster.

- **Default:** 1 (serial execution)
- **Example:** `--jobs 8` to run 8 jobs in parallel
- **Note:** Set this to the number of CPU cores available on your machine for optimal parallelization. Be mindful of memory usage when increasing this value.

### `use_wandb`

Enables or disables [Weights & Biases](https://wandb.ai/) integration for logging metrics.

- **Default:** `false`
- **Example:** the launcher's `--wandb` sets `use_wandb=true`
- **Requirement:** You must run `uv run wandb login` before using W&B, and set
  the `WANDB_ENTITY` and `WANDB_PROJECT` environment variables, or the run
  fails before any job. The terminal links each job's W&B run with them
  ([Log to Weights & Biases](../how-to/run_pipeline.md#log-to-weights--biases)).
- **Requirement:** [`wandb_group`](#wandb_group) must be set too; the launcher sets it.

### `wandb_group`

The [W&B group](../../CONTEXT.md) every W&B run of the invocation joins. The
resolved configuration links it.

- **Default:** none; required with `use_wandb=true`, or the run fails before any job
- **Example:** the launcher's `--wandb` names a new group per invocation, such as
  `all-20261012T121005Z-k3f9` (pipeline profile, UTC time, random characters,
  and `-smoke` for a smoke test); `--wandb-group all-20261012T121005Z-k3f9` sets
  `use_wandb=true wandb_group=all-20261012T121005Z-k3f9`, so a rerun continues
  that group

### `smoke_test`

Enables smoke testing mode, where each script runs as fast as possible while still verifying correctness. Useful for checking if the pipeline can execute successfully before running expensive experiments.

- **Default:** `false`
- **Example:** `smoke_test=true` for quick validation runs
- **Note:** A smoke test writes under its own output root, `output/smoke` by default (see [`output_root`](#output_root))

### `list_jobs`

Prints every job's messages in the terminal: its rule block with its resources, and, in a dry run, its script command; in a real run also its submission and completion. Without it, the terminal shows only the resolved configuration, how many jobs of each rule run, progress (`N of M steps done`), each job's rule and wildcards when it is submitted or starts and when it finishes, with its W&B run link under [`use_wandb`](#use_wandb), and errors. A real run's log file under `.snakemake/log` keeps every job's messages either way.

- **Default:** `false`
- **Example:** `list_jobs=true` to check each job's partition, GPU request and `device=` argument

### `output_root`

The output root the run writes all its bundles, evaluation tables, job records, merged results and plots under. A smoke test has its own, so its outputs never satisfy a real run and a real run after a smoke test still runs every job.

- **Default:** `output/production`, or `output/smoke` with `smoke_test=true`
- **Example:** `output_root=/scratch/afabench/output`
- **Note:** An `output_root` that is, contains or lies inside the other kind of run's default root is a config error, for example `smoke_test=true` with `output_root=output/production`, or `output_root=output`

### `soft_budget_setting`

Whether the run trains and evaluates in the [soft-budget setting](../../CONTEXT.md) as well as the hard-budget setting. With `false`, every method runs only its hard budgets, as if its `soft_budget_datasets` listed no dataset, so pretraining skips the datasets a method also ignores in the hard-budget setting.

- **Default:** `true`
- **Example:** `soft_budget_setting=false`, or the launcher option `--no-soft-budget`, to run only the hard-budget setting
- **Note:** `true` or `false`, in any case; another value is a config error. The dry run's resolved configuration shows it, and lists no soft-budget parameters with `false`

### `stages`

Turns whole pipeline stages off. A stage that is off defines no rules, so the run reads that stage's outputs as they are under the output root, never schedules its jobs, and fails during planning if an output it needs is missing. Transformation, aggregation and visualization are always on.

| Key | Pipeline stages it covers |
| --- | --- |
| `dataset_generation` | Dataset generation |
| `training` | Pretraining, training and classifier training |
| `evaluation` | Evaluation |

- **Default:** every key `true`
- **Example:** `"stages={dataset_generation: false, training: false}"` to evaluate existing trained methods; add `evaluation: false` to only transform, aggregate and plot existing evaluation tables
- **Note:** Each value is `true` or `false`, in any case. An unknown key, another value, or a `stages` that is not a mapping is a config error. The dry run's resolved configuration lists the stages that are off

### `reuse_release`

- **Type:** `bool`
- **Default:** `false`
- **Requires:** `<output_root>/release_manifest.json` and
  `stages={dataset_generation: false}` during execution. The launcher's
  `--configfile` mode manages dataset preparation and these switches
  automatically. For a named-profile run, generate missing datasets first
  with the `all_generate_datasets` target and reuse disabled.
- **Behavior:** Checks actual dataset bundle content hashes against existing
  released artifacts' recorded input hashes, for the selected datasets and
  realizations. A mismatch, unknown hash or missing required dataset fails
  planning. Only matching dataset dependencies of existing released outputs
  ignore timestamps. Missing outputs, locally replaced model bundles and
  non-release artifacts retain normal dependencies. Other dependency edges
  remain unchanged, so a newly trained prerequisite can invalidate evaluations.
- **Example:** `reuse_release=true "stages={dataset_generation: false}"`
- **Scope:** Bundle contents, not external source files, code compatibility
  or changed hyperparameters. Ordinary runs with the default `false` retain
  normal invalidation. With `--configfile`, dry runs and estimates may preview
  reuse of missing datasets, conditional on content validation after generation.
  The internal `release_preview` switch is refused during execution. See the
  [launcher](pipeline_launcher_command.md) and
  [release guide](../how-to/manage_releases.md#extend-a-release).

### `per_dataset_plots`

- **Type:** bool
- **Default:** false
- **Example:** `--per-dataset-plots`, or `per_dataset_plots=true`
- **Meaning:** Request additional per-dataset figures, once per method set,
  alongside usual classifier-selected plots. Each figure shows Accuracy,
  Macro F1 and AUROC for external and available built-in classifiers.
  [Plot output layout](evaluation_plots.md#output-layout).

### `plotting`

- **Type:** mapping with only `dataset_metrics` and `default_metric`
- **Default:** `{}` (keep the Hydra display defaults)
- **Example:** `"plotting={dataset_metrics: {cube: auroc}, default_metric: accuracy}"`
- **Meaning:** Choose the headline metric for usual plots. `dataset_metrics`
  maps dataset keys to exact identifiers `accuracy`, `f1`, `auroc`;
  `default_metric` is one of those identifiers. Both are optional. Unknown
  keys, invalid types and unsupported identifiers fail configuration loading.
  There is no `--plot-metric` flag. These choices are forwarded to Hydra
  plotting config and do not change the optional three-metric figures.
  [Defaults and column mapping](evaluation_plots.md#metric-selection).

## Requested walltime (site configuration)

The site's `site.yaml` can set requested walltime in hours by Snakemake
rule, method/pretrained model and dataset. This is not a method
hyperparameter or an acquisition budget. See the
[walltime lookup reference](requested_walltime.md) for fields, precedence,
validation and a MiniBooNE example.

## Datasets

The `workflow/conf/datasets/` directory contains dataset configuration files. Each file specifies which datasets are used in the pipeline.

`workflow/conf/dataset_sets/` contains files that define *dataset sets*,
which group related datasets for plotting. Each dataset set gets its own
`dataset_set-<name>` plot folder for every method set. Datasets outside the
run are left out of each set, and sets without any dataset of the run are
skipped. The `all` dataset set, of every dataset of the run, always exists
and cannot be defined in these files.

## Unmaskers

`workflow/conf/unmaskers/` contains files that map datasets to unmaskers. The values correspond to files in `conf/components/unmasker/`.

For example, if `workflow/conf/unmaskers/all.yaml` contains
```yaml
unmaskers:
  default: direct
  imagenette: 224x224_to_14x14
```
then `imagenette` will use a patch-based unmasker while all other datasets will have the "normal" unmasker that maps actions directly to features.

`conf/components/unmasker/224x224_to_14x14.yaml` contains the details about this specific unmasker:
```yaml
class_name: "ImagePatchUnmasker"
kwargs:
  image_side_length: 224  # Imagenette size
  n_channels: 3
  patch_size: 16  # 14x14 grid = 196 patches
```

## Hard budgets

`workflow/conf/eval_hard_budgets/` determines what hard budgets are used for each dataset **during evaluation**. Methods are free to use different budgets during training, see [below](#methods-and-their-soft-budget-parameters).

For example, a file in `workflow/conf/eval_hard_budgets/` might contain
```yaml
eval_hard_budgets:
  default: [5, 10, 15]
  cube: [3, 5, 10]
  imagenette: [5, 10, 15]
```

Note that the `default` setting is used for all unlisted datasets, and that the budget describes the number of **allowed actions**, not the number of features. This is an important distinction when a different unmasker than `direct` is used.

## Methods and their soft-budget parameters

The methods require the most configuration, and use the directories
- `workflow/conf/methods/`
- `workflow/conf/method_sets/`
- `workflow/conf/pretrain_mappings/`
- `workflow/conf/method_options/`
- `workflow/conf/soft_budget_params/`

`methods/` contains files listing which methods are included in the pipeline.

`method_sets/` contains files that define *method sets*, which group related methods to prevent cluttered plots when visualizing results. Each method set gets its own separate plot. The `all` method set, of every
method in the run, always exists and cannot be defined in these files.

Some methods require a pretraining stage. For such methods,
`pretrain_mappings/` provides the mapping to the pretraining script. For example, a file in `workflow/conf/pretrain_mappings/` with contents
```yaml
pretrain_mapping:
  pvae:
    pretrain_script_name: "odin"
    pretrain_params: []
```
will define a model `pvae` which is produced by the `scripts/pretrain_model/odin.py` script. This can later be reused across different methods.

For example, a file in `workflow/conf/method_options/` contains miscellaneous options for each method. An example configuration:
```yaml
method_options:
  eddi_external:
    pretrained_model_name: "pvae"
    train_script_name: "eddi_external"
    use_max_hard_budget_when_training_soft_budget: true
    eval_batch_size:
      default: 8
    train_hard_budget: {cube_nonuniform_costs: 20}
    hard_budget_ignored_datasets: [mnist, fashion_mnist, imagenette]
    soft_budget_datasets: [cube, cube_nonuniform_costs, miniboone]
  odin_model_based:
    pretrained_model_name: "pvae"
    train_script_name: "odin"
    method_specific_params:
      - "additional_generation_fraction=1.0"
    eval_batch_size:
      default: 128
    hard_budget_ignored_datasets: [imagenette]
    soft_budget_datasets: [cube, miniboone]
```
defines two methods `eddi_external` and `odin_model_based` which both use the same pretrained `pvae` model. Furthermore, they use different batch sizes during evaluation, skip some datasets in the hard-budget setting, and run the soft-budget setting only on the datasets they list. `eddi_external` is a bit special, in that on `cube_nonuniform_costs` it trains every hard-budget run with hard budget 20, whatever budget it is evaluated with.

Usually during the *soft-budget* setting, the hard budget is disabled. `use_max_hard_budget_when_training_soft_budget` enforces the largest hard budget instead.

### Method option keys and validation

A `method_options` entry takes only these keys:

| Key | Meaning |
| --- | --- |
| `train_script_name` | Required. The training script, `scripts/train_method/<name>.py`. |
| `pretrained_model_name` | A key of `pretrain_mapping`; the method then has a pretraining stage. |
| `method_specific_params` | Extra `key=value` arguments for the training script. |
| `eval_batch_size` | An integer for every dataset, or a mapping from dataset key to integer with a `default` for the others. If absent, evaluation uses batch size 1 and the pipeline warns. |
| `hard_budget_ignored_datasets` | Dataset keys the method is not trained or evaluated on in the hard-budget setting. |
| `soft_budget_datasets` | Dataset keys the method is trained and evaluated on in the soft-budget setting; it runs no soft budgets on any other dataset, whatever `soft_budget_params` gives it ([ADR 0013](../adr/0013-soft-budget-datasets-are-opt-in.md)). |
| `train_hard_budget` | Per dataset key, the one train hard budget of every hard-budget run on that dataset, whatever its eval hard budget. Other datasets train at the eval hard budget. |
| `eval_to_train_hard_budget_mapping` | Per dataset key, the train hard budget for each eval hard budget, when one train hard budget does not fit them all. It must map every eval hard budget the dataset gets. |
| `use_max_hard_budget_when_training_soft_budget` | `true` to train soft-budget runs under the largest train hard budget. |
| `classifier` | The method's built-in classifier: `script_name` and optional `script_params`, trained instead of the external classifier. |

The workflow config is validated when Snakemake parses the workflow, by
`load_config` in `src/afabench/core/workflow_settings.py`, before any job runs.
Each error names the method and the key or value at fault:

- An unknown key in a `method_options` entry, such as a misspelt
  `pretrained_model_nme`, or a missing `train_script_name`, is rejected, for
  every entry, whether or not the method is selected.
- An ignored or soft-budget dataset, and a dataset in `train_hard_budget` or
  `eval_to_train_hard_budget_mapping`, must be a dataset key, a file in
  `conf/components/dataset_key/`. It need not be in this run's
  `datasets`, which only selects what runs.
- A dataset takes `train_hard_budget` or `eval_to_train_hard_budget_mapping`,
  not both, and its `eval_to_train_hard_budget_mapping` must map every eval
  hard budget it gets in the run, unless the method ignores it in the
  hard-budget setting.
- `eval_batch_size`, when a mapping, needs a `default`.
- A `pretrain_mapping` entry takes only `pretrain_script_name`, which it
  requires, and `pretrain_params`.
- A method's `classifier`, and a `classifier_names` value given as a
  mapping, take only `script_name` and `script_params`.
- Every method in `methods` must have a
  `method_options` entry and a `soft_budget_params` entry, and its
  `pretrained_model_name`, if any, must be a key of `pretrain_mapping`.

Lastly, files in `workflow/conf/soft_budget_params/` contain the per-dataset soft-budget parameters for each method. Each soft-budget parameter is represented as a tuple `(train_soft_budget_param, eval_soft_budget_param)`. While the `default` key **can** be used, it is recommended to tune the values for each dataset due to sensitivity issues.

## Classifiers

During evaluation, we need predictions from an *external* classifier. Files in `workflow/conf/classifier_names/` determine which classifier is used for which dataset. You can edit these mappings to test different classifiers on your datasets. For example,
```yaml
classifier_names:
  default: "masked_mlp_classifier"
  imagenette: "masked_vit_classifier"
```
will use a vision transformer for the `imagenette` dataset but a normal `MLP` classifier for the other datasets.

## Running specific steps of the pipeline

The whole point of Snakemake is to run jobs maximally parallelized. Still, there might be good reasons for only running specific steps across all methods. For example, perhaps all methods need to be pretrained and trained without being evaluated. For this, we provide various `all_X` rules in `workflow/snakefiles/rules/helpers.smk`. These rules should replace the `all` rule from the command at the beginning of this document.

The currently available `all_X` rules are:

| Rule | Purpose |
|------|---------|
| `all_generate_datasets` | Generate all dataset splits (train/val/test) |
| `all_train_classifiers` | Train classifiers for all datasets |
| `all_pretrain_models` | Pretrain models for all methods that require it |
| `all_train_methods` | Train all methods across all datasets and budget configurations |
| `all_eval_methods` | Evaluate all trained methods and produce evaluation results |

To use one of these rules, pass it with `--target`. For example, to only generate datasets and train classifiers without training methods, use:

```shell
uv run workflow/run.py --target all_train_classifiers ...
```

A target only narrows what Snakemake builds; it still regenerates any missing or outdated upstream output. To use upstream outputs as they are instead, turn their stages off with [`stages`](#stages).
