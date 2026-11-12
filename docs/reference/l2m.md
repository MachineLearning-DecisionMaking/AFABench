# L2M

L2M (Learning-To-Measure, also LTM) is an [in-context
method](../../CONTEXT.md): one transformer that is the classifier and the
policy. This page lists where its runs are configured, which files a run
uses, and what the three runs and the three datasets they are configured for
are. How to run it: [run L2M](../how-to/maintainer/run_l2m.md). How it
compares with the authors' code:
[the audit](../explanation/l2m_baseline_audit.md).

Terms: [task prior](../../CONTEXT.md), [pretraining pool](../../CONTEXT.md),
[context set](../../CONTEXT.md), [dataset override](../../CONTEXT.md).

## The runs

The model's width is fixed by the dataset: its feature count, and its class
count (`label_dim` 1 with a sigmoid for two classes, otherwise one column per
class). Each run therefore trains one model per dataset realization, and a
model is never used on another dataset. The runs differ in their task prior
only; the three datasets they are configured for are `cube`,
`miniboone_subset` and `metabric`, in the hard-budget setting.

| Run (method name) | Task prior | What the fit may see | Pipeline parameters |
| --- | --- | --- | --- |
| L2M-real (`l2m`) | `real_pool`: train split's features, labelled by upstream's random BNN | The train split's features, not its labels; the validation split as the context set | none |
| L2M-synth (`l2m_synth`) | `synthetic`: upstream's simulation prior, new uniform features on [-2, 2] for every task with independent missing values, labelled by upstream's random BNN | Only the feature and class counts; the validation split as the context set | `task_prior=synthetic` |
| L2M-better-synth (`l2m_better_synth`) | `tabicl_v1` with `prior_type=prior_bag_plus_tree`: MLP 66.1 %, Gaussian process 3.9 %, TreeSCM 30 % | Only the feature and class counts; the validation split as the context set | `task_prior=tabicl_v1 prior_type=prior_bag_plus_tree` |

[L2M-synth is defined](../explanation/l2m_baseline_audit.md#the-definition-of-l2m-synth)
as upstream's simulation prior with the real-data experiments' labels. A run
that learns from the train split labels does not exist; no run reads them.

`l2m`, `l2m_synth` and `l2m_better_synth` are in `workflow/conf/methods/all.yaml`
and run, with the other methods, on the three datasets with

```shell
uv run workflow/run.py l2m_comparison
```

(profile `workflow/profiles/pipeline/l2m_comparison/config.yaml`, datasets
`workflow/conf/datasets/l2m_comparison.yaml`). They skip the datasets in their
`hard_budget_ignored_datasets`, and there is no soft-budget setting. One
policy per dataset realization is trained for the training hard budget and
serves every evaluation budget (`train_hard_budget`, below).

### Per-dataset settings

Every run uses the settings of the dataset's override files, which
`conf/scripts/pretrain_model/l2m/dataset_override/<dataset>.yaml` and
`conf/scripts/train_method/l2m/dataset_override/<dataset>.yaml` hold. Shapes
come from the dataset at run time. The override files repeat
values that equal the defaults, so that all of a dataset's settings are in one
file; leaving them out would change nothing.

| | `cube` | `miniboone_subset` | `metabric` |
| --- | --- | --- | --- |
| Features, classes, `label_dim` | 20, 8, 8 | 10, 2, 1 | 11, 6, 6 |
| Train, validation instances (60/20/20 split) | 600, 200 | 77,755, 25,918 | 1,138, 379 |
| `sequence_length` (both stages) | 600 | 1000 | 500 |
| `pretraining_pool_size` (`l2m` only) | 600 | 5000 | 1000 |
| `n_steps` pretraining / training | 100,000 / 50,000 | 100,000 / 50,000 | 100,000 / 50,000 |
| `batch_size` (both stages) | 8 | 8 | 8 |
| `context_set_size` (maximum; capped at the validation split) | 900 (200) | 900 | 900 (379) |
| Stored context set | 200 | 900 | 379 |
| Training hard budget | 10 | 7 | 7 |
| Authors' numbers | no: 600 instances, the train split; pretraining steps chosen for the thesis | `sequence_length`, pool and training steps; pretraining steps chosen for the thesis (theirs: 50,000) | yes (their METABRIC config) |

`sequence_length - 50` is the number of labelled instances a training query
can see before it; the context set should not be larger, and `train_method`
warns when it is. `l2m` draws its tasks from a pool of the train split without
replacement, so its tasks cannot have more instances than the pool; the cube's
600 are the whole train split, and every run uses them. The split sizes are
60 % and 20 % of 1,000, 129,592 and 1,898 instances, rounded down
(`conf/scripts/dataset_generation/config.yaml`).

### Settings per run and dataset

| Run | Dataset | Task prior (`prior_type`) | Task workers | Task source | `sequence_length` | Pool | `n_steps` pretraining / training | `batch_size` | Context set stored | Training hard budget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `l2m` | `cube` | `real_pool` | 0 | pool of the train split's features | 600 | 600 | 100,000 / 50,000 | 8 | 200 | 10 |
| `l2m` | `miniboone_subset` | `real_pool` | 0 | pool of the train split's features | 1000 | 5000 | 100,000 / 50,000 | 8 | 900 | 7 |
| `l2m` | `metabric` | `real_pool` | 0 | pool of the train split's features | 500 | 1000 | 100,000 / 50,000 | 8 | 379 | 7 |
| `l2m_synth` | `cube` | `synthetic` | 0 | new uniform features per task | 600 | unused | 100,000 / 50,000 | 8 | 200 | 10 |
| `l2m_synth` | `miniboone_subset` | `synthetic` | 0 | new uniform features per task | 1000 | unused | 100,000 / 50,000 | 8 | 900 | 7 |
| `l2m_synth` | `metabric` | `synthetic` | 0 | new uniform features per task | 500 | unused | 100,000 / 50,000 | 8 | 379 | 7 |
| `l2m_better_synth` | `cube` | `tabicl_v1` (`prior_bag_plus_tree`) | 4 | TabICL v1 prior with TreeSCM | 600 | unused | 100,000 / 50,000 | 8 | 200 | 10 |
| `l2m_better_synth` | `miniboone_subset` | `tabicl_v1` (`prior_bag_plus_tree`) | 4 | TabICL v1 prior with TreeSCM | 1000 | unused | 100,000 / 50,000 | 8 | 900 | 7 |
| `l2m_better_synth` | `metabric` | `tabicl_v1` (`prior_bag_plus_tree`) | 4 | TabICL v1 prior with TreeSCM | 500 | unused | 100,000 / 50,000 | 8 | 379 | 7 |

The task prior and its workers come from `task_prior/<name>.yaml`, the
other columns from the dataset's override files, so within a dataset only the
task prior differs between the runs. Every other value (model size, learning
rates, temperature, validation tasks) is the same for every run and dataset,
except the gradient clipping: in both stages `l2m_better_synth` clips the
gradient's total norm to 1.0 and the other two runs do not clip
(`max_grad_norm`, below).

## Where L2M is configured

Hydra composes a script's configuration in the order of the defaults list of
`conf/scripts/{pretrain_model,train_method}/l2m/config.yaml`: the structured
config (`L2MPretrainingConfig` or `L2MTrainingConfig` in
`src/afabench/components/methods/discriminative/l2m/config.py`), the
`initializer`, `unmasker` and `dataset_key` components, the `task_prior`
group, the file's own default hyperparameters, then
`dataset_override/<dataset_key>.yaml`. Command-line overrides come last: the
pipeline's contract arguments, then the parameters of
`workflow/conf/pretrain_mappings/all.yaml` (pretraining) or
`method_options` (training). A bad combination raises when the structured
config is built.

| Key | File | Default | Overridden by |
| --- | --- | --- | --- |
| `task_prior` | `task_prior/<name>.yaml` beside `config.yaml` | `real_pool` | `task_prior=<name>` in `pretrain_params` and `method_specific_params` |
| `task_workers` | `task_prior/<name>.yaml` | 0 (`real_pool`, `synthetic`), 4 (`tabicl_v1`) | the command line; must be 0 except with `tabicl_v1` |
| `max_grad_norm` | `task_prior/<name>.yaml` beside `config.yaml`, in both stages | null, no clipping (`real_pool`, `synthetic`); 1.0 (`tabicl_v1`) | the command line (`max_grad_norm=null` or a positive number); in training the policy head's and the other parameters' gradients are clipped together |
| `prior_type` | `config.yaml` | `prior_bag` | `prior_type=<type>`; only with `tabicl_v1` |
| `sequence_length`, `n_steps`, `batch_size`, `pretraining_pool_size` | `config.yaml` | 1000, 50,000, 8, 5000 | `dataset_override/<dataset>.yaml`, then the command line |
| `context_set_size` | `train_method/l2m/config.yaml` | 900 | the dataset override |
| `architecture.*` | `pretrain_model/l2m/config.yaml` | 256 wide, 6 layers, 4 heads, 512 feed-forward, embedding depth 4, prediction depth 1, no dropout | the command line; training takes it from the pretrained model |
| `lr`, `warmup_steps`, `checkpoint_interval`, `n_validation_tasks`, `mixed_precision` (pretraining) | `pretrain_model/l2m/config.yaml` | 1e-4, 500, 500, 200, true | the command line |
| `policy_lr`, `backbone_lr`, `temperature`, `checkpoint_interval`, `n_validation_tasks`, `mixed_precision` (training) | `train_method/l2m/config.yaml` | 1e-4, 1e-5, 0.1, 200, 200, true | the command line |
| Smoke-test sizes | `scripts/pretrain_model/l2m.py`, `scripts/train_method/l2m.py` (`limit(...)`) | 64 (`real_pool`, `synthetic`) or 100 (`tabicl_v1`) instances, 2 steps, 2 validation tasks, no task workers | |
| `pretrained_model_name`, `train_script_name`, `method_specific_params` | `workflow/conf/method_options/all.yaml` | per method | a profile's own `method_options` file |
| `pretrain_script_name`, `pretrain_params` | `workflow/conf/pretrain_mappings/all.yaml` | per pretrained model | a profile's own file |
| `train_hard_budget` | `method_options/all.yaml` | `{cube: 10, miniboone_subset: 7, metabric: 7}` | a profile's own file |
| `hard_budget_ignored_datasets` | `method_options/all.yaml` | every dataset except the three | a profile's own file |
| `eval_batch_size` | `method_options/all.yaml` | 128 | a profile's own file |
| `eval_hard_budgets` | `workflow/conf/eval_hard_budgets/all.yaml` | `cube` 3, 5, 10; `miniboone_subset` 1 to 4; `metabric` 3, 5, 7 | the profile |
| Soft-budget entry (required, empty) | `workflow/conf/soft_budget_params/all.yaml` | `default: []` | |
| Method names, method set | `workflow/conf/methods/all.yaml`, `method_sets/all.yaml` | the three L2M names | |
| Device | `workflow/profiles/execution/default.yaml` | `cuda`, training and evaluation | `execution_file=` |
| Requested walltime | `workflow/profiles/site/<site>/site.yaml` | [per site](requested_walltime.md) | |
| Display name, colour | `conf/scripts/plotting/common/default.yaml` | `L2M(real)`, `L2M(synth)`, `L2M(better synth)` in three greys | |

## Which files a run uses

1. The Snakemake rule `pretrain_model` (`workflow/snakefiles/rules/training.smk`)
   starts `scripts/pretrain_model/l2m.py` with the contract and the
   pretraining parameters.
2. The script builds the config (above), shortens it in a smoke test, loads
   the datasets (`afabench.fit.inputs.load_inputs`) and calls
   `pretrain_l2m` in `l2m/pretrain.py`.
3. `L2MModel.build` (`l2m/models.py`) builds the authors' `LTM` through
   `upstream/model_utils.py` and `upstream/ltm.py`.
4. `make_task_sampler` (`l2m/upstream_fit.py`) builds the task sampler of
   the task prior:
   - `real_pool` draws a pool from the train split's features
     (`draw_pretraining_pool`), which the vendored `RealDataSampler`
     (`upstream/data_utils.py`) samples and labels with the vendored random
     function;
   - `synthetic` builds `SyntheticTaskSampler`
     (`l2m/synthetic_task_sampler.py`): new uniform features and missing
     values for every task as upstream's simulation draws them, labelled with
     the vendored random function;
   - `tabicl_v1` builds `TabICLTaskSampler` (`l2m/tabicl_task_sampler.py`),
     which asks the vendored prior (`l2m/tabicl_v1_prior/`: the MLP and GP
     sub-priors, and TreeSCM when `prior_type` includes it, behind the
     classification adapter and `BagPrior`) for datasets with exactly the
     model's features and classes.
5. `fit_upstream` (`upstream_fit.py`) runs the loop: the vendored
   `predictor_training_step` per batch and the vendored `eval` on the
   validation tasks that `gen_evalset` drew, keeping the best state.
6. `save_result` saves the model as the pretrained model bundle
   (`pretrained_models/<tag>/<pretrained model>/dataset-<key>+realization_index-<k>/pretrain_seed-<s>/`).
7. The rule `train_method` starts `scripts/train_method/l2m.py`, which calls
   `train_l2m` (`l2m/train.py`): the same sampler, the vendored
   `afa_training_step` and `eval_afa`, then the context set drawn from the
   validation split, packaged with the model as `L2MAFAMethod`
   (`l2m/afa_methods.py`) and saved as the method bundle
   (`trained_methods/<tag>/<training>/`).
8. At evaluation `L2MAFAMethod.act` calls the model's `select_action` and
   `predict` calls its predictor, both on the stored context set and the
   query instances.

Vendored, with the patches listed in each folder's README:
`l2m/upstream/` (the authors' model, training steps, validation and random
function) and `l2m/tabicl_v1_prior/` (the TabICL v1 prior; see
[ADR 0017](../adr/0017-tabicl-prior-vendored-with-opt-in-edits.md)). Written
for the benchmark: everything else in `l2m/`, the two scripts and the
workflow and Hydra files above.

## Task generation rate

`prior_bag_plus_tree` and `tree_only` need `xgboost-cpu` and scikit-learn.
Tree models use one thread per task process (`_TREE_MODEL_THREADS` in
`tabicl_task_sampler.py`). Tasks generated per second for the CUBE at 250
instances (the runs use 600), on one Windows machine with 12 CPUs, one measurement each:

| Prior | In the fit's process | 4 task workers |
| --- | --- | --- |
| `prior_bag` | 71 | 187 |
| `prior_bag_plus_tree` | 67 | 81 |
| `tree_only` | 23 | 77 |

One step needs `batch_size` tasks (8 by default). In one process, tree models
with one thread were 14 % (`prior_bag_plus_tree`) and 7 % (`tree_only`) faster
than with every CPU. Between 0 and 30 % of the draws are rejected because
they miss the dataset's feature or class count; `TabICLTaskSampler.rejections`
counts why.

## Checks

The configs raise `ValueError` for: a `sequence_length` of at most 50; a
pretraining pool smaller than `sequence_length` (`real_pool`); task workers
except with `tabicl_v1`; a negative worker count; `prior_type` other than
`prior_bag` except with `tabicl_v1`; `batch_size`, `checkpoint_interval` or
`n_validation_tasks` below 1; `context_set_size` below 1. The TabICL sampler raises for fewer than 2 or
more than 10 classes (the prior's range) and for a `label_dim` that does not
encode the class count. `train_method` raises when the pretrained model's
feature or class count differs from the train split's, and when the training
hard budget is not below the number of selections.

Not corrected, known: `data_prior_config.py` names `tree_depth_lambda` and
`tree_n_estimators_lambda`, but `TreeSCM` reads `max_depth_lambda` and
`n_estimators_lambda`, so its default 0.5 applies.
