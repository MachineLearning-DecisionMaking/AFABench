# L2M trained on the TabICL v1 prior

`l2m_better_synth` is L2M with one change: its [task
prior](../../CONTEXT.md) is the TabICL v1 prior, with its TreeSCM
sub-prior, instead of a [pretraining pool](../../CONTEXT.md) labelled by
upstream's random BNN. The model, both fit stages, their hyperparameters and
the evaluation are `l2m`'s. It never sees the benchmark's data during
fitting, apart from the dataset's feature and class counts and the [context
set](../../CONTEXT.md) it stores, so a comparison with other methods asks
how much L2M needs real data of the dataset's kind at all. Its runs are
described in [L2M](../reference/l2m.md).

## What a task is

A task is a classification dataset the prior generates: an MLP structural
causal model (66.1 %), a tree-based structural causal model (TreeSCM, 30 %)
or a Gaussian process (3.9 %) gives the features, and random class
boundaries on a latent value give the labels. The prior's feature and class
counts are fixed to the model's, such as 20 and 8 on the cube, and the
sampler checks every dataset and redraws one that does not match
(`TabICLTaskSampler.rejections` counts why). Tasks are fully observed: the
prior has no missing values and the context set is fully observed at
evaluation, whereas the other task priors simulate missingness. Every task is
new; there is no pool and no fixed set of tasks.

The pipeline gives each stage `task_prior=tabicl_v1
prior_type=prior_bag_plus_tree` (`pretrain_mappings/all.yaml` and the
`method_specific_params` of `method_options/all.yaml`); the task size and the
four task workers come from L2M's configs
([L2M](../reference/l2m.md#where-l2m-is-configured)).

## Why the task sizes

Training is autoregressive: query instance `i` attends to the 50 context
instances and to the true labels of the `i` query instances before it. A
task of `sequence_length` instances therefore shows a query between 50 and
`sequence_length - 1` labelled instances. At evaluation the context set is a
fixed draw from the validation split, at most `context_set_size` instances.
On the cube the tasks have 600 instances, so queries see up to 599 labelled
instances and the 200-instance context set lies inside the range training
covers; the 600 train instances are not part of the context set.

## Where training and evaluation differ

These differences are known and not corrected:

- **Class balance.** The prior puts class boundaries at random data points,
  so the classes of a task differ in size, whereas the cube's 8 classes are
  equally likely.
- **Attention among labelled instances.** In training only the 50 context
  instances attend to each other in both directions; the labelled query
  instances attend causally. At evaluation all context instances are plain
  context and attend to each other.
- **Distribution.** The prior's datasets are not the benchmark's. The model
  learns to classify datasets of the prior's kind in context, and how well
  that transfers is what the comparison measures.
- **Feature order.** The adapter does not shuffle feature columns; it keeps
  the prior's own column order.

## Cost

The prior runs on the CPU. Tasks per second for the cube are in the
[reference](../reference/l2m.md#task-generation-rate); up to about 30 % of
the draws are rejected for missing the feature or class count. A step needs
`batch_size` tasks, so each fit stage generates them in `task_workers`
processes ahead of use, while the GPU trains. Batch `i` of a stage is
generated under a seed derived from the fit's seed, the stage and `i` alone,
so a fit does not depend on the number of workers beyond floating-point
rounding, and the two stages do not see the same tasks. The workers need the
job's CPUs and about 0.5 GB each.

## Where it runs

On `cube`, `miniboone_subset` and `metabric`: the model's width is fixed by
a dataset's features and classes, so each dataset trains its own model, every
other dataset is ignored in the hard-budget setting, and there is no
soft-budget setting. One policy is trained per dataset realization and serves
every evaluation budget. The prior has 2 to 10 classes. The vendored prior
and its edits are described in
[ADR 0017](../adr/0017-tabicl-prior-vendored-with-opt-in-edits.md).
