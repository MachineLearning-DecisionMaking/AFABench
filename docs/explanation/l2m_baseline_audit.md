# How the L2M runs relate to the authors' code

The three L2M runs (`l2m`, `l2m_synth`, `l2m_better_synth`, described in the
[L2M reference](../reference/l2m.md)) train the authors' model with the
authors' training steps. This page records what was compared with the
authors' repository and with the thesis repository `lerning-to-measure`
(called the reference repository below), what differs, and what is still open.

## What was compared

The vendored files of `upstream/` were compared function by function with the
reference repository, by syntax tree with docstrings removed, and with the
last commit of the authors' repository that its history holds (`f852dc1`,
2025-10-14). The vendored commit (`9cb4643`) is not in the reference
repository's clone, so the two upstream commits were compared by their code,
which is identical apart from the listed patches.

| Vendored file | Functions | Result |
| --- | --- | --- |
| `ltm.py` | `ResidualMLP`, `TNP.__init__`, `construct_input`, `encode`, `LTM.__init__`, `forward`, `predict`, `select_action` | Identical |
| `ltm.py` | `TNP.create_mask` | The mask is built on the batch's device, not `'cuda'` (`PATCH(afabench)`) |
| `train_utils.py` | `predictor_training_step`, `afa_training_step`, `afa_training_step_random` | Identical |
| `model_utils.py` | `get_model` | Identical |
| `experiment_utils.py` | `get_eval_dir`, `get_eval_path` | Identical |
| `experiment_utils.py` | `gen_evalset`, `eval`, `eval_afa` | Validation tasks are returned and kept in the config instead of cached on disk, and use the fit's device (`PATCH(afabench)`). The reference repository additionally treats the experiment name `better_synth` like `sim` in `gen_evalset`; the vendored code does not, which changes nothing because `dataset_key` is never one of the two |
| `utils.py` | `ConcreteSelector`, `MLP`, `get_logger`, `make_onehot` | Identical |
| `utils.py` | `generate_uniform_mask` | Only vendored; the reference repository removed it. Unused |
| `data_utils.py` | `RealDataSampler` (`sample`, `_sample_indices`, `_normalize`, `_generate_missingness`, `mask_features`, `initialize_mask`, `acquire_features`), `TabularDataset` | Identical |
| `data_utils.py` | `sample_random_function` | The generator's seed is drawn from torch's global generator, and the multiclass prior forks every visible CUDA device's generator (`PATCH(afabench)`) |
| `data_utils.py` | `sample_test`, `sample_context` | Only vendored; the reference repository removed them. Unused |
| `utils.py`, `data_utils.py` | Baseline models, dataset loaders | Removed from the vendored copy (`PATCH(afabench)`); not in the reference repository either |

The attention mask, the input tokens, the predictor and AFA losses, the
sequential strategy, the Gumbel-softmax selector, Adam, bf16 autocast and
both validation functions are therefore the reference repository's. No
difference between the vendored upstream commit and the reference
repository's code changes training behaviour.

### The fit loops

`upstream_fit.py`, `pretrain.py` and `train.py` replace the loops of
`real.py` (`real_train`, `real_train_afa`), which serve the pool task priors,
and `sim.py`, which serves the TabICL prior in the reference repository.

| Aspect | Reference repository | AFABench |
| --- | --- | --- |
| Batch of tasks | `sampler.sample(batch_size, max_num_points=num_points, num_observed=feature_dim)` | The same call |
| Phase 1 steps, learning rate | `num_steps`, Adam, linear warm-up then linear decay to 0 | `n_steps`, `lr`, `warmup_steps`; `LambdaLR` equal to `transformers.get_linear_schedule_with_warmup` |
| Phase 2 steps, learning rates | `num_steps_afa`, policy head `lr_selector`, everything else `lr_backbone`, constant (`scheduler_afa: None`), temperature constant (`temp_decay: False`) | `n_steps`, `policy_lr`, `backbone_lr`, constant, `temperature` |
| Phase 2 start | The phase 1 checkpoint | The pretrained model bundle, which holds the phase 1 best state |
| Validation | After the training step when `step % checkpoint_interval == 0`, so at step 0; the last step only if it falls on the interval | The same |
| Best state | Lowest validation loss, strictly lower than before | The same, kept in memory and loaded when the fit ends |
| Validation tasks | `eval_batch_size` tasks generated once from `train_seed` | `n_validation_tasks` tasks generated once from the fit seed |
| Batch size | `batch_size` in phase 1; `batch_size_afa` in phase 2 of `real.py`, `batch_size` in `sim.py` | One `batch_size` per stage |
| Seeding | The model and task stream are not seeded | Model initialization and pool tasks follow the fit seed; TabICL batches are seeded per batch |
| Mixed precision | `Accelerator(mixed_precision="bf16")` | `UpstreamAccelerator`, a stand-in with the same autocast and fp32 outputs |

On a CPU, bf16 autocast is slow: a smoke-test fit stage takes minutes
unless `mixed_precision=false` is given.

### Parameters trained in phase 2

`real.py` with `freeze_encoder: False` trains every parameter, and so does
`train.py`. `sim.py`, the loop that the reference repository's `better_synth`
run uses, freezes the encoder and embedder in both branches of the flag and
trains the policy head and the predictor head only (the reference
repository's `KNOWN_ISSUES.md`, B5). That is the authors' own behaviour, not
a change of the reference repository. All three L2M runs train every
parameter, so a result of the reference repository's `better_synth` run is
not reproduced exactly by `l2m_better_synth`. This was the decision taken
for the thesis; a switch would restore the other behaviour.

## What the L2M runs do not reproduce of the evaluation

Inherited from the authors' training and kept:

- **Validation and training attend to labelled queries.** Training and
  validation run with `autoreg=True`, so query `i` also sees the true labels
  and features of the queries before it; prediction and the policy at
  evaluation run with `autoreg=False` (`KNOWN_ISSUES.md`, B1 and B2). The
  best state is chosen under the first regime.
- **States are random prefixes.** Phase 2 advances the state with a random
  feature, not the policy's choice (B3), and a checkpoint is judged on one
  acquisition step (B4).

Where training and evaluation differ because of the task prior and the
benchmark:

- **Context size.** Every training task has 50 context instances and
  `sequence_length - 50` queries that see up to that many labelled instances.
  At evaluation the context set is a draw from the validation split of at
  most `context_set_size` instances. `train_method` warns when a task is
  shorter than the context set plus 50.
- **Normalization.** A `real_pool` task is normalized over all its
  instances with its own statistics, a `synthetic` task is not normalized
  (uniform on [-2, 2]), and the TabICL prior normalizes its own datasets; at
  evaluation queries are normalized with the context set's.
- **Missingness.** The `real_pool` and `synthetic` task priors simulate
  missing feature values in the context and the queries; the TabICL prior
  has none; the context set is fully observed at evaluation.
- **Labels.** The `real_pool` and `synthetic` task priors label instances
  with a random function of the features, never the dataset's labels; the
  evaluated labels are real.
  The TabICL prior's classes differ in size, whereas the cube's are balanced,
  and its datasets are not the benchmark's.
- **Feature order.** The TabICL adapter keeps the prior's column order; the
  `real_pool` prior keeps the dataset's; the `synthetic` columns are alike.
- **Attention among labelled instances.** In training only the 50 context
  instances attend to each other in both directions; the labelled queries
  attend causally. At evaluation all context instances attend to each other.

The TabICL prior's own limits are in
[L2M trained on the TabICL v1 prior](l2m_tabicl_prior.md).

## The definition of L2M-synth

Upstream has no synthetic classification sampler. Its synthetic experiment
(`sim`) draws regression targets from a Gaussian process, which needs the
regression head and a Gaussian likelihood, whereas the benchmark's classifiers
are one-hot (CUBE, METABRIC) or one logit (MiniBooNE). Its only way to label
classification tasks is `sample_random_function`, which the real-data
experiments apply to the features of a pool.

`SyntheticTaskSampler` (`TaskPrior.synthetic`) therefore copies everything of
`sim`'s `GPWithMissingSampler.sample` except the labels: every task has new
features, uniform on [-2, 2]; every task and feature has a probability of
missing values drawn uniformly from [0, 0.5), applied independently to the
context and the queries; the queries' mask is built by the same
`num_observed` rule. The labels are `sample_random_function`'s, as in
`RealDataSampler`. Three differences remain: a seeded batch does not reseed
the global generators (upstream's does, which the reference repository's
`KNOWN_ISSUES.md` lists as C2); the features are not normalized within a task
(neither does `sim`, whereas `RealDataSampler` does); and the random function
labels features on [-2, 2], not a dataset's. Thresholding a Gaussian process
into classes would be a new labelling mechanism, not upstream's.

## What the context-coverage warning checks

**What a training task looks like.** A task has `sequence_length` instances.
The first 50 are its context: they carry labels and attend to each other in
both directions, as the context set does at evaluation. The remaining
`sequence_length - 50` instances are the queries. Each query is predicted
from the 50 context instances and from the queries before it, whose labels
the model may read (the attention mask of `autoreg=True`). Query `i` therefore
sees `50 + i` labelled instances, and the last one sees `sequence_length - 1`.
Those earlier queries are not full context: each of them attends only to the
context and to the queries before it, never to later ones.

**What evaluation looks like.** The method stores a context set of
`context_set_size` instances, or the whole validation split if that is
smaller, and every query instance reads all of it. All of it is context: its
instances attend to each other in both directions, and no instance of it is
predicted.

**What the warning compares.** It compares the number of labelled instances a
query reads at evaluation, the stored context set, with the most a training
query ever read beyond the 50 context instances, `sequence_length - 50`. If
the first is larger, evaluation shows the model more labelled instances than
any training query had, and the model has never been trained at that length.
With the CUBE's 600-instance tasks, a training query reads up to 549 further
instances and the stored context set has 200, so the warning stays quiet.
With a 200-instance context set but 200-instance tasks (150 further
instances), it would fire: evaluation would read 50 more than training ever
did. The same arithmetic holds for the other datasets:

| Dataset | `sequence_length` | Instances beyond the 50 context | Stored context set | Warning |
| --- | --- | --- | --- | --- |
| `cube` | 600 | 550 | 200 | no |
| `miniboone_subset` | 1000 | 950 | 900 | no |
| `metabric` | 500 | 450 | 379 | no |

**Why it matters, and what it cannot show.** A transformer that has read at
most 600 labelled instances is not guaranteed to behave when it reads
thousands; long contexts can change how attention spreads. The warning makes
a choice of sizes that goes beyond what training covered visible. It does
not address the other difference you named: only 50 instances are proper
context, and the labelled instances after them are visible only to later
queries, whereas at evaluation all stored instances are proper context that
attend to each other. A stored context set of 200 or 900 therefore has an
attention pattern that training never showed, and the warning cannot detect
or remove that; it is the "attention among labelled instances" difference
listed above, and it is inherent to the authors' training. The warning is a
check of lengths only. It warns rather than fails, because a shorter task is
a legitimate, cheaper choice, and only a comparison of results can say
whether it hurts.

## Open decisions and what is not verified

- **Phase-2 parameters** are decided as above (all parameters); revisit it
  to compare directly with the reference repository's `better_synth` result.
- **Task size.** The CUBE's tasks have 600 instances in every run, the most
  `l2m` can draw from the 600 train instances. The thesis repository's
  `better_synth` run used 650. Its notes give 6.8 GB at batch size 4 and
  12.8 GB at batch size 8 for 650 instances, 20 features and 8 classes in
  bf16 (not re-measured); the runs use batch size 8, and 4 would halve the
  memory.
- **Steps.** Pretraining takes 100,000 steps on every dataset, the thesis
  choice; the authors' are 100,000 on METABRIC and 50,000 on MiniBooNE, and
  the CUBE has none. Training takes 50,000 steps, the authors' value. None has
  been run on a GPU.
- **The authors' dataset numbers** (`sequence_length`, `n_steps`, pool size
  for MiniBooNE and METABRIC) are those already in the repository's configs;
  the authors' configs were not re-read for this page.
- **Not run:** nothing on CUDA, and no pipeline run of any L2M method on the
  Windows checkout where the runs were prepared (its jobs failed there:
  `python` did not resolve to the project environment). Both fit stages ran on a CPU for every run on
  every dataset (dataset realization 0) through the two scripts with
  `smoke_test=true mixed_precision=false`, and through the tests with
  bf16 autocast; tasks of the configured sizes were drawn on a CPU for every
  task prior and dataset, but no fit used them. The resource limits of the
  new runs in `workflow/profiles/site/arrhenius/site.yaml` are copied from
  the closest existing entry and not measured, and no GPU memory was
  measured at the configured task sizes.
- **The task rate of the TreeSCM prior** was measured once on one CPU (see the
  [reference](../reference/l2m.md#task-generation-rate)), not on the cluster.
