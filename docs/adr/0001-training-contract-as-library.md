---
status: accepted
---

# The training contract is a library, not a framework

Every pretraining and training script re-declared and re-handled the same
pipeline inputs, and the copies drifted into bugs (see
[issue #43](https://github.com/MachineLearning-DecisionMaking/AFABench/issues/43)). We fix this by defining a
**training contract**, the fixed inputs the pipeline passes and the bundle it
expects back at `save_path`, and by offering optional helpers that scripts
call. We do not dispatch all methods through one script. The only hard rule
for a method is: accept the training contract on the command line and write a
bundle to `save_path`. How the method trains, configures itself or uses the
dataset key is the method's business.

## Considered options

- **Framework: one dispatching script.** One training script and one
  pretraining script resolve a per-method `train(ctx, cfg)` function (or a
  registered `Trainer` class) by method name, and own seeding, logging,
  smoke-test overrides, hyperparameter resolution and saving. This removes the
  most repetition and gives every method an end-to-end test for free, but the
  runner calls the method. Authors would lose Hydra for method configs, would
  have to use a prescribed hyperparameter layout, and could not control the
  training lifecycle. Rejected because developer freedom over how a method
  trains matters more than the remaining boilerplate.
- **Library: scripts call shared helpers.** Chosen. It fixes the drift and
  the bugs, and adds a contract test, while each method keeps its own script.

## Decisions

- **Command-line form.** The contract uses plain `key=value` arguments that
  any argument parser can read: dataset bundle paths, classifier and
  pretrained-model bundle paths, `save_path`, `initializer`, `unmasker`,
  `dataset_key`, `hard_budget`, `soft_budget_param`, `device`, `seed`,
  `use_wandb`, `smoke_test`. Hydra-specific forms (`components/...@initializer`,
  `experiment@_global_`) are no longer part of the contract. The
  pretraining contract is a sibling of the training contract under a shared
  base, not a subset of it: it currently omits the pretrained-model path and
  budgets, but the two may diverge further.
- **Helpers, all optional:** `PretrainingContract` / `TrainingContract`
  dataclasses (siblings under `BaseContract`) that method configs inherit (keyword-only, so the command line
  stays flat); `load_inputs(contract)` for lazily loaded datasets, Initializer,
  Unmasker, classifier and pretrained model; a `fit_run(...)` context
  manager for seeding, metric logger (wandb or null) and cleanup; and
  `save_result(...)` for one bundle-metadata shape. The stage is a property
  of the contract class, not an argument to these helpers.
- **One copy of the contract's field names:** the Python dataclasses. The
  Snakemake argument renderer reads them with `dataclasses.fields`, and a
  field that defaults to null may be left off the command line. This first
  kept a second copy in the renderer on the mistaken premise that Snakemake
  could not import `afabench`; it can, since it runs in the project's
  environment. Importing the contract must not load torch, because Snakemake
  does it on every parse; a unit test enforces this, and the conformance test
  checks the rendered arguments end to end.
- **Per-dataset hyperparameters stay with each method.** The repo's own
  methods keep Hydra dataset overrides, deduplicated through shared
  family files. Dataset overrides may not set contract fields, and a test
  enforces this.
- **Conformance test.** Every method script is run as a subprocess on a
  generated smoke dataset with contract arguments and must produce a loadable
  bundle at `save_path`. Marked `pipeline`, with an unmarked `random_dummy`
  case.
- **Migration bar.** Ports must keep the resolved hyperparameters of every
  method × dataset key identical. They need not reproduce bit-identical
  bundles, because moving `set_seed` may change RNG order.
