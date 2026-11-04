---
status: accepted
---

# The TabICL v1 prior is vendored with two opt-in edits that change what it computes

L2M's tasks come from a task prior ([CONTEXT.md](../../CONTEXT.md)). The
pool task priors draw a task's instances from the train split's features
and label them with upstream's random BNN, so the tasks resemble the
dataset only through its features. The `l2m_tabicl_prior` variant instead
trains on tasks from the TabICL v1 prior, which generates whole
classification datasets, features and labels, and shares nothing with the
benchmark's datasets. Its tasks must have the model's exact feature count
and class count, since L2M's model width is fixed by them.

We vendored the prior into
`src/afabench/components/methods/discriminative/l2m/tabicl_v1_prior/`
under [ADR 0012](0012-vendored-method-code.md): copied close to verbatim,
every change marked, excluded from ruff and basedpyright, adapted at the
boundary (`tabicl_task_sampler.py`, written to the repository's
standards). Two of its edits break ADR 0012's rule that a patch may not
change what is computed: `fixed_num_classes` and the `fixed` feature
sampler, both marked `L2M CHANGE`. They are opt-in; without the config
entries the prior computes what the original does.

- **Why the prior must change, not only be filtered.** Left alone, the prior
  draws 1 to 10 features, so about 4 % of its datasets have 20, and 2 to 10
  classes, with 70 % binary and 6 % with exactly 8. Rejecting the rest
  would waste almost every draw. The edits make the prior ask for the sizes
  instead, and the adapter still checks every dataset and redraws one that
  does not match, so the guarantee does not rest on the edits alone.
- **Everything else is mechanical.** The other edits are import paths, the
  removal of a `sys.path` change, and a reduction of Mother-Tree's
  `utils_torch.py` to the four functions the prior imports and the helpers
  they call, so the prior needs neither `wandb` nor Mother-Tree's training
  code. The folder's README lists every edit.

## Considered options

- **Rejection sampling on the unmodified prior.** Rejected for the reason
  above.
- **Padding features and masking classes.** Rejected: the cube has 20
  features and 8 classes, and tasks with fewer would train the model on a
  different problem than the one it is evaluated on. Zero-padded columns
  would also have to be excluded from acquisition.
- **Keeping the edits in the adapter by subclassing the prior's classes.**
  Rejected: both edits sit inside methods (`ClassificationAdapter.__init__`
  and `__call__`) that draw random numbers in sequence, so a subclass would
  have to copy them.

## Consequences

- Diffing the folder against the original prior shows the `PATCH(afabench)`
  and `L2M CHANGE` edits and nothing else. Updating the prior means copying
  it again and reapplying them.
- A new dependency, `gpytorch` (with `linear-operator`), both pure-Python
  wheels, so the aarch64 environment needs no build. `scipy` is now
  declared; it was already installed through scikit-learn.
- The prior draws from the global `random`, numpy and torch generators, so
  the adapter seeds them per batch and restores them. A batch's content then
  depends only on its seed, not on which process generates it.
