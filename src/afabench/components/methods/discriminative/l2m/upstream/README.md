# Vendored Learning-To-Measure code

Copied from the authors' repository,
<https://github.com/reAIM-Lab/Learning-To-Measure>, at commit
`9cb464382f120751cd978ddd1f8ccc40604f887e`, under the MIT license in
`LICENSE`. See `docs/adr/0012-vendored-method-code.md` for why this folder is
kept close to verbatim and exempt from the repository's lint and type rules.

| File here | Upstream file |
| --- | --- |
| `ltm.py` | `src/models/ltm.py` |
| `model_utils.py` | `src/utils/model_utils.py` |
| `train_utils.py` | `src/utils/train_utils.py` |
| `experiment_utils.py` | `src/utils/experiment_utils.py` |
| `utils.py` | `src/utils/utils.py` |
| `data_utils.py` | `src/data/data_utils.py` |

Every change is marked in place with a `# PATCH(afabench)` comment:

- Import paths point at this package (`ltm.py`, `model_utils.py`,
  `experiment_utils.py`).
- `TNP.create_mask` builds its attention mask on the batch's device rather
  than `'cuda'` (`ltm.py`).
- `gen_evalset` uses `config['device']` rather than `'cuda'` and returns the
  validation tasks instead of caching them on disk; `eval` and `eval_afa`
  read them from `config['eval_sets']` (`experiment_utils.py`).
- Without a seed, `sample_random_function` draws its generator's seed from
  torch's global generator rather than calling `torch.seed()`, which reseeds
  the global generator nondeterministically. The distribution is unchanged;
  fits become reproducible from their seed (`data_utils.py`).
- The seeded multiclass label prior of `sample_random_function` forks the
  generators of every visible CUDA device (`fork_rng()`), not
  `devices=[x.device]`: for features on the CPU that forked the current
  CUDA device, the same with one GPU per job, and failed without CUDA
  (`data_utils.py`).
- `predictor_training_step` and `afa_training_step` clip the gradient's total
  norm to `config['max_grad_norm']` before the optimizer step when that key is
  set to a number (`train_utils.py`). Without it, or with `None`, the steps
  are upstream's.
- Code L2M does not use, and whose imports are not dependencies of this
  repository, is removed: the dataset loaders of `data_utils.py` (pandas,
  polars, scikit-learn) and the baseline models of `utils.py` (PyTorch
  Lightning).

What the adapter in the parent package replaces, rather than patches:

- `accelerate.Accelerator(mixed_precision="bf16")` becomes a minimal stand-in
  with the same `device`, `autocast`, `backward` and `clip_grad_norm_`, and a forward pass
  wrapped in bf16 autocast as `Accelerator.prepare` does.
- `transformers.get_linear_schedule_with_warmup` becomes an equivalent
  `LambdaLR`.
- The fit loops of `src/experiments/real.py` (`real_train`,
  `real_train_afa`) are rewritten around the fit contract, keeping their
  step, validation and checkpoint logic.
