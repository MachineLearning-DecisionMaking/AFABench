# Vendored TabICL v1 prior

The synthetic-dataset prior of TabICL v1 (a bag of an MLP structural causal
model, 96.1 %, and a Gaussian process, 3.9 %, behind a classification
adapter), as the Mother-Tree repository carries it. It reached this
repository from the author's thesis repository `lerning-to-measure` (branch
`dev`, commit `cc161a7`), which holds the only copy that was verified with
these edits; that repository is not public. See
`docs/adr/0017-tabicl-prior-vendored-with-opt-in-edits.md` for why it is
here and why two edits change what it computes, and
`docs/adr/0012-vendored-method-code.md` for how vendored code is kept.

**License: not recorded.** The files came without a license file, so the
license of the Mother-Tree and TabICL sources, and the attribution they
require, have to be confirmed before this folder is published. Add the
license text here when it is known.

The adapter that uses the prior is `../tabicl_task_sampler.py`.

| File here | Origin |
| --- | --- |
| `activation.py` | `prior/activation.py` |
| `data_prior_config.py` | `prior/data_prior_config.py` |
| `tabiclv1/classification_adapter.py` | `prior/tabiclv1/classification_adapter.py` |
| `tabiclv1/distributions.py` | `prior/tabiclv1/distributions.py` |
| `tabiclv1/fast_gp.py` | `prior/tabiclv1/fast_gp.py` |
| `tabiclv1/mlp.py` | `prior/tabiclv1/mlp.py` |
| `tabiclv1/prior_bag.py` | `prior/tabiclv1/prior_bag.py` |
| `tabiclv1/treeSCM.py` | `prior/tabiclv1/treeSCM.py` |
| `tabiclv1/utils.py` | `prior/tabiclv1/utils.py` |
| `utils_torch.py` | `model/utils_torch.py` of Mother-Tree, reduced |

The origin's TabICL v2 prior, data loader and generate scripts are not
here. `tabiclv1/treeSCM.py` (the origin's `prior/tabiclv1/treeSCM.py`, from
the thesis repository's commit `3f5d546`) is the optional third sub-prior
of the adapter's `prior_type` mixtures (`prior_bag_plus_tree`, `tree_only`).

Every change is marked in place, with `# PATCH(afabench)` for the
environment and `# L2M CHANGE` for the two that alter what the prior
computes:

- **Import paths.** `from prior.activation import *` and
  `from model.utils_torch import ...` point at this package
  (`data_prior_config.py`, `tabiclv1/classification_adapter.py`,
  `tabiclv1/fast_gp.py`, `tabiclv1/mlp.py`). `data_prior_config.py` no longer
  appends its parent directory to `sys.path`, which put this repository's
  `l2m/` folder on the import path.
- **`utils_torch.py`.** Only `default_device`, `get_nan_value`,
  `normalize_data`, `remove_outliers` and the helpers they call
  (`torch_masked_mean`, `torch_masked_std`, `torch_nanmean`, `torch_nanstd`)
  remain, copied verbatim; the rest of the module imports `wandb` and
  `tqdm` and holds training code the prior does not use. `default_device`
  is only the default of `get_batch`'s device argument; the adapter always
  passes the CPU.
- **`from typing import List`** was added to `data_prior_config.py` in the
  thesis repository, which the prior's star import from `activation.py`
  provided implicitly before unused imports were removed there.
- **`L2M CHANGE` (opt-in).** `ClassificationAdapter.__init__` uses
  `fixed_num_classes`, when the config has it, instead of the random class
  count (otherwise 2 in half of the datasets and uniform on 2 to 10 in the
  others). `ClassificationAdapter.__call__` accepts
  `num_features_sampler: 'fixed'`, which uses all `num_features` instead of
  drawing from 1 to `num_features - 1`. Without the config entries the
  prior computes what the original does. Their comments name the adapter
  here instead of the thesis repository's file.
- **`tabiclv1/treeSCM.py`** (`PATCH(afabench)`, three edits): `from
  model.utils_torch import default_device` points at this package's
  `utils_torch.py`; `from xgboost import XGBRegressor` sits in the
  `tree_model == "xgboost"` branch of `TreeLayer` instead of at module level
  (the original's import was commented out although xgboost is the default
  tree model, so every TreeSCM raised `NameError`), which also lets runs
  without TreeSCM do without the package; and every tree model's `n_jobs=-1`
  (every CPU) became the module constant `N_JOBS`, which stays -1 until the
  adapter sets it to 1. The module imports scikit-learn at module level, so
  the adapter imports it only when a mixture contains TreeSCM.
- **Not corrected:** `data_prior_config.py` configures TreeSCM with
  `tree_depth_lambda` and `tree_n_estimators_lambda`, but `TreeSCM.__init__`
  reads `max_depth_lambda` and `n_estimators_lambda`. The configured values
  fall into its `**kwargs` and the defaults (0.5) apply, as in the original.
- **Empty `__init__.py`** files make the folders packages.

Importing the prior prints `SyntaxWarning: invalid escape sequence`
messages from docstrings in `activation.py` the first time Python compiles
it. They are harmless, as is gpytorch's "A not p.d., added jitter"
`NumericalWarning` from the GP sub-prior.
