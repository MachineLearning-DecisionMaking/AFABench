# Add a dataset

Implement a dataset and include it in a pipeline profile.

## 1. Implement the dataset class

Create `src/afabench/datasets/<dataset_key>.py`, implementing `AFADataset`
from `src/afabench/core/types.py`, and export the class from
`src/afabench/datasets/__init__.py`.

Choose a starting point:

- In-memory tensors: subclass `InMemoryAFADataset` and implement `_build`,
  `feature_shape`, `label_shape`, `config_keys` and `accepts_seed`.
- A prepared CSV: subclass `TabularCSVDataset`; follow `DiabetesDataset`.
- Data that cannot fit in memory: implement the protocol directly; follow
  `ImagenetteDataset`.

For an in-memory dataset:

```python
from typing import ClassVar, final, override

import torch
import torch.nn.functional as F
from torch import Tensor

from afabench.datasets.in_memory import InMemoryAFADataset


@final
class MyDataset(InMemoryAFADataset):
    config_keys: ClassVar[tuple[str, ...]] = ("n_samples",)

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([5])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([3])

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        super().__init__()

    @override
    def _build(self) -> tuple[Tensor, Tensor]:
        features = torch.randn(self.n_samples, 5)
        labels = F.one_hot(
            torch.randint(low=0, high=3, size=(self.n_samples,)), num_classes=3
        ).float()
        return features, labels
```

For seeded synthetic generation, return `True` from `accepts_seed`, accept
`seed` in the constructor and include it in `config_keys`.

Keep real-world source order. Do not drop, sort or shuffle instances without
preserving their source mapping. If `_read_source` drops non-instance CSV
rows, retain the frame index; follow `MiniBooNEDataset`.
[Generation indices and source rows](../../CONTEXT.md#datasets).

For non-unit feature costs, either set `self.feature_costs` after
`super().__init__()`, or supply one value per feature in flattened order in
`data/<dataset_key>/feature_costs.csv`. Do not use both approaches.

## 2. Add generation config and register the class

Create `conf/scripts/dataset_generation/dataset/my_dataset.yaml`:

```yaml
class_name: "MyDataset"
kwargs:
  n_samples: 10000
```

Add the registry entry in `src/afabench/core/registry.py`:

```python
"MyDataset": "afabench.datasets.my_dataset.MyDataset",
```

Create `conf/components/dataset_key/my_dataset.yaml` by copying an existing
key file and replacing its dataset key.

## 3. Prepare CSV sources, if needed

For `TabularCSVDataset`:

1. Implement `make_prepared_source` to return prepared bytes. Pin and check
   the SHA-256 of every downloaded raw file; follow `MiniBooNEDataset` or
   `PhysionetChallenge2012Dataset`.
2. Pin the prepared file's SHA-256 as the class's `SHA256` and generate it:

   ```shell
   uv run python scripts/dataset_generation/prepare_source.py my_dataset
   ```

3. If redistribution is permitted, commit the file under `data/my_dataset/`
   with a `LICENSE.md` giving the licence and attribution. Otherwise, or
   without stated terms, set `SOURCE_COMMITTED = False` and gitignore the folder.

## 4. Add the dataset to your pipeline profile

Append `my_dataset` to the profile's `workflow/conf/datasets/<variant>.yaml`.
In the other selected configuration files:

- Set appropriate hard budgets in `eval_hard_budgets`.
- Choose a classifier and Unmasker where the defaults do not fit.
- For soft-budget runs, add the dataset to each applicable method's
  `soft_budget_datasets` and choose its `soft_budget_params` pairs.
- Add required method hyperparameter overrides under
  `conf/scripts/<stage>/<method>/dataset_override/my_dataset.yaml`.
  Use Imagenette's overrides as an image-data example.

[Pipeline configuration](../reference/pipeline_configuration.md).

## 5. Record redistribution policy

Add the key, including any variant keys, to
`conf/release/dataset_redistribution.yaml`. Copy an appropriate entry and fill
in the source, licence or terms, reviewer and notes.

Set `status` to `permitted`, `restricted` or `unreviewed`. Use
`bundle_policy: include` only with a permitted review; otherwise use `omit`
and an `omission_reason`. For synthetic data intended for local regeneration,
record that reason rather than a redistribution restriction.
[Policy fields](../reference/release_manifest.md#dataset-redistribution).

## 6. Add plot labels and documentation

Add a display name to `dataset_name_mapping` in
`conf/scripts/plotting/common/default.yaml`.
Optionally add the key to named `dataset_sets` in
`workflow/conf/dataset_sets/<variant>.yaml`; do not configure the automatic `all` set.

Write and index a dataset reference page using the
[dataset template](../reference/datasets/README.md).

## 7. Test the dataset

Add a small construction case to `dataset_kwargs` in
`test/src/common/datasets/test_generation_indices.py`. For CSV data, also add
it to `TABULAR_SOURCES` there.

Run the dataset tests and quality gate:

```shell
uv run pytest test/src/common/datasets/test_generation_indices.py \
    test/src/common/datasets/test_feature_costs.py \
    test/src/common/datasets/test_in_memory_dataset.py
uv run just qa
```

To evaluate release methods on it, continue with
[extend a release](manage_releases.md#extend-a-release).
