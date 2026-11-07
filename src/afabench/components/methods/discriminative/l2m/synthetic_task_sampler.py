"""
Tasks of upstream's simulation prior, labelled as its real-data tasks are.

Upstream's synthetic experiment (`sim`, `GPWithMissingSampler.sample` in
`src/data/data.py` of the authors' repository) draws every task's features
anew, uniformly on [-2, 2], with independent missing values in the context
and the queries. Its labels are Gaussian-process regression targets, which
need a regression head; this benchmark's datasets are classification
datasets. `SyntheticTaskSampler` keeps the simulation's features and
missingness and takes the labels from `sample_random_function`, the only
way upstream labels classification tasks (`RealDataSampler`).

Unlike upstream, a seeded batch does not reseed the global generators.
"""

from typing import override

import torch

from afabench.components.methods.discriminative.l2m.config import (
    UPSTREAM_CONTEXT_SET_SIZE,
)
from afabench.components.methods.discriminative.l2m.models import UpstreamBatch
from afabench.components.methods.discriminative.l2m.upstream.data_utils import (
    RealDataSampler,
    TabularDataset,
    sample_random_function,
)

# Upstream's `x_range` and `max_p_missing` defaults.
FEATURE_LOW = -2.0
FEATURE_HIGH = 2.0
MAX_MISSING_PROBABILITY = 0.5


class SyntheticTaskSampler(RealDataSampler):
    """New uniform features for every task, labelled by upstream's random BNN."""

    def __init__(self, *, n_features: int, label_dim: int) -> None:
        if n_features < 1:
            msg = f"n_features={n_features} must be at least 1"
            raise ValueError(msg)
        dataset = TabularDataset(
            data=torch.empty(0, n_features),
            dim_y=label_dim,
            feature_names=[f"feature_{i}" for i in range(n_features)],
            dataframe=None,
            free_indices=[],
        )
        super().__init__(dataset, dataset)
        self.n_features: int = n_features

    @override
    def sample(
        self,
        batch_size: int = 16,
        num_ctx: int | None = None,
        num_tar: int | None = None,
        max_num_points: int = 1000,
        device: str | torch.device = "cpu",
        num_observed: int | None = None,
        seed: int | None = None,
        # The signature of `RealDataSampler.sample`.
        return_groups: bool = False,
    ) -> UpstreamBatch:
        """
        Draw `batch_size` tasks of `max_num_points` instances.

        The first `num_ctx` (default 50) instances of a task are its
        context. With a `seed` the batch is reproducible.
        """
        if return_groups:
            msg = "The synthetic task prior has no feature groups"
            raise NotImplementedError(msg)
        num_ctx = num_ctx or UPSTREAM_CONTEXT_SET_SIZE
        n_rows = num_ctx + (
            max_num_points - num_ctx if num_tar is None else num_tar
        )
        generator = (
            None if seed is None else torch.Generator().manual_seed(seed)
        )
        shape = (batch_size, n_rows, self.n_features)

        x = FEATURE_LOW + (FEATURE_HIGH - FEATURE_LOW) * torch.rand(
            shape, generator=generator
        )
        # Per task and feature, values are missing with a probability drawn
        # from [0, MAX_MISSING_PROBABILITY), in the context and the queries.
        missing_probabilities = (
            torch.rand(batch_size, 1, self.n_features, generator=generator)
            * MAX_MISSING_PROBABILITY
        )
        r = (
            torch.rand(shape, generator=generator) > missing_probabilities
        ).float()
        mask = torch.ones_like(x)
        mask[:, num_ctx:] = self._generate_missingness(
            x[:, num_ctx:], num_observed=num_observed, seed=seed
        )
        # The vendored function annotates its optional seed as an int.
        y = sample_random_function(
            x,
            output_dim=self.dim_y,
            seed=seed,  # pyright: ignore[reportArgumentType]
            label=True,
        )

        batch = {
            "x": x.to(device),
            "y": y.to(device),
            "mask": mask.to(device),
            "r": r.to(device),
        }
        for name, values in list(batch.items()):
            batch[f"{name}c"] = values[:, :num_ctx]
            batch[f"{name}t"] = values[:, num_ctx:]
        return batch
