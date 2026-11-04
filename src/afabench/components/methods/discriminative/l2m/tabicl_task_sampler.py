"""
Tasks from the TabICL v1 task prior, in the layout of upstream's sampler.

The prior (the vendored `tabicl_v1_prior` package) generates a whole
classification dataset per call: features and labels. `TabICLTaskSampler`
replaces `RealDataSampler`'s task generator with it and inherits its
masking helpers, which the vendored training steps call. Every task has
exactly the model's feature count and class count; the prior is asked for
them, and each dataset is checked and redrawn if it does not match.

The prior draws from the global `random`, numpy and torch generators, so
every batch is generated under its own seed (`seeded_generators`). A batch's
content then depends only on its seed, not on which process generates it
or on what else uses the global generators.
"""

import logging
import random
from collections import Counter
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast, override

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from numpy.random import SeedSequence
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from afabench.components.methods.discriminative.l2m.config import (
    UPSTREAM_CONTEXT_SET_SIZE,
)
from afabench.components.methods.discriminative.l2m.models import UpstreamBatch
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.data_prior_config import (
    get_prior_config,
)
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.tabiclv1.classification_adapter import (
    ClassificationAdapterPrior,
)
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.tabiclv1.fast_gp import (
    GPPrior,
)
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.tabiclv1.mlp import (
    MLPPrior,
)
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.tabiclv1.prior_bag import (
    BagPrior,
)
from afabench.components.methods.discriminative.l2m.upstream.data_utils import (
    RealDataSampler,
    TabularDataset,
)

log = logging.getLogger(__name__)

type TaskFeatures = Float[Tensor, "rows features"]
type TaskLabels = Float[Tensor, "rows label_dim"]

# The sub-priors' mixture of the TabICL v1 data loader (`generate_one_batch`):
# an MLP structural causal model or a Gaussian process.
_PRIOR_WEIGHTS = {"mlp": 0.961, "gp": 0.039}
# The prior's own marker of a dataset without a usable train/test class
# overlap, which it emits instead of redrawing.
_INVALID_LABEL = -100
# Redraws per task before giving up; about a fifth of the draws are rejected.
_MAX_DRAWS_PER_TASK = 100


class TabICLTaskSampler(RealDataSampler):
    """
    Upstream's task sampler with the TabICL v1 prior as the task generator.

    Tasks are fully observed (`r` is 1): the prior has no missing values, and
    the context set is fully observed at evaluation. `RealDataSampler`
    simulates missingness on its pool's features instead.
    """

    def __init__(
        self, *, n_features: int, n_classes: int, label_dim: int
    ) -> None:
        dataset = TabularDataset(
            data=torch.empty(0, n_features),
            dim_y=label_dim,
            feature_names=[f"feature_{i}" for i in range(n_features)],
            dataframe=None,
            free_indices=[],
        )
        super().__init__(dataset, dataset)
        self.n_features: int = n_features
        self.n_classes: int = n_classes
        self.prior: BagPrior = _build_prior(n_features, n_classes)
        self.rejections: Counter[str] = Counter()

    def draw_task(
        self, n_rows: int, n_context: int
    ) -> tuple[TaskFeatures, TaskLabels]:
        """Draw one dataset from the prior, redrawing until it fits."""
        for _ in range(_MAX_DRAWS_PER_TASK):
            try:
                features, labels, _ = self.prior.get_batch(
                    batch_size=1,
                    n_samples=n_rows,
                    num_features=self.n_features,
                    device=torch.device("cpu"),
                    single_eval_pos=n_context,
                )
            except (AssertionError, ValueError):
                # The failures the prior's own data loader retries on.
                self.rejections["failed"] += 1
                continue
            # (rows, batch, features) and (rows, batch) with one dataset.
            features, labels = features[:, 0], labels[:, 0]
            reason = self._rejection_reason(features, labels)
            if reason is None:
                return features, self._label_columns(labels)
            self.rejections[reason] += 1
        msg = (
            f"No valid dataset after {_MAX_DRAWS_PER_TASK} draws from the "
            f"TabICL v1 prior; rejections: {dict(self.rejections)}"
        )
        raise RuntimeError(msg)

    def _rejection_reason(
        self, features: Tensor, labels: Tensor
    ) -> str | None:
        if (labels == _INVALID_LABEL).any():
            return "invalid"
        if not (features.isfinite().all() and labels.isfinite().all()):
            return "not_finite"
        if features.shape[1] != self.n_features:
            return "wrong_features"
        # Exactly the model's classes, with ids 0..n_classes - 1: the prior
        # can draw coinciding class boundaries or leave a class empty.
        classes = torch.unique(labels)
        if not torch.equal(
            classes, torch.arange(self.n_classes, dtype=classes.dtype)
        ):
            return "wrong_classes"
        return None

    def _label_columns(self, labels: Tensor) -> TaskLabels:
        """Upstream's label encoding: class 1's column if binary."""
        one_hot = F.one_hot(labels.long(), self.n_classes).float()
        return one_hot[:, 1:] if self.dim_y == 1 else one_hot

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

        As `RealDataSampler.sample`, the first `num_ctx` (default 50)
        instances of a task are its context. With a `seed` the batch is
        reproducible and the global generators are left as they were.
        """
        if return_groups:
            msg = "The TabICL v1 task prior has no feature groups"
            raise NotImplementedError(msg)
        num_ctx = num_ctx or UPSTREAM_CONTEXT_SET_SIZE
        n_rows = num_ctx + (
            max_num_points - num_ctx if num_tar is None else num_tar
        )
        with seeded_generators(seed):
            tasks = [
                self.draw_task(n_rows, num_ctx) for _ in range(batch_size)
            ]
            x = torch.stack([features for features, _ in tasks]).to(device)
            y = torch.stack([labels for _, labels in tasks]).to(device)
            mask = torch.ones_like(x)
            mask[:, num_ctx:] = self._generate_missingness(
                x[:, num_ctx:], num_observed=num_observed
            )
        batch = {"x": x, "y": y, "mask": mask, "r": torch.ones_like(x)}
        for name, values in list(batch.items()):
            batch[f"{name}c"] = values[:, :num_ctx]
            batch[f"{name}t"] = values[:, num_ctx:]
        return batch


def _build_prior(n_features: int, n_classes: int) -> BagPrior:
    """
    Build the prior of TabICL v1's `generate_one_batch`, with fixed sizes.

    Padding is off: every dataset has exactly `n_features` features, which
    the vendored `fixed` feature sampler and `fixed_num_classes` option
    (the two `L2M CHANGE` edits of the prior) request.
    """
    config: dict[str, Any] = cast(
        "dict[str, Any]",
        get_prior_config(mlp_prior_type="tabiclv1", max_features=n_features),
    )["prior"]
    config["num_features"] = n_features
    classification: dict[str, Any] = {
        **config["classification"],
        "num_features_sampler": "fixed",
        "fixed_num_classes": n_classes,
    }
    base_priors: dict[str, Any] = {}
    for name, prior_class in (("mlp", MLPPrior), ("gp", GPPrior)):
        base_priors[name] = ClassificationAdapterPrior(
            prior_class(config[name], batch_size_per_subgroup=1),
            num_features=n_features,
            padding=False,
            **classification,
        )
    return BagPrior(base_priors=base_priors, prior_weights=_PRIOR_WEIGHTS)


@contextmanager
def seeded_generators(seed: int | None) -> Generator[None]:
    """
    Seed the global generators the prior draws from, then restore them.

    Python's `random`, numpy's and torch's CPU generator. Without a seed
    they are used as they are.
    """
    if seed is None:
        yield
        return
    # The prior draws from numpy's legacy global generator.
    python_state, numpy_state = random.getstate(), np.random.get_state()  # noqa: NPY002
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)  # noqa: NPY002
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)  # noqa: NPY002


def batch_seed(base_seed: int, stream: int, index: int) -> int:
    """Seed of the `index`-th batch of a stream of the fit seeded `base_seed`."""
    return int(SeedSequence([base_seed, stream, index]).generate_state(1)[0])


class _BatchStream(Dataset[UpstreamBatch]):
    """The fit's training batches by index, each under its own seed."""

    def __init__(
        self,
        sampler: TabICLTaskSampler,
        *,
        n_batches: int,
        batch_size: int,
        sequence_length: int,
        num_observed: int,
        seed: int,
        stream: int,
    ) -> None:
        self.sampler: TabICLTaskSampler = sampler
        self.n_batches: int = n_batches
        self.batch_size: int = batch_size
        self.sequence_length: int = sequence_length
        self.num_observed: int = num_observed
        self.seed: int = seed
        self.stream: int = stream

    def __len__(self) -> int:
        return self.n_batches

    @override
    def __getitem__(self, index: int) -> UpstreamBatch:
        return self.sampler.sample(
            batch_size=self.batch_size,
            max_num_points=self.sequence_length,
            num_observed=self.num_observed,
            seed=batch_seed(self.seed, self.stream, index),
        )


@contextmanager
def task_batches(
    sampler: TabICLTaskSampler,
    *,
    n_batches: int,
    batch_size: int,
    sequence_length: int,
    num_observed: int,
    seed: int,
    stream: int,
    device: torch.device,
    n_workers: int,
) -> Generator[Iterator[UpstreamBatch]]:
    """
    Generate the training batches in `n_workers` processes ahead of use.

    Batch `i` is seeded from `seed`, `stream` and `i` alone, so the batches
    do not depend on `n_workers`, up to floating-point rounding that torch's
    thread count changes (a worker uses one thread). Zero workers generate
    them in this process, on demand. The workers end when the block does.
    """
    loader = DataLoader(
        _BatchStream(
            sampler,
            n_batches=n_batches,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_observed=num_observed,
            seed=seed,
            stream=stream,
        ),
        batch_size=None,
        num_workers=n_workers,
        # Two batches per worker queued: more only fills memory.
        prefetch_factor=2 if n_workers > 0 else None,
    )

    def on_device() -> Generator[UpstreamBatch]:
        for batch in loader:
            yield {name: value.to(device) for name, value in batch.items()}

    batches = on_device()
    try:
        yield batches
    finally:
        # Ends the loader's iterator, whose worker processes stop with it.
        batches.close()
