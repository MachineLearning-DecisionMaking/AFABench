"""The TabICL v1 task prior as L2M's task generator."""

import random

import numpy as np
import pytest
import torch

from afabench.components.methods.discriminative.l2m.config import (
    TaskPrior,
)
from afabench.components.methods.discriminative.l2m.tabicl_task_sampler import (
    TabICLTaskSampler,
    batch_seed,
    task_batches,
)

N_FEATURES = 20
N_CLASSES = 8
# The cube-sized setting: 50 context instances and 200 query instances.
SEQUENCE_LENGTH = 250
CONTEXT_SIZE = 50
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def sampler() -> TabICLTaskSampler:
    return TabICLTaskSampler(
        n_features=N_FEATURES, n_classes=N_CLASSES, label_dim=N_CLASSES
    )


def _sample(
    sampler: TabICLTaskSampler, seed: int, batch_size: int = 4
) -> dict[str, torch.Tensor]:
    return sampler.sample(
        batch_size=batch_size,
        max_num_points=SEQUENCE_LENGTH,
        num_observed=N_FEATURES,
        seed=seed,
    )


def test_every_task_has_exactly_the_models_features_and_classes(
    sampler: TabICLTaskSampler,
) -> None:
    batch = _sample(sampler, seed=0, batch_size=8)

    assert batch["x"].shape == (8, SEQUENCE_LENGTH, N_FEATURES)
    assert batch["y"].shape == (8, SEQUENCE_LENGTH, N_CLASSES)
    assert batch["x"].isfinite().all()
    # One-hot labels, with every class present in every task.
    assert torch.equal(batch["y"].sum(dim=-1), torch.ones(8, SEQUENCE_LENGTH))
    for labels in batch["y"].argmax(dim=-1):
        assert labels.unique().numel() == N_CLASSES


def test_a_batch_splits_into_context_and_query_instances(
    sampler: TabICLTaskSampler,
) -> None:
    batch = _sample(sampler, seed=1)

    for name in ("x", "y", "mask", "r"):
        assert torch.equal(batch[f"{name}c"], batch[name][:, :CONTEXT_SIZE])
        assert torch.equal(batch[f"{name}t"], batch[name][:, CONTEXT_SIZE:])
    # Fully observed: nothing is unrecorded, and the queries start with all
    # features acquired as the fit asks for.
    for name in ("mask", "r"):
        assert torch.equal(batch[name], torch.ones_like(batch["x"]))


def test_binary_tasks_use_the_upstream_label_column() -> None:
    binary = TabICLTaskSampler(n_features=3, n_classes=2, label_dim=1)

    batch = binary.sample(
        batch_size=2, max_num_points=120, num_observed=3, seed=0
    )

    assert batch["y"].shape == (2, 120, 1)
    assert set(batch["y"].unique().tolist()) == {0.0, 1.0}


def test_a_seeded_batch_is_reproducible(sampler: TabICLTaskSampler) -> None:
    first = _sample(sampler, seed=3)

    assert torch.equal(first["x"], _sample(sampler, seed=3)["x"])
    assert torch.equal(first["y"], _sample(sampler, seed=3)["y"])
    assert not torch.equal(first["x"], _sample(sampler, seed=4)["x"])


def test_a_seeded_batch_leaves_the_global_generators_alone(
    sampler: TabICLTaskSampler,
) -> None:
    random.seed(10)
    np.random.seed(10)  # noqa: NPY002
    torch.manual_seed(10)
    states = (
        random.getstate(),
        np.random.get_state(),  # noqa: NPY002
        torch.get_rng_state(),
    )

    _sample(sampler, seed=5)

    assert random.getstate() == states[0]
    assert (
        np.random.get_state()[1].tolist()  # noqa: NPY002
        == states[1][1].tolist()
    )
    assert torch.equal(torch.get_rng_state(), states[2])


class _ScriptedPrior:
    """A prior that returns the given datasets in turn."""

    def __init__(self, *datasets: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.datasets = list(datasets)
        self.n_calls = 0

    def get_batch(
        self, **_: object
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        features, labels = self.datasets[
            min(self.n_calls, len(self.datasets) - 1)
        ]
        self.n_calls += 1
        return features.unsqueeze(1), labels.unsqueeze(1), None


def _dataset(
    n_features: int, labels: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(len(labels), n_features), torch.tensor(
        labels, dtype=torch.float
    )


@pytest.mark.parametrize(
    ("bad_dataset", "reason"),
    [
        (_dataset(N_FEATURES - 1, [0, 1, 2, 3] * 3), "wrong_features"),
        (_dataset(N_FEATURES, [0, 1, 2] * 4), "wrong_classes"),
        # A skipped class id: three classes, but not 0, 1, 2.
        (_dataset(N_FEATURES, [0, 1, 3] * 4), "wrong_classes"),
        (_dataset(N_FEATURES, [0, 1, 2, -100] * 3), "invalid"),
        (
            (
                torch.full((12, N_FEATURES), torch.nan),
                torch.tensor([0.0, 1.0, 2.0, 3.0] * 3),
            ),
            "not_finite",
        ),
    ],
)
def test_a_dataset_of_the_wrong_size_is_redrawn(
    bad_dataset: tuple[torch.Tensor, torch.Tensor], reason: str
) -> None:
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES, n_classes=4, label_dim=4
    )
    good = _dataset(N_FEATURES, [0, 1, 2, 3] * 3)
    sampler.prior = _ScriptedPrior(bad_dataset, good)  # pyright: ignore[reportAttributeAccessIssue]

    features, labels = sampler.draw_task(12, 5)

    assert features.shape == (12, N_FEATURES)
    assert labels.shape == (12, 4)
    assert sampler.prior.n_calls == 2  # pyright: ignore[reportAttributeAccessIssue]
    assert dict(sampler.rejections) == {reason: 1}


def test_a_prior_that_never_fits_fails_loudly() -> None:
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES, n_classes=4, label_dim=4
    )
    sampler.prior = _ScriptedPrior(_dataset(N_FEATURES - 1, [0, 1, 2, 3] * 3))  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(RuntimeError, match="wrong_features"):
        sampler.draw_task(12, 5)


# Worker processes start slowly where processes are spawned, not forked.
@pytest.mark.optional
def test_batches_do_not_depend_on_the_number_of_workers(
    sampler: TabICLTaskSampler,
) -> None:
    def batches(n_workers: int) -> list[dict[str, torch.Tensor]]:
        with task_batches(
            sampler,
            n_batches=3,
            batch_size=2,
            sequence_length=SEQUENCE_LENGTH,
            num_observed=N_FEATURES,
            seed=7,
            stream=0,
            device=CPU,
            n_workers=n_workers,
        ) as stream:
            return list(stream)

    in_process = batches(0)

    assert len(in_process) == 3
    assert not torch.equal(in_process[0]["x"], in_process[1]["x"])
    assert torch.equal(in_process[1]["x"], batches(0)[1]["x"])
    # A worker computes with one thread, which rounds differently.
    for expected, drawn in zip(in_process, batches(2), strict=True):
        torch.testing.assert_close(
            expected["x"], drawn["x"], atol=1e-4, rtol=0
        )
        assert torch.equal(expected["y"], drawn["y"])


def test_streams_and_batches_have_distinct_seeds() -> None:
    seeds: set[int] = {
        batch_seed(seed, stream, index)
        for seed in (0, 1)
        for stream in (0, 1)
        for index in range(100)
    }

    assert len(seeds) == 400


def test_only_the_tabicl_prior_has_no_pool() -> None:
    assert {prior: prior.uses_pool for prior in TaskPrior} == {
        TaskPrior.real_pool: True,
        TaskPrior.synthetic_pool: True,
        TaskPrior.tabicl_v1: False,
    }
