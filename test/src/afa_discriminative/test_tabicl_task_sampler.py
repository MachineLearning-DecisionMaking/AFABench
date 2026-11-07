"""The TabICL v1 task prior as L2M's task generator."""

import random
import sys

import numpy as np
import pytest
import torch

from afabench.components.methods.discriminative.l2m.config import (
    TabICLPriorType,
    TaskPrior,
)
from afabench.components.methods.discriminative.l2m.tabicl_task_sampler import (
    MAX_PRIOR_CLASSES,
    TabICLTaskSampler,
    batch_seed,
    task_batches,
)

TREE_MODULE = (
    "afabench.components.methods.discriminative.l2m.tabicl_v1_prior."
    "tabiclv1.treeSCM"
)
# The rejection reasons of `TabICLTaskSampler`.
REJECTION_REASONS = {
    "failed",
    "invalid",
    "not_finite",
    "wrong_features",
    "wrong_classes",
}
# (name, features, classes, label_dim, instances per task) of the datasets
# L2M runs on: CUBE, the MiniBooNE subset (binary, one label column) and
# METABRIC. The instance counts are the smallest at which the prior draws
# every class reliably, not the configured task sizes.
DATASET_SHAPES = [
    pytest.param(20, 8, 8, 250, id="cube"),
    pytest.param(10, 2, 1, 120, id="miniboone_subset"),
    pytest.param(11, 6, 6, 200, id="metabric"),
]

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


def test_only_real_pool_has_a_pool_and_only_tabicl_has_workers() -> None:
    assert {prior: prior.uses_pool for prior in TaskPrior} == {
        TaskPrior.real_pool: True,
        TaskPrior.synthetic: False,
        TaskPrior.tabicl_v1: False,
    }
    assert {prior: prior.uses_workers for prior in TaskPrior} == {
        TaskPrior.real_pool: False,
        TaskPrior.synthetic: False,
        TaskPrior.tabicl_v1: True,
    }


@pytest.mark.parametrize(
    ("prior_type", "expected"),
    [
        (TabICLPriorType.prior_bag, {"mlp": 0.961, "gp": 0.039}),
        (
            TabICLPriorType.prior_bag_plus_tree,
            {"mlp": 0.661, "gp": 0.039, "tree": 0.3},
        ),
        (TabICLPriorType.tree_only, {"tree": 1.0}),
    ],
)
def test_prior_types_mix_the_documented_sub_priors(
    prior_type: TabICLPriorType, expected: dict[str, float]
) -> None:
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        label_dim=N_CLASSES,
        prior_type=prior_type,
    )

    assert sampler.prior.prior_weights == expected
    assert set(sampler.prior.base_priors) == set(expected)


@pytest.mark.parametrize(
    ("n_features", "n_classes", "label_dim", "n_rows"), DATASET_SHAPES
)
@pytest.mark.parametrize(
    "prior_type",
    [TabICLPriorType.prior_bag_plus_tree, TabICLPriorType.tree_only],
)
def test_tree_tasks_have_exactly_the_datasets_features_and_classes(
    prior_type: TabICLPriorType,
    n_features: int,
    n_classes: int,
    label_dim: int,
    n_rows: int,
) -> None:
    sampler = TabICLTaskSampler(
        n_features=n_features,
        n_classes=n_classes,
        label_dim=label_dim,
        prior_type=prior_type,
    )

    batch = sampler.sample(
        batch_size=3,
        max_num_points=n_rows,
        num_observed=n_features,
        seed=2,
    )

    assert batch["x"].shape == (3, n_rows, n_features)
    assert batch["y"].shape == (3, n_rows, label_dim)
    assert batch["x"].isfinite().all()
    labels = (
        batch["y"].argmax(dim=-1)
        if label_dim > 1
        else batch["y"].squeeze(-1).long()
    )
    # Never more classes than the label encodes, and exactly the dataset's.
    for task_labels in labels:
        assert task_labels.unique().tolist() == list(range(n_classes))
    assert set(sampler.rejections) <= REJECTION_REASONS


def test_tree_tasks_are_reproducible_from_their_seed() -> None:
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        label_dim=N_CLASSES,
        prior_type=TabICLPriorType.tree_only,
    )

    first = _sample(sampler, seed=9, batch_size=2)

    assert torch.equal(first["x"], _sample(sampler, seed=9, batch_size=2)["x"])
    assert not torch.equal(
        first["x"], _sample(sampler, seed=10, batch_size=2)["x"]
    )


def test_tree_models_use_one_thread_per_task_process() -> None:
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        label_dim=N_CLASSES,
        prior_type=TabICLPriorType.tree_only,
    )

    _sample(sampler, seed=0, batch_size=1)

    assert sys.modules[TREE_MODULE].N_JOBS == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_the_bag_prior_never_imports_the_tree_prior_or_xgboost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, TREE_MODULE, raising=False)
    # A module set to None raises ImportError on import.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES, n_classes=N_CLASSES, label_dim=N_CLASSES
    )

    _sample(sampler, seed=0, batch_size=2)

    assert sampler.prior_type is TabICLPriorType.prior_bag
    assert TREE_MODULE not in sys.modules


def test_the_tree_prior_fails_loudly_without_xgboost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "xgboost", None)
    sampler = TabICLTaskSampler(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        label_dim=N_CLASSES,
        prior_type=TabICLPriorType.tree_only,
    )

    with pytest.raises(ImportError):
        _sample(sampler, seed=0, batch_size=1)


@pytest.mark.parametrize(
    ("n_features", "n_classes", "label_dim", "match"),
    [
        (5, MAX_PRIOR_CLASSES + 1, MAX_PRIOR_CLASSES + 1, "between 2 and 10"),
        (5, 1, 1, "between 2 and 10"),
        (5, 2, 2, "label_dim=2 does not encode n_classes=2"),
        (5, 6, 1, "label_dim=1 does not encode n_classes=6"),
        (0, 4, 4, "n_features=0"),
    ],
)
def test_sizes_the_prior_or_model_cannot_have_fail_loudly(
    n_features: int, n_classes: int, label_dim: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TabICLTaskSampler(
            n_features=n_features, n_classes=n_classes, label_dim=label_dim
        )
