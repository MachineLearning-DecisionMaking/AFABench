"""Upstream's simulation prior as L2M's synthetic task prior."""

import random

import numpy as np
import pytest
import torch

from afabench.components.methods.discriminative.l2m.synthetic_task_sampler import (
    FEATURE_HIGH,
    FEATURE_LOW,
    MAX_MISSING_PROBABILITY,
    SyntheticTaskSampler,
)

N_FEATURES = 12
N_ROWS = 300
CONTEXT_SIZE = 50


def _sample(
    sampler: SyntheticTaskSampler, seed: int | None, batch_size: int = 6
) -> dict[str, torch.Tensor]:
    return sampler.sample(
        batch_size=batch_size,
        max_num_points=N_ROWS,
        num_observed=N_FEATURES,
        seed=seed,
    )


@pytest.fixture(scope="module")
def sampler() -> SyntheticTaskSampler:
    return SyntheticTaskSampler(n_features=N_FEATURES, label_dim=4)


def test_features_are_uniform_on_the_boxes_of_upstreams_simulation(
    sampler: SyntheticTaskSampler,
) -> None:
    batch = _sample(sampler, seed=1)

    assert batch["x"].shape == (6, N_ROWS, N_FEATURES)
    assert batch["x"].min() >= FEATURE_LOW
    assert batch["x"].max() <= FEATURE_HIGH
    assert batch["x"].mean().abs() < 0.1
    # Uniform on [-2, 2] has standard deviation 4 / sqrt(12).
    torch.testing.assert_close(
        batch["x"].std(), torch.tensor(4 / 12**0.5), atol=0.05, rtol=0
    )


def test_every_task_has_new_features(sampler: SyntheticTaskSampler) -> None:
    batch = _sample(sampler, seed=2)

    assert not torch.equal(batch["x"][0], batch["x"][1])
    assert not torch.equal(batch["x"], _sample(sampler, seed=3)["x"])
    # Unseeded batches come from the global generator and differ as well.
    assert not torch.equal(
        _sample(sampler, seed=None)["x"], _sample(sampler, seed=None)["x"]
    )


def test_missing_values_have_a_probability_per_task_and_feature(
    sampler: SyntheticTaskSampler,
) -> None:
    batch = _sample(sampler, seed=4, batch_size=16)

    assert set(batch["r"].unique().tolist()) == {0.0, 1.0}
    missing = 1 - batch["r"].mean(dim=1)  # per task and feature
    # Each probability is drawn from [0, 0.5); allow sampling noise.
    assert missing.max() < MAX_MISSING_PROBABILITY + 0.1
    assert missing.min() < 0.1
    assert missing.max() > 0.3


def test_a_batch_splits_into_context_and_query_instances(
    sampler: SyntheticTaskSampler,
) -> None:
    batch = _sample(sampler, seed=5)

    for name in ("x", "y", "mask", "r"):
        assert torch.equal(batch[f"{name}c"], batch[name][:, :CONTEXT_SIZE])
        assert torch.equal(batch[f"{name}t"], batch[name][:, CONTEXT_SIZE:])
    # The queries start with every feature acquired, as the fit asks for.
    assert torch.equal(batch["maskt"], torch.ones_like(batch["xt"]))


def test_labels_are_upstreams_one_hot_classes(
    sampler: SyntheticTaskSampler,
) -> None:
    batch = _sample(sampler, seed=6)

    assert batch["y"].shape == (6, N_ROWS, 4)
    assert torch.equal(batch["y"].sum(dim=-1), torch.ones(6, N_ROWS))


def test_binary_tasks_use_one_label_column() -> None:
    binary = SyntheticTaskSampler(n_features=3, label_dim=1)

    batch = binary.sample(
        batch_size=2, max_num_points=200, num_observed=3, seed=0
    )

    assert batch["y"].shape == (2, 200, 1)
    assert set(batch["y"].unique().tolist()) <= {0.0, 1.0}


def test_a_seeded_batch_is_reproducible(sampler: SyntheticTaskSampler) -> None:
    first = _sample(sampler, seed=7)
    second = _sample(sampler, seed=7)

    for name, values in first.items():
        assert torch.equal(values, second[name]), name


def test_a_seeded_batch_leaves_the_global_generators_alone(
    sampler: SyntheticTaskSampler,
) -> None:
    random.seed(10)
    np.random.seed(10)  # noqa: NPY002
    torch.manual_seed(10)
    state = torch.get_rng_state()

    _sample(sampler, seed=8)

    assert torch.equal(torch.get_rng_state(), state)


def test_a_task_needs_a_feature() -> None:
    with pytest.raises(ValueError, match="n_features=0"):
        SyntheticTaskSampler(n_features=0, label_dim=2)
