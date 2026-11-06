from pathlib import Path

import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets.datasets import CubeNMDataset
from afabench.datasets.training_views import (
    TrainingDatasetView,
    mean_filled_training_views,
    pvae_restored_training_view,
    restricted_training_view,
    sklearn_filled_training_views,
    true_value_training_view,
    zero_filled_training_view,
)


def _dataset(seed: int) -> CubeNMDataset:
    return CubeNMDataset(n_samples=20, seed=seed, n_contexts=3)


def _availability(dataset: CubeNMDataset) -> torch.Tensor:
    available = torch.ones_like(dataset.features, dtype=torch.bool)
    available[::2, ::3] = False
    return available


def test_restricted_view_has_a_strong_leakage_boundary() -> None:
    dataset = _dataset(1)
    available = _availability(dataset)
    view = restricted_training_view(dataset, available)

    assert torch.equal(view.source_availability, available)
    assert torch.equal(view.selection_availability, available)
    assert (view.features[~available] == 0).all()
    assert torch.equal(view.features[available], dataset.features[available])


def test_mean_fill_uses_only_observed_training_values() -> None:
    train_dataset = _dataset(1)
    val_dataset = _dataset(2)
    train_available = _availability(train_dataset)
    val_available = _availability(val_dataset)
    train_restricted = restricted_training_view(
        train_dataset,
        train_available,
    )
    val_restricted = restricted_training_view(val_dataset, val_available)

    train_view, val_view = mean_filled_training_views(
        train_restricted,
        val_restricted,
    )
    expected_means = (train_dataset.features * train_available).sum(
        dim=0
    ) / train_available.sum(dim=0)

    assert train_view.selection_availability.all()
    assert val_view.selection_availability.all()
    assert torch.equal(
        val_view.features[~val_available],
        expected_means.expand_as(val_view.features)[~val_available],
    )


def test_completion_strategies_preserve_factual_cells() -> None:
    dataset = _dataset(1)
    available = _availability(dataset)
    restricted = restricted_training_view(dataset, available)

    zero = zero_filled_training_view(restricted)
    true = true_value_training_view(dataset, restricted)
    reconstruction = torch.full_like(dataset.features, 99.0)
    pvae = pvae_restored_training_view(
        restricted,
        reconstruction,
        strategy="pvae_label_free",
    )

    assert zero.selection_availability.all()
    assert true.selection_availability.all()
    assert pvae.selection_availability.all()
    assert torch.equal(pvae.features[available], dataset.features[available])
    assert (pvae.features[~available] == 99.0).all()


def test_training_view_bundle_roundtrip(tmp_path: Path) -> None:
    dataset = _dataset(1)
    view = restricted_training_view(dataset, _availability(dataset))
    path = tmp_path / "view.bundle"
    save_bundle(view, path, {"strategy": view.strategy})

    loaded, manifest = load_bundle(path)

    assert isinstance(loaded, TrainingDatasetView)
    assert torch.equal(loaded.features, view.features)
    assert torch.equal(loaded.source_availability, view.source_availability)
    assert torch.equal(
        loaded.selection_availability,
        view.selection_availability,
    )
    assert manifest["metadata"]["strategy"] == "restricted"


@pytest.mark.parametrize("strategy", ["mice_fill", "knn_fill"])
def test_sklearn_fill_keeps_factual_values(strategy: str) -> None:
    train_dataset = _dataset(1)
    val_dataset = _dataset(2)
    train_available = _availability(train_dataset)
    val_available = _availability(val_dataset)

    train_view, val_view = sklearn_filled_training_views(
        restricted_training_view(train_dataset, train_available),
        restricted_training_view(val_dataset, val_available),
        strategy=strategy,
        seed=0,
    )

    for view, dataset, available in (
        (train_view, train_dataset, train_available),
        (val_view, val_dataset, val_available),
    ):
        assert view.strategy == strategy
        assert view.selection_availability.all()
        assert torch.equal(view.source_availability, available)
        assert torch.equal(
            view.features[available], dataset.features[available]
        )
        assert torch.isfinite(view.features).all()
