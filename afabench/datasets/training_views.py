"""Immutable dataset bundles used for training under missingness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

import torch
from sklearn.experimental import (  # noqa: F401
    enable_iterative_imputer,  # pyright: ignore[reportUnusedImport]
)
from sklearn.impute import IterativeImputer, KNNImputer
from torch.utils.data import Dataset

from afabench.core.types import AFADataset, Features, Label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TrainingDatasetView(
    Dataset[tuple[Features, Label]],
    AFADataset,
):
    """A filled dataset together with factual and selectable support masks."""

    _class_version: str = "1.0.0"
    features: torch.Tensor
    labels: torch.Tensor
    source_availability: torch.Tensor
    selection_availability: torch.Tensor
    feature_costs: torch.Tensor | None
    source_dataset_class_name: str
    strategy: str

    def __init__(
        self,
        *,
        features: torch.Tensor,
        labels: torch.Tensor,
        source_availability: torch.Tensor,
        selection_availability: torch.Tensor,
        feature_costs: torch.Tensor,
        source_dataset_class_name: str,
        strategy: str,
    ) -> None:
        if features.shape != source_availability.shape:
            msg = "source_availability must match features."
            raise ValueError(msg)
        if features.shape != selection_availability.shape:
            msg = "selection_availability must match features."
            raise ValueError(msg)
        if features.shape[0] != labels.shape[0]:
            msg = "features and labels must have equal instance counts."
            raise ValueError(msg)
        if feature_costs.shape != features.shape[1:]:
            msg = "feature_costs must match the feature shape."
            raise ValueError(msg)
        if source_availability.dtype is not torch.bool:
            msg = "source_availability must be boolean."
            raise TypeError(msg)
        if selection_availability.dtype is not torch.bool:
            msg = "selection_availability must be boolean."
            raise TypeError(msg)
        self.features = features.detach().clone()
        self.labels = labels.detach().clone()
        self.source_availability = source_availability.bool().detach().clone()
        self.selection_availability = (
            selection_availability.bool().detach().clone()
        )
        self.feature_costs = feature_costs.detach().clone()
        self.source_dataset_class_name = source_dataset_class_name
        self.strategy = strategy

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return self.features.shape[1:]

    @property
    @override
    def label_shape(self) -> torch.Size:
        return self.labels.shape[1:]

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        index = torch.as_tensor(indices, dtype=torch.long)
        return type(self)(
            features=self.features[index],
            labels=self.labels[index],
            source_availability=self.source_availability[index],
            selection_availability=self.selection_availability[index],
            feature_costs=self.get_feature_acquisition_costs(),
            source_dataset_class_name=self.source_dataset_class_name,
            strategy=self.strategy,
        )

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    @override
    def get_feature_acquisition_costs(self) -> torch.Tensor:
        assert self.feature_costs is not None
        return self.feature_costs

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "source_availability": self.source_availability,
                "selection_availability": self.selection_availability,
                "feature_costs": self.get_feature_acquisition_costs(),
                "source_dataset_class_name": self.source_dataset_class_name,
                "strategy": self.strategy,
            },
            path / "training_view.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "training_view.pt", weights_only=True)
        return cls(**data)


def restricted_training_view(
    dataset: AFADataset,
    source_availability: torch.Tensor,
) -> TrainingDatasetView:
    features, labels = dataset.get_all_data()
    masked_features = features.clone()
    masked_features[~source_availability] = 0.0
    return TrainingDatasetView(
        features=masked_features,
        labels=labels,
        source_availability=source_availability,
        selection_availability=source_availability,
        feature_costs=dataset.get_feature_acquisition_costs(),
        source_dataset_class_name=type(dataset).__name__,
        strategy="restricted",
    )


def zero_filled_training_view(
    restricted_view: TrainingDatasetView,
) -> TrainingDatasetView:
    return TrainingDatasetView(
        features=restricted_view.features,
        labels=restricted_view.labels,
        source_availability=restricted_view.source_availability,
        selection_availability=torch.ones_like(
            restricted_view.selection_availability
        ),
        feature_costs=restricted_view.get_feature_acquisition_costs(),
        source_dataset_class_name=restricted_view.source_dataset_class_name,
        strategy="zero_fill",
    )


def mean_filled_training_views(
    train_view: TrainingDatasetView,
    val_view: TrainingDatasetView,
) -> tuple[TrainingDatasetView, TrainingDatasetView]:
    train_available = train_view.source_availability
    counts = train_available.sum(dim=0)
    if (counts == 0).any():
        msg = "Cannot mean-fill a feature with no available training values."
        raise ValueError(msg)
    means = (train_view.features * train_available).sum(dim=0) / counts

    def fill(view: TrainingDatasetView) -> TrainingDatasetView:
        features = view.features.clone()
        expanded_means = means.expand_as(features)
        features[~view.source_availability] = expanded_means[
            ~view.source_availability
        ]
        return TrainingDatasetView(
            features=features,
            labels=view.labels,
            source_availability=view.source_availability,
            selection_availability=torch.ones_like(
                view.selection_availability
            ),
            feature_costs=view.get_feature_acquisition_costs(),
            source_dataset_class_name=view.source_dataset_class_name,
            strategy="mean_fill",
        )

    return fill(train_view), fill(val_view)


KNN_MAX_DONORS = 10_000


def sklearn_filled_training_views(
    train_view: TrainingDatasetView,
    val_view: TrainingDatasetView,
    *,
    strategy: str,
    seed: int,
) -> tuple[TrainingDatasetView, TrainingDatasetView]:
    if strategy == "mice_fill":
        imputer = IterativeImputer(
            sample_posterior=True, max_iter=10, random_state=seed
        )
    elif strategy == "knn_fill":
        imputer = KNNImputer(n_neighbors=5)
    else:
        msg = f"Unsupported imputer strategy: {strategy}"
        raise ValueError(msg)

    def as_nan(view: TrainingDatasetView) -> torch.Tensor:
        flat = view.features.flatten(start_dim=1).double().clone()
        flat[~view.source_availability.flatten(start_dim=1)] = float("nan")
        return flat

    train_nan = as_nan(train_view)
    if train_view.source_availability.flatten(start_dim=1).sum(0).eq(0).any():
        msg = "Cannot impute a feature with no available training values."
        raise ValueError(msg)
    donors = train_nan
    if strategy == "knn_fill" and len(donors) > KNN_MAX_DONORS:
        generator = torch.Generator().manual_seed(seed)
        donors = donors[
            torch.randperm(len(donors), generator=generator)[:KNN_MAX_DONORS]
        ]
    imputer.fit(donors.numpy())

    def fill(view: TrainingDatasetView) -> TrainingDatasetView:
        imputed = torch.from_numpy(imputer.transform(as_nan(view).numpy()))
        imputed = imputed.to(view.features.dtype).reshape(view.features.shape)
        available = view.source_availability
        imputed[available] = view.features[available]
        return TrainingDatasetView(
            features=imputed,
            labels=view.labels,
            source_availability=available,
            selection_availability=torch.ones_like(available),
            feature_costs=view.get_feature_acquisition_costs(),
            source_dataset_class_name=view.source_dataset_class_name,
            strategy=strategy,
        )

    return fill(train_view), fill(val_view)


def concatenate_training_views(
    views: list[TrainingDatasetView],
) -> TrainingDatasetView:
    first = views[0]
    return TrainingDatasetView(
        features=torch.cat([view.features for view in views]),
        labels=torch.cat([view.labels for view in views]),
        source_availability=torch.cat(
            [view.source_availability for view in views]
        ),
        selection_availability=torch.cat(
            [view.selection_availability for view in views]
        ),
        feature_costs=first.get_feature_acquisition_costs(),
        source_dataset_class_name=first.source_dataset_class_name,
        strategy=first.strategy,
    )


def true_value_training_view(
    dataset: AFADataset,
    restricted_view: TrainingDatasetView,
) -> TrainingDatasetView:
    features, labels = dataset.get_all_data()
    return TrainingDatasetView(
        features=features,
        labels=labels,
        source_availability=restricted_view.source_availability,
        selection_availability=torch.ones_like(
            restricted_view.selection_availability
        ),
        feature_costs=dataset.get_feature_acquisition_costs(),
        source_dataset_class_name=type(dataset).__name__,
        strategy="true_completion",
    )


def pvae_restored_training_view(
    restricted_view: TrainingDatasetView,
    reconstructed_features: torch.Tensor,
    *,
    strategy: str,
) -> TrainingDatasetView:
    """Complete missing cells while preserving every factual source value."""
    if reconstructed_features.shape != restricted_view.features.shape:
        msg = "PVAE reconstruction must match the training-view features."
        raise ValueError(msg)
    features = reconstructed_features.to(restricted_view.features).clone()
    available = restricted_view.source_availability
    features[available] = restricted_view.features[available]
    return TrainingDatasetView(
        features=features,
        labels=restricted_view.labels,
        source_availability=available,
        selection_availability=torch.ones_like(available),
        feature_costs=restricted_view.get_feature_acquisition_costs(),
        source_dataset_class_name=restricted_view.source_dataset_class_name,
        strategy=strategy,
    )
