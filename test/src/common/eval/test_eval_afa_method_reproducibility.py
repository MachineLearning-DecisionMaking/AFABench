from collections.abc import Sequence
from pathlib import Path
from typing import Self, override

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal
from torch.utils.data import Dataset

from afabench.core.types import (
    AFAAction,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.evaluation.eval import eval_afa_method


class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, n_samples: int = 20, n_features: int = 4):
        self.features = torch.arange(
            n_samples * n_features, dtype=torch.float32
        ).view(n_samples, n_features)
        self.labels = torch.nn.functional.one_hot(
            torch.arange(n_samples) % 3, num_classes=3
        ).float()
        self.generation_indices = torch.arange(n_samples)

    @property
    def feature_shape(self) -> torch.Size:
        return torch.Size((self.features.shape[-1],))

    @property
    def label_shape(self) -> torch.Size:
        return torch.Size((self.labels.shape[-1],))

    @classmethod
    def accepts_seed(cls) -> bool:
        return False

    def create_subset(self, indices: Sequence[int]) -> Self:
        subset = self.__class__.__new__(self.__class__)
        subset.features = self.features[list(indices)]
        subset.labels = self.labels[list(indices)]
        subset.generation_indices = self.generation_indices[list(indices)]
        return subset

    def get_generation_indices(self) -> torch.Tensor:
        return self.generation_indices

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return self.features.shape[0]

    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features, self.labels

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> Self:
        raise NotImplementedError

    def get_feature_acquisition_costs(self) -> torch.Tensor:
        return torch.ones(self.feature_shape)


def initialize_all_masked(
    features: Features,
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,
) -> FeatureMask:
    assert feature_shape is not None
    return torch.zeros_like(features, dtype=torch.bool)


class StochasticDummyMethod:
    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes
        self.rng = torch.Generator()

    def set_seed(self, seed: int) -> None:
        self.rng.manual_seed(seed)

    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG002
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,  # noqa: ARG002
        feature_shape: torch.Size | None = None,  # noqa: ARG002
    ) -> AFAAction:
        assert selection_mask is not None
        actions = torch.zeros((masked_features.shape[0], 1), dtype=torch.long)
        for sample_idx, sample_selection_mask in enumerate(selection_mask):
            if sample_selection_mask.any():
                continue
            available_actions = (~sample_selection_mask).nonzero().squeeze(-1)
            chosen_action_idx = torch.randint(
                available_actions.numel(), (1,), generator=self.rng
            )
            actions[sample_idx] = available_actions[chosen_action_idx] + 1
        return actions

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG002
        label: Label | None = None,  # noqa: ARG002
        feature_shape: torch.Size | None = None,  # noqa: ARG002
    ) -> Label:
        predicted_class = torch.randint(
            self.n_classes,
            (masked_features.shape[0],),
            generator=self.rng,
        )
        return torch.nn.functional.one_hot(
            predicted_class, num_classes=self.n_classes
        ).float()


def unmask_directly(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    features: Features,  # noqa: ARG001
    afa_selection: AFASelection,
    selection_mask: SelectionMask,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> FeatureMask:
    new_feature_mask = feature_mask.clone()
    rows = torch.arange(new_feature_mask.shape[0])
    new_feature_mask[rows, afa_selection.squeeze(-1)] = True
    return new_feature_mask


def run_eval_after_global_rng_perturbation(seed: int) -> pl.DataFrame:
    method = StochasticDummyMethod()
    method.set_seed(seed)
    torch.rand(100)
    return eval_afa_method(
        afa_action_fn=method.act,
        afa_unmask_fn=unmask_directly,
        n_selection_choices=4,
        afa_initialize_fn=initialize_all_masked,
        dataset=DummyDataset(),
        builtin_afa_predict_fn=method.predict,
        only_n_samples=8,
        batch_size=3,
        seed=seed,
    )


def evaluate_early_stop_method(
    budget: float | None,
    batch_size: int = 3,
    *,
    force_acquisition: bool = False,
) -> pl.DataFrame:
    method = StochasticDummyMethod()
    method.set_seed(123)
    return eval_afa_method(
        afa_action_fn=method.act,
        afa_unmask_fn=unmask_directly,
        n_selection_choices=4,
        afa_initialize_fn=initialize_all_masked,
        dataset=DummyDataset(n_samples=5),
        selection_budget=budget,
        batch_size=batch_size,
        force_acquisition=force_acquisition,
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_early_stop_method_reaches_hard_budget(batch_size: int) -> None:
    table = evaluate_early_stop_method(3, batch_size, force_acquisition=True)
    episodes = table.group_by("episode_id").agg(pl.all().last())
    assert episodes["accumulated_cost"].to_list() == [3.0] * 5
    for episode in table.partition_by("episode_id"):
        actions = episode["action_performed"].to_list()
        assert all(action > 0 for action in actions[:3])
        assert actions[-1] == 0
    assert episodes["forced_stop"].to_list() == [True] * 5


@pytest.mark.parametrize(
    ("budget", "expected_cost", "expected_actions", "forced_stop"),
    [
        (None, 1.0, 2, False),
        (0.0, 0.0, 1, True),
        (2.5, 2.0, 3, True),
        (6.0, 6.0, 7, True),
    ],
)
def test_early_stop_respects_budget_regime(
    budget: float | None,
    expected_cost: float,
    expected_actions: int,
    *,
    forced_stop: bool,
) -> None:
    table = evaluate_early_stop_method(
        budget, force_acquisition=budget is not None
    )
    episodes = table.group_by("episode_id").agg(
        pl.all().last(), pl.len().alias("count")
    )
    assert episodes["accumulated_cost"].to_list() == [expected_cost] * 5
    assert episodes["count"].to_list() == [expected_actions] * 5
    assert episodes["action_performed"].to_list() == [0] * 5
    assert episodes["forced_stop"].to_list() == [forced_stop] * 5


def test_hard_budget_allows_early_stop_without_forcing() -> None:
    episodes = (
        evaluate_early_stop_method(3)
        .group_by("episode_id")
        .agg(pl.all().last(), pl.len().alias("count"))
    )
    assert episodes["accumulated_cost"].to_list() == [1.0] * 5
    assert episodes["count"].to_list() == [2] * 5
    assert episodes["action_performed"].to_list() == [0] * 5
    assert episodes["forced_stop"].to_list() == [False] * 5


def test_episode_identity_survives_batch_boundaries() -> None:
    result = run_eval_after_global_rng_perturbation(seed=123)
    assert result["episode_id"].n_unique() == 8
    assert not result.select("episode_id", "step").is_duplicated().any()
    assert (
        result.group_by("episode_id").agg("step")["step"].to_list()
        == [[0, 1]] * 8
    )


def test_eval_afa_method_is_reproducible_with_seed() -> None:
    first = run_eval_after_global_rng_perturbation(seed=123)
    torch.rand(1000)

    second = run_eval_after_global_rng_perturbation(seed=123)

    assert_frame_equal(first, second)


def test_eval_afa_method_seed_changes_stochastic_results() -> None:
    first = run_eval_after_global_rng_perturbation(seed=123)
    second = run_eval_after_global_rng_perturbation(seed=456)

    assert not first.equals(second)
