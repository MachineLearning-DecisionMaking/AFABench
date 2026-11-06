from typing import TYPE_CHECKING, cast

import torch

from afabench.datasets.datasets import CubeNMDataset
from afabench.datasets.training_views import restricted_training_view
from afabench.missing_values.restoration import (
    PVAEStepwiseRestorer,
    restore_view_with_pvae,
)

if TYPE_CHECKING:
    from afabench.components.methods.rl.odin.models import ODINPretrainingModel


class _FakePVAE:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.labels: list[torch.Tensor | None] = []

    def masked_reconstruction(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,  # noqa: ARG002
        n_classes: int,  # noqa: ARG002
        label: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.labels.append(label)
        fill = 7.0 if label is None else 9.0
        if generator is not None:
            fill = torch.rand((), generator=generator).item()
        return torch.zeros((len(masked_features), 1)), torch.full_like(
            masked_features,
            fill,
        )


def test_pvae_restoration_uses_requested_label_mode() -> None:
    dataset = CubeNMDataset(n_samples=4, seed=1, n_contexts=2)
    availability = torch.ones_like(dataset.features, dtype=torch.bool)
    availability[:, ::2] = False
    restricted = restricted_training_view(dataset, availability)
    model = cast("ODINPretrainingModel", cast("object", _FakePVAE()))

    label_free = restore_view_with_pvae(
        restricted,
        model,
        strategy="pvae_label_free",
        batch_size=2,
    )
    conditioned = restore_view_with_pvae(
        restricted,
        model,
        strategy="pvae_label_conditioned",
        batch_size=3,
    )

    assert (label_free.features[~availability] == 7.0).all()
    assert (conditioned.features[~availability] == 9.0).all()
    assert torch.equal(
        conditioned.features[availability],
        dataset.features[availability],
    )


def test_stepwise_pvae_is_seeded_and_label_free() -> None:
    model = _FakePVAE()
    typed_model = cast("ODINPretrainingModel", cast("object", model))
    first = PVAEStepwiseRestorer(typed_model, n_classes=2, seed=4)
    second = PVAEStepwiseRestorer(typed_model, n_classes=2, seed=4)
    features = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])

    first_draw = first(features, mask)
    second_draw = second(features, mask)

    assert torch.equal(first_draw, second_draw)
    assert model.labels[-2:] == [None, None]


def test_multi_draw_stacks_conditioned_draws() -> None:
    dataset = CubeNMDataset(n_samples=4, seed=1, n_contexts=2)
    availability = torch.ones_like(dataset.features, dtype=torch.bool)
    availability[:, ::2] = False
    restricted = restricted_training_view(dataset, availability)
    model = _FakePVAE()

    restored = restore_view_with_pvae(
        restricted,
        cast("ODINPretrainingModel", cast("object", model)),
        strategy="pvae_multi_draw",
        batch_size=3,
    )

    draws = len(restored) // len(restricted)
    assert draws == 5
    assert all(label is not None for label in model.labels)
    assert torch.equal(restored.labels, restricted.labels.repeat(draws, 1))
    repeated = availability.repeat(draws, 1)
    assert torch.equal(restored.source_availability, repeated)
    assert restored.selection_availability.all()
    assert torch.equal(
        restored.features[repeated],
        dataset.features.repeat(draws, 1)[repeated],
    )
