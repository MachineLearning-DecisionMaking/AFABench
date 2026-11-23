from pathlib import Path
from typing import Self, cast, final, override

import pytest
import torch
from torch import nn

from afabench.components.methods.generative.eddi.afa_methods import (
    EDDIAFAMethod,
)
from afabench.components.methods.rl.odin.models import (
    PartialVAE,
    PointNet,
    PointNetType,
)
from afabench.core.types import (
    AFAClassifier,
    ClassifierOutputKind,
    FeatureMask,
    Label,
    MaskedFeatures,
)


class FakeSampler(nn.Module):
    def forward(
        self, masked_features: torch.Tensor, feature_mask: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del feature_mask
        batch_size = masked_features.shape[0]
        z = torch.zeros(batch_size, 3, device=masked_features.device)
        return z, z, z, z, masked_features


def tiny_partial_vae(n_features: int, n_classes: int) -> PartialVAE:
    pointnet = PointNet(
        identity_size=4,
        n_features=n_features + n_classes,
        feature_map_encoder=nn.Linear(5, 3),
        pointnet_type=PointNetType.POINTNET,
    )
    return PartialVAE(
        pointnet, nn.Linear(3, 6), nn.Linear(3, n_features), latent_size=3
    )


class FakePredictor(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z[:, 0], z[:, 0] + 1], dim=-1)


@final
class FakeExternalClassifier(AFAClassifier):
    """Minimal AFAClassifier fake with a declared output_kind."""

    def __init__(self, output_kind: ClassifierOutputKind | str) -> None:
        self._output_kind: ClassifierOutputKind | str = output_kind
        self._device: torch.device = torch.device("cpu")

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_mask, label, feature_shape
        # Scores that include negative values, to exercise the bug where
        # negative logits fed directly into a KL log() produce NaNs.
        score0 = masked_features.sum(dim=-1) - 5.0
        logits = torch.stack([score0, -score0], dim=-1)
        if self._output_kind == "probabilities":
            return logits.softmax(dim=-1)
        return logits

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return cast("ClassifierOutputKind", self._output_kind)

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        raise NotImplementedError

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


def test_eddi_predict_flattens_image_features() -> None:
    method = EDDIAFAMethod(
        sampler=FakeSampler(),
        predictor=FakePredictor(),
        num_classes=2,
        num_mc_samples=2,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros_like(masked_features, dtype=torch.bool)

    prediction = method.predict(
        masked_features=masked_features,
        feature_mask=feature_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    assert prediction.shape == (2, 2)


def test_eddi_act_supports_synthetic_mnist_patch_selection() -> None:
    method = EDDIAFAMethod(
        sampler=tiny_partial_vae(n_features=784, n_classes=2),
        predictor=FakePredictor(),
        num_classes=2,
        selection_costs=torch.ones(49),
        num_mc_samples=2,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros_like(masked_features, dtype=torch.bool)
    selection_mask = torch.zeros((2, 49), dtype=torch.bool)

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    assert action.shape == (2, 1)
    assert torch.all((action >= 1) & (action <= 49))


def _make_method_with_external_classifier(
    output_kind: ClassifierOutputKind | str,
) -> EDDIAFAMethod:
    method = EDDIAFAMethod(
        sampler=FakeSampler(),
        predictor=FakePredictor(),
        num_classes=2,
        num_mc_samples=2,
    )
    method.classifier = FakeExternalClassifier(output_kind)
    return method


@pytest.mark.parametrize("output_kind", ["logits", "probabilities"])
def test_eddi_act_with_external_classifier_yields_finite_scores(
    output_kind: ClassifierOutputKind,
) -> None:
    method = _make_method_with_external_classifier(output_kind)
    masked_features = torch.rand((3, 6))
    feature_mask = torch.ones_like(masked_features, dtype=torch.bool)
    feature_mask[:, -1] = False

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
    )

    assert action.shape == (3, 1)
    assert torch.isfinite(action.float()).all()
    assert torch.all((action >= 0) & (action <= 6))


def test_eddi_external_classifier_probs_softmaxes_negative_logits() -> None:
    method = _make_method_with_external_classifier("logits")
    logits = torch.tensor([[-3.0, -7.0], [2.0, -1.0]])

    probs = method._external_classifier_probs(logits)  # noqa: SLF001

    assert torch.isfinite(probs).all()
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))


def test_eddi_external_classifier_probs_passes_through_probabilities() -> None:
    method = _make_method_with_external_classifier("probabilities")
    probs_in = torch.tensor([[0.1, 0.9], [0.5, 0.5]])

    probs_out = method._external_classifier_probs(probs_in)  # noqa: SLF001

    assert torch.equal(probs_out, probs_in)


def test_eddi_external_classifier_probs_rejects_unknown_output_kind() -> None:
    method = _make_method_with_external_classifier("unknown")

    with pytest.raises(ValueError, match="output_kind"):
        method._external_classifier_probs(torch.zeros((2, 2)))  # noqa: SLF001
