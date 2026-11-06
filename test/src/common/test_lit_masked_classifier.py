import torch

from afabench.components.classifiers.lit_models import LitMaskedMLPClassifier


def _module() -> tuple[LitMaskedMLPClassifier, list[torch.Tensor]]:
    module = LitMaskedMLPClassifier(
        n_features=6,
        n_classes=2,
        min_masking_probability=0.0,
        max_masking_probability=0.5,
    )
    module.log = lambda *_args, **_kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[torch.Tensor] = []
    forward = module.classifier.forward

    def capture(
        masked_features: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        seen.append(masked_features.clone())
        seen.append(feature_mask.clone())
        return forward(masked_features, feature_mask)

    module.classifier.forward = capture  # pyright: ignore[reportAttributeAccessIssue]
    return module, seen


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    features = torch.randn(64, 6) + 10.0
    labels = torch.nn.functional.one_hot(torch.randint(0, 2, (64,)), 2).float()
    availability = torch.rand(64, 6) > 0.6
    return features, labels, availability


def test_unavailable_features_never_reach_classifier() -> None:
    module, seen = _module()
    features, labels, availability = _batch()

    module.training_step((features, labels, availability), 0)
    module.validation_step((features, labels, availability), 0)

    assert len(seen) == 6
    for masked_features, feature_mask in zip(
        seen[::2], seen[1::2], strict=True
    ):
        assert not (feature_mask & ~availability).any()
        assert (masked_features[~availability] == 0).all()


def test_two_tensor_batches_are_unchanged() -> None:
    features, labels, availability = _batch()
    module, seen = _module()
    torch.manual_seed(1)
    module.training_step((features, labels), 0)
    torch.manual_seed(1)
    module.training_step((features, labels, torch.ones_like(availability)), 0)

    assert torch.equal(seen[0], seen[2])
    assert torch.equal(seen[1], seen[3])
