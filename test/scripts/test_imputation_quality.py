from __future__ import annotations

import pytest
import torch

from scripts.analysis.imputation_quality import accuracy, masked_rmse


def test_masked_rmse_uses_only_missing_values() -> None:
    truth = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    restored = torch.tensor([[0.0, 4.0], [100.0, 3.0]])
    missing = torch.tensor([[False, True], [False, False]])
    assert masked_rmse(restored, truth, missing) == pytest.approx(3.0)


def test_masked_rmse_repeats_truth_for_multiple_draws() -> None:
    truth = torch.tensor([[1.0], [2.0]])
    restored = torch.tensor([[1.0], [2.0], [3.0], [2.0]])
    missing = torch.ones(4, 1, dtype=torch.bool)
    assert masked_rmse(restored, truth, missing) == pytest.approx(1.0)


def test_masked_rmse_without_missing_values_is_zero() -> None:
    truth = torch.zeros(2, 2)
    assert (
        masked_rmse(truth + 1, truth, torch.zeros(2, 2, dtype=torch.bool)) == 0
    )


def test_accuracy_observes_all_features() -> None:
    def classifier(
        features: torch.Tensor,
        mask: torch.Tensor,
        feature_shape: torch.Size,
    ) -> torch.Tensor:
        assert mask.all()
        assert feature_shape == torch.Size([2])
        return torch.stack([features[:, 0], features[:, 1]], dim=1)

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert accuracy(classifier, features, labels) == pytest.approx(2 / 3)
