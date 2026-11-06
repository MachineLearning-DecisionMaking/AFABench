"""
Lightning wrapper for the masked MLP classifier, used only during training.

Kept out of `models.py` because importing `lightning` costs ~1.3 s (it pulls
`torch._dynamo`), and `models.py` sits on the eval path via
`afabench.components.classifiers`. Evaluation loads a plain `state_dict` and
never touches this, so the cost belongs to whoever actually trains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.components.methods.rl.common.utils import mask_data

if TYPE_CHECKING:
    from afabench.core.types import (
        FeatureMask,
        Features,
        Label,
        Logits,
        MaskedFeatures,
    )


def _availability(batch: tuple[Tensor, ...]) -> Tensor | None:
    return batch[2].bool() if len(batch) > 2 else None


@final
class LitMaskedMLPClassifier(pl.LightningModule):
    """A lit module for a MaskedMLPClassifier that takes masked features and feature masks as input."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_cells: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        class_probabilities: Tensor | None = None,
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.num_cells = num_cells
        self.dropout = dropout

        self.classifier = MaskedMLPClassifier(
            n_features=n_features,
            n_classes=n_classes,
            num_cells=num_cells,
            dropout=dropout,
        )

        if class_probabilities is None:
            self.class_probabilities = torch.ones(n_classes) / n_classes
        else:
            self.class_probabilities = class_probabilities
        class_weight = 1 / self.class_probabilities
        self.class_weight = class_weight / class_weight.sum()

        self.lr = lr

        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

        self.save_hyperparameters()

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        """
        Forward.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            logits

        """
        return self.classifier(masked_features, feature_mask)

    @override
    def training_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        features: Features = batch[0]
        label: Label = batch[1]
        availability = _availability(batch)

        masking_probability = self.min_masking_probability + torch.rand(
            1
        ).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(
            features, p=masking_probability
        )
        if availability is not None:
            feature_mask &= availability
            masked_features = features * feature_mask.float()

        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(
            logits, label, weight=self.class_weight.to(logits.device)
        )
        self.log("train_loss", loss)
        return loss

    def _get_loss_and_acc(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(
            logits, y, weight=self.class_weight.to(logits.device)
        )
        predicted_class = torch.argmax(logits, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (predicted_class == true_class).float().mean()
        return loss, acc

    @override
    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> None:
        feature_values, y = batch[0], batch[1]
        availability = _availability(batch)

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.min_masking_probability
        )
        if availability is not None:
            feature_mask_many_observations &= availability
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[
            feature_mask_many_observations == 0
        ] = 0
        loss_many_observations, acc_many_observations = self._get_loss_and_acc(
            feature_values_many_observations, feature_mask_many_observations, y
        )
        self.log("val_loss_many_observations", loss_many_observations)
        self.log("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        feature_mask_few_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.max_masking_probability
        )
        if availability is not None:
            feature_mask_few_observations &= availability
        feature_values_few_observations = feature_values.clone()
        feature_values_few_observations[feature_mask_few_observations == 0] = 0
        loss_few_observations, acc_few_observations = self._get_loss_and_acc(
            feature_values_few_observations, feature_mask_few_observations, y
        )
        self.log("val_loss_few_observations", loss_few_observations)
        self.log("val_acc_few_observations", acc_few_observations)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
