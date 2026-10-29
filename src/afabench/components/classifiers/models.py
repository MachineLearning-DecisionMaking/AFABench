from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torchrl.modules import MLP

from afabench.components.methods.rl.common.utils import (
    mask_data,
)
from afabench.fit.step_logging import epoch_means, should_log_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from afabench.components.classifiers.protocols import VisionBackbone
    from afabench.components.methods.discriminative.common.utils import (
        MaskLayer2d,
    )
    from afabench.core.types import (
        FeatureMask,
        Features,
        Label,
        Logits,
        MaskedFeatures,
    )


@final
class Predictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


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
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> torch.Tensor:
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = self.min_masking_probability + torch.rand(
            1
        ).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        log_step = should_log_step(self)
        if log_step:
            self.log("masking_probability", masking_probability)

        masked_features, feature_mask, _ = mask_data(
            features, p=masking_probability
        )

        logits = self(masked_features, feature_mask)
        loss = F.cross_entropy(
            logits, label, weight=self.class_weight.to(logits.device)
        )
        if log_step:
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
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        means = epoch_means(self).start_batch(len(batch[0]))
        feature_values, y = batch

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.min_masking_probability
        )
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[
            feature_mask_many_observations == 0
        ] = 0
        loss_many_observations, acc_many_observations = self._get_loss_and_acc(
            feature_values_many_observations, feature_mask_many_observations, y
        )
        means.add("val_loss_many_observations", loss_many_observations)
        means.add("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        feature_mask_few_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.max_masking_probability
        )
        feature_values_few_observations = feature_values.clone()
        feature_values_few_observations[feature_mask_few_observations == 0] = 0
        loss_few_observations, acc_few_observations = self._get_loss_and_acc(
            feature_values_few_observations, feature_mask_few_observations, y
        )
        means.add("val_loss_few_observations", loss_few_observations)
        means.add("val_acc_few_observations", acc_few_observations)

    @override
    def on_validation_epoch_end(self) -> None:
        epoch_means(self).log(self)

    @override
    def on_fit_start(self) -> None:
        self.class_weight = self.class_weight.to(self.device)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class MaskedMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_cells: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.num_cells = num_cells
        self.dropout = dropout

        self.classifier = MLP(
            in_features=self.n_features * 2,
            out_features=self.n_classes,
            num_cells=self.num_cells,
            activation_class=nn.ReLU,
            dropout=self.dropout,
        )

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        # Concatenate the masked features and the feature mask
        x = torch.cat((masked_features, feature_mask), dim=1)
        logits = self.classifier(x)
        return logits

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        return self.forward(masked_features, feature_mask)


class MaskedViTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super().__init__()
        feat_dim = getattr(
            backbone, "embed_dim", getattr(backbone, "num_features", None)
        )
        if feat_dim is None:
            msg = "Backbone must expose embed_dim or num_features."
            raise AttributeError(msg)
        self.backbone: VisionBackbone = cast(
            "VisionBackbone", cast("object", backbone)
        )
        self.fc: nn.Linear = nn.Linear(feat_dim, num_classes)

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        x = masked_features
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        logits = self.fc(pooled)
        return logits

    # @override
    # def __call__(
    #     self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    # ) -> Logits:
    #     return self.forward(masked_features, feature_mask)


class MaskedViTTrainer(nn.Module):
    def __init__(self, model: nn.Module, mask_layer: MaskLayer2d) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.mask_layer: MaskLayer2d = mask_layer

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def fit(  # noqa: PLR0915
        self,
        train_loader: torch.utils.data.DataLoader[tuple[Tensor, Tensor]],
        val_loader: torch.utils.data.DataLoader[tuple[Tensor, Tensor]],
        lr: float,
        nepochs: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        val_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        val_loss_mode: Literal["min", "max"] | None = None,
        factor: float = 0.2,
        patience: int = 2,
        min_lr: float = 1e-6,
        min_mask: float = 0.1,
        max_mask: float = 0.9,
        logger: logging.Logger | None = None,
        metric_logger: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        assert val_loss_fn is not None
        assert val_loss_mode in ["min", "max"]
        log = logger if logger is not None else logging.getLogger(__name__)

        model = self.model
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        def better(a: float, b: float) -> bool:
            return (a < b) if val_loss_mode == "min" else (a > b)

        best_state = None
        best_metric = None

        for epoch in range(nepochs):
            model.train()
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x = x_batch.to(device)
                y = self._to_class_indices(y_batch).to(device)

                # Random mask ratio
                p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                n = self.mask_layer.mask_size
                m = (torch.rand(x.size(0), n, device=device) < p).float()

                x_masked = self.mask_layer(x, m)
                logits = model(x_masked, m)
                loss = loss_fn(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                all_preds, all_labels = [], []
                correct = 0
                total = 0
                for x_batch, y_batch in val_loader:
                    x = x_batch.to(device)
                    y = self._to_class_indices(y_batch).to(device)
                    p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                    n = self.mask_layer.mask_size
                    m = (torch.rand(x.size(0), n, device=device) < p).float()
                    x_masked = self.mask_layer(x, m)
                    logits = model(x_masked, m)
                    all_preds.append(logits)
                    all_labels.append(y)

                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

                y_full = torch.cat(all_labels)
                preds_full = torch.cat(all_preds)
                val_loss = val_loss_fn(preds_full, y_full).item()
                val_accuracy = correct / total
                log.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                    epoch,
                    nepochs,
                    avg_train_loss,
                    val_loss,
                    val_accuracy,
                )
                if metric_logger is not None:
                    metric_logger(
                        {
                            "epoch": epoch,
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                            "lr": opt.param_groups[0]["lr"],
                        }
                    )

            scheduler.step(val_loss)

            if (best_metric is None) or better(val_loss, best_metric):
                best_metric = val_loss
                best_state = deepcopy(model.state_dict())

        if best_state:
            model.load_state_dict(best_state)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked, mask)
