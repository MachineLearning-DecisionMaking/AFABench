from enum import Enum
from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from afabench.components.methods.rl.common.utils import mask_data
from afabench.core.types import (
    AFAClassifier,
    AFAPredictFn,
    ClassifierOutputKind,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)
from afabench.core.utils import flatten_afa_input


class PointNetType(Enum):
    POINTNET = 1
    POINTNETPLUS = 2


@final
class PointNet(nn.Module):
    """
    Implements the PointNet and PointNetPlus architectures for encoding sets of features, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE".

    This module learns a per-feature identity embedding and combines it with observed feature values
    using either concatenation (PointNet) or pointwise multiplication (PointNetPlus). The resulting
    representations are passed through a feature map encoder and aggregated to produce a fixed-size encoding.

    Args:
        identity_size (int): Size of the identity embedding for each feature.
        n_features (int): Number of input features.
        feature_map_encoder (nn.Module): Module to encode per-feature representations.
        pointnet_type (PointNetType): Type of PointNet variant to use (POINTNET or POINTNETPLUS).
        max_embedding_norm (float | None, optional): Maximum norm for the identity embeddings.
    """

    def __init__(
        self,
        identity_size: int,
        n_features: int,
        feature_map_encoder: nn.Module,
        pointnet_type: PointNetType,
        max_embedding_norm: float | None = None,
    ):
        """
        Initialize the PointNet module.

        Args:
            identity_size (int): Size of the identity embedding for each feature.
            n_features (int): Number of input features.
            feature_map_encoder (nn.Module): Module to encode per-feature representations.
            pointnet_type (PointNetType): Type of PointNet variant to use (POINTNET or POINTNETPLUS).
            max_embedding_norm (float | None, optional): Maximum norm for the identity embeddings.

        """
        super().__init__()

        self.identity_size = identity_size
        self.n_features = n_features
        self.feature_map_encoder = feature_map_encoder  # h in the paper
        self.pointnet_type = pointnet_type
        self.max_embedding_norm = max_embedding_norm

        self.embedding_net = nn.Embedding(
            self.n_features,
            self.identity_size,
            max_norm=self.max_embedding_norm,
        )

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Float[Tensor, "*batch pointnet_size"]:
        """
        Encode a batch of masked feature vectors using PointNet or PointNetPlus.

        Args:
            masked_features (MaskedFeatures):
                Tensor of observed feature values, with zeros for missing features.
                Shape: (batch_size, n_features)
            feature_mask (FeatureMask):
                Binary mask indicating observed features (1 if observed, 0 if missing).
                Shape: (batch_size, n_features)

        Returns:
            Float[Tensor, "*batch pointnet_size"]:
                Encoded representation of the input features.
                Shape: (batch_size, feature_map_size)

        """
        assert masked_features.ndim == 2, (
            f"Expected masked_features to have 2 dimensions, but got {masked_features.ndim}"
        )
        assert feature_mask.ndim == 2, (
            f"Expected feature_mask to have 2 dimensions, but got {feature_mask.ndim}"
        )
        assert masked_features.shape[1] == self.n_features, (
            f"Expected masked_features to have {self.n_features} features, but got {masked_features.shape[1]}"
        )
        assert feature_mask.shape[1] == self.n_features, (
            f"Expected feature_mask to have {self.n_features} features, but got {feature_mask.shape[1]}"
        )

        # Mask out the unobserved features and sum over the set
        return (
            self.element_maps(masked_features) * feature_mask.unsqueeze(-1)
        ).sum(dim=-2)

    def element_maps(
        self, masked_features: Float[Tensor, "*batch n_features"]
    ) -> Float[Tensor, "*batch n_features feature_map_size"]:
        """Feature map h(s_j) of every element j, before masking and pooling."""
        # Identity is a learnable embedding according to EDDI paper
        # Embed each feature once; duplicate max-norm embedding indices make
        # CUDA renormalisation and gradient accumulation nondeterministic.
        identity = self.embedding_net(
            torch.arange(self.n_features, device=masked_features.device)
        )
        if self.pointnet_type == PointNetType.POINTNETPLUS:
            # PointNetPlus does pointwise-multiplication between each identity vector and feature value
            s = masked_features.unsqueeze(-1) * identity
        elif self.pointnet_type == PointNetType.POINTNET:
            # Normal PointNet concatenates feature value with identity vector
            s = torch.cat(
                [
                    masked_features.unsqueeze(-1),
                    identity.expand(*masked_features.shape, -1),
                ],
                dim=-1,
            )
        else:
            msg = f"Unknown PointNet type: {self.pointnet_type}"
            raise ValueError(msg)
        return self.feature_map_encoder(s)


@final
class PartialVAE(nn.Module):
    """
    A partial VAE for masked data, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE".

    To make the model work with different shapes of data, change the pointnet.
    """

    def __init__(
        self,
        pointnet: PointNet,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_size: int,
    ):
        """
        Initialize.

        Args:
            pointnet: maps unordered sets of features to a single vector
            encoder: a network that maps the output from the pointnet to the latent space
            decoder: the network to use for the decoder.
            latent_size: what the latent size is
        """
        super().__init__()

        self.pointnet = pointnet
        self.encoder = encoder
        self.decoder = (
            decoder  # Maps from latent space to the original feature space
        )
        self.latent_size = latent_size

    def encode(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pointnet_output = self.pointnet.forward(masked_features, feature_mask)
        encoding = self.encoder.forward(pointnet_output)
        assert isinstance(encoding, Tensor)

        mu = encoding[..., : encoding.shape[1] // 2]
        logvar = encoding[..., encoding.shape[1] // 2 :]
        std = torch.exp(0.5 * logvar.clamp_max(20.0))
        z = mu + std * torch.randn_like(std)

        return encoding, mu, logvar, z

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encode the masked features
        encoding, mu, logvar, z = self.encode(masked_features, feature_mask)

        # Decode
        x_hat = self.decoder(z)

        return encoding, mu, logvar, z, x_hat


class PartialVAELossType(Enum):
    SQUARED_ERROR = 1
    BINARY_CROSS_ENTROPY = 2


@final
class ODINPretrainingModel(pl.LightningModule):
    """Training the PartialVAE model as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning". This means that labels are appended to the masked feature vector, creating "augmented" features."""

    def __init__(
        self,
        partial_vae: PartialVAE,
        classifier: nn.Module,
        class_probabilities: Float[Tensor, "n_classes"],
        min_masking_probability: float,
        max_masking_probability: float,
        lr: float,
        start_kl_scaling_factor: float,
        end_kl_scaling_factor: float,
        n_annealing_epochs: int,
        # how much more to weigh the classifier's loss compared to the PVAE's loss
        classifier_loss_scaling_factor: float,
    ):
        super().__init__()
        self.partial_vae: PartialVAE = partial_vae
        self.classifier: nn.Module = classifier
        assert class_probabilities.ndim == 1
        class_weights = 1 / class_probabilities
        self.class_weights = class_weights / class_weights.sum()
        self.min_masking_probability: float = min_masking_probability
        self.max_masking_probability: float = max_masking_probability
        self.lr: float = lr
        self.start_kl_scaling_factor: float = start_kl_scaling_factor
        self.end_kl_scaling_factor: float = end_kl_scaling_factor
        self.n_annealing_epochs = n_annealing_epochs
        self.classifier_loss_scaling_factor = classifier_loss_scaling_factor

    @property
    def latent_size(self) -> int:
        """Return latent size of the PVAE."""
        return self.partial_vae.latent_size

    @property
    def n_classes(self) -> int:
        """Number of classes that the model is trained on."""
        return len(self.class_weights)

    @property
    def n_input_features(self) -> int:
        """Number of features in the input that the model is trained on."""
        return self.partial_vae.pointnet.n_features

    @property
    def n_output_features(self) -> int:
        """Number of features in the output that the model is trained on."""
        return self.n_input_features - self.n_classes

    def current_kl_weight(self) -> float:
        """Compute the current KL weight using linear annealing."""
        progress = min(1.0, self.current_epoch / self.n_annealing_epochs)
        return (
            self.start_kl_scaling_factor
            + (self.end_kl_scaling_factor - self.start_kl_scaling_factor)
            * progress
        )

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor, Tensor]:
        features: Features = batch[0]
        label: Label = batch[1]

        assert features.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {features.ndim}"
        )
        assert label.ndim == 2, (
            f"Expected a single batch dimension and single label dimension, got {label.ndim}"
        )

        # According to the paper, labels are appended to the features. "Augmented" = features + labels
        augmented_features = torch.cat(
            [features, label], dim=-1
        )  # (batch_size, n_features+n_classes)

        return augmented_features, features, label

    @override
    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        augmented_features, features, label = self.shared_step(
            batch=batch, batch_idx=batch_idx
        )

        masking_probability = self.min_masking_probability + torch.rand(
            1
        ).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        augmented_masked_features, augmented_feature_mask, _ = mask_data(
            augmented_features, p=masking_probability
        )

        # Pass masked features through VAE, returning estimated features but also encoding which will be passed through classifier
        # IMPORTANT NOTE: the PVAE is trained to reconstruct the *normal* features, not the augmented ones! We can do this
        # since we train a classifier anyways.
        _encoding, mu, logvar, z, estimated_features = (
            self.partial_vae.forward(
                augmented_masked_features, augmented_feature_mask
            )
        )

        self.log(
            "feature_norm",
            (features**2).sum(dim=-1).mean(0).sqrt(),
        )
        self.log(
            "estimated_feature_norm",
            (estimated_features**2).sum(dim=-1).mean(0).sqrt(),
        )

        (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_kl_div_loss,
        ) = self.partial_vae_loss_function(
            estimated_features, features, mu, logvar
        )
        self.log("train_loss_vae", partial_vae_loss, sync_dist=True)
        self.log(
            "train_feature_recon_loss_vae",
            partial_vae_feature_recon_loss,
            sync_dist=True,
        )
        self.log(
            "train_kl_div_loss_vae", partial_vae_kl_div_loss, sync_dist=True
        )

        # Pass the encoding through the classifier
        # A bit unclear whether to use z or mu here. Using z should add a bit of regularization
        logits = self.classifier(z)
        classifier_loss = F.cross_entropy(
            logits, label.float(), weight=self.class_weights.to(logits.device)
        )
        classifier_loss = classifier_loss * self.classifier_loss_scaling_factor
        self.log("train_loss_classifier", classifier_loss, sync_dist=True)

        total_loss = partial_vae_loss + classifier_loss
        self.log("train_loss", total_loss, sync_dist=True)

        return total_loss

    @override
    def on_train_epoch_end(self) -> None:
        self.log("kl_weight", self.current_kl_weight())

    def _get_loss_and_acc(
        self,
        augmented_masked_features: MaskedFeatures,
        augmented_feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert augmented_masked_features.ndim == 2
        assert augmented_feature_mask.ndim == 2
        assert label.ndim == 2

        # Pass masked features through VAE, returning estimated normal features but also latent variables which will be passed through classifier
        _encoder, mu, logvar, _z, estimated_features = self.partial_vae(
            augmented_masked_features, augmented_feature_mask
        )
        (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_kl_div_loss,
        ) = self.partial_vae_loss_function(
            estimated_features, features, mu, logvar
        )

        # Pass the encoding through the classifier
        logits = self.classifier(mu)
        classifier_loss = F.cross_entropy(
            logits, label, weight=self.class_weights.to(logits.device)
        )
        classifier_loss = classifier_loss * self.classifier_loss_scaling_factor

        # For validation, additionally calculate accuracy
        y_pred = torch.argmax(logits, dim=-1)
        y_cls = torch.argmax(label, dim=-1)
        acc = (y_pred == y_cls).float().mean()

        return (
            partial_vae_loss,
            partial_vae_feature_recon_loss,
            partial_vae_kl_div_loss,
            classifier_loss,
            acc,
        )

    @override
    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        augmented_features, features, label = self.shared_step(
            batch=batch, batch_idx=batch_idx
        )

        # Mask features with minimum probability -> see many features (observations)
        augmented_feature_mask_many_observations = (
            torch.rand(
                augmented_features.shape, device=augmented_features.device
            )
            > self.min_masking_probability
        )

        augmented_masked_features_many_observations = (
            augmented_features.clone()
        )
        augmented_masked_features_many_observations[
            augmented_feature_mask_many_observations == 0
        ] = 0
        (
            loss_vae_many_observations,
            feature_recon_loss_vae_many_observations,
            kl_div_loss_vae_many_observations,
            loss_classifier_many_observations,
            acc_many_observations,
        ) = self._get_loss_and_acc(
            augmented_masked_features_many_observations,
            augmented_feature_mask_many_observations,
            features,
            label,
        )
        self.log("val_loss_vae_many_observations", loss_vae_many_observations)
        self.log(
            "val_feature_recon_loss_vae_many_observations",
            feature_recon_loss_vae_many_observations,
        )
        self.log(
            "val_kl_div_loss_vae_many_observations",
            kl_div_loss_vae_many_observations,
        )
        self.log(
            "val_loss_classifier_many_observations",
            loss_classifier_many_observations,
        )
        self.log(
            "val_loss_many_observations",
            loss_vae_many_observations + loss_classifier_many_observations,
        )
        self.log("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        augmented_feature_mask_few_observations = (
            torch.rand(
                augmented_features.shape, device=augmented_features.device
            )
            > self.max_masking_probability
        )

        augmented_masked_features_few_observations = augmented_features.clone()
        augmented_masked_features_few_observations[
            augmented_feature_mask_few_observations == 0
        ] = 0
        (
            loss_vae_few_observations,
            feature_recon_loss_vae_few_observations,
            kl_div_loss_vae_few_observations,
            loss_classifier_few_observations,
            acc_few_observations,
        ) = self._get_loss_and_acc(
            augmented_masked_features_few_observations,
            augmented_feature_mask_few_observations,
            features,
            label,
        )
        self.log("val_loss_vae_few_observations", loss_vae_few_observations)
        self.log(
            "val_feature_recon_loss_vae_few_observations",
            feature_recon_loss_vae_few_observations,
        )
        self.log(
            "val_kl_div_loss_vae_few_observations",
            kl_div_loss_vae_few_observations,
        )
        self.log(
            "val_loss_classifier_few_observations",
            loss_classifier_few_observations,
        )
        self.log(
            "val_loss_few_observations",
            loss_vae_few_observations + loss_classifier_few_observations,
        )
        self.log("val_acc_few_observations", acc_few_observations)

        # if self.verbose:
        #     self.verbose_log()

    def partial_vae_loss_function(
        self,
        estimated_features: Tensor,
        features: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert estimated_features.ndim == 2, (
            f"Expected estimated_features to have 2 dimensions, but got {estimated_features.ndim}"
        )
        assert features.ndim == 2, (
            f"Expected features to have 2 dimensions, but got {features.ndim}"
        )

        feature_recon_loss = (
            ((estimated_features - features) ** 2).sum(dim=1).mean(dim=0)
        )
        kl_div_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(
            dim=1
        ).mean(dim=0)
        kl_div_loss = kl_div_loss * self.current_kl_weight()
        return (
            feature_recon_loss + kl_div_loss,
            feature_recon_loss,
            kl_div_loss,
        )

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def generate_data(
        self,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate `n_samples` of new data.

        Remember, since the model is trained on flat features, this is also what it will generate.
        """
        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_size),
            covariance_matrix=torch.eye(self.latent_size),
        )
        z = dist.sample(torch.Size((n_samples,))).to(self.device)

        # Decode for features
        estimated_flat_features = self.partial_vae.decoder(
            z
        )  # (batch_size, n_features)

        # Apply classifier for class probabilities
        logits = self.classifier(z)
        classifier_probs = logits.softmax(dim=-1)

        return z, estimated_flat_features, classifier_probs

    def fully_observed_reconstruction(
        self,
        features: Features,
        n_classes: int,
        label: Label | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a sample by providing all features. Optionally provide the label as well."""
        assert features.ndim == 2

        return self.masked_reconstruction(
            masked_features=features,
            feature_mask=torch.ones(
                features.shape, dtype=torch.bool, device=features.device
            ),
            n_classes=n_classes,
            label=label,
        )

    def masked_reconstruction(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        n_classes: int,
        label: Label | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a sample by providing masked features. Optionally provide the label as well."""
        assert masked_features.ndim == 2
        assert feature_mask.ndim == 2

        if label is None:
            label = torch.zeros(
                masked_features.shape[0],
                n_classes,
                dtype=torch.float32,
                device=masked_features.device,
            )
            label_mask = torch.full_like(label, False)
        else:
            assert label.ndim == 2
            label_mask = torch.full_like(label, True)

        augmented_masked_features = torch.cat(
            [masked_features, label], dim=-1
        )  # (batch_size, n_features+n_classes)
        augmented_feature_mask = torch.cat([feature_mask, label_mask], dim=-1)

        _encoder, _mu, _logvar, z, estimated_features = self.partial_vae(
            augmented_masked_features, augmented_feature_mask
        )

        return z, estimated_features


@final
class ODINAFAPredictFn(AFAPredictFn):
    """A wrapper for the ODINPretrainingModel to make it compatible with the AFAPredictFn interface."""

    def __init__(self, model: ODINPretrainingModel):
        super().__init__()
        self.model = model

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        # The model assumes flat features
        assert feature_shape is not None

        masked_features, feature_mask, label = flatten_afa_input(
            masked_features, feature_mask, label, feature_shape
        )

        batch_size = masked_features.shape[0]

        augmented_masked_features = torch.cat(
            [
                masked_features,
                torch.zeros(
                    batch_size,
                    self.model.n_classes,
                    device=masked_features.device,
                ),
            ],
            dim=-1,
        )
        augmented_feature_mask = torch.cat(
            [
                feature_mask,
                torch.full(
                    (batch_size, self.model.n_classes),
                    False,
                    device=feature_mask.device,
                ),
            ],
            dim=-1,
        )
        _encoding, mu, _logvar, _z = self.model.partial_vae.encode(
            augmented_masked_features, augmented_feature_mask
        )
        logits = self.model.classifier(mu)
        classifier_probs = logits.softmax(dim=-1)
        return classifier_probs


@final
class ODINAFAClassifier(AFAClassifier):
    """A wrapper for the ODINPretrainingModel to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        model: ODINPretrainingModel,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.model = model.to(self._device)
        self.n_classes = len(model.class_weights)

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        # The model assumes flat features
        assert feature_shape is not None

        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        masked_features, feature_mask, label = flatten_afa_input(
            masked_features, feature_mask, label, feature_shape
        )

        batch_size = masked_features.shape[0]

        augmented_masked_features = torch.cat(
            [
                masked_features,
                torch.zeros(
                    batch_size, self.n_classes, device=masked_features.device
                ),
            ],
            dim=-1,
        )
        augmented_feature_mask = torch.cat(
            [
                feature_mask,
                torch.full(
                    (batch_size, self.n_classes),
                    False,
                    device=feature_mask.device,
                ),
            ],
            dim=-1,
        )

        _encoding, mu, _logvar, _z = self.model.partial_vae.encode(
            augmented_masked_features, augmented_feature_mask
        )
        logits = self.model.classifier(mu)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.model.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        model = torch.load(path, weights_only=False, map_location=device)
        return cls(model, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.model = self.model.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "probabilities"
