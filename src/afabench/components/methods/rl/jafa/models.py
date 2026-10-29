from pathlib import Path
from typing import Self, cast, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torchrl.modules import MLP

from afabench.components.methods.rl.common.utils import (
    mask_data,
    observed_indices,
)
from afabench.components.methods.rl.jafa.custom_types import (
    Embedder,
    Embedding,
    EmbeddingClassifier,
)
from afabench.core.types import (
    AFAClassifier,
    AFAPredictFn,
    ClassifierOutputKind,
    FeatureMask,
    Features,
    Label,
    Logits,
    MaskedFeatures,
)
from afabench.core.utils import flatten_afa_input
from afabench.fit.step_logging import epoch_means, should_log_step


@final
class ReadProcessEncoder(nn.Module):
    """A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391)."""

    def __init__(
        self,
        # size of each element in input set. Usually 1 (scalar sequence)
        set_element_size: int,
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: tuple[int, ...] = (32, 32),
        writing_block_cells: tuple[int, ...] = (32, 32),
        # each element in the sequence gets converted into a memory with this size
        memory_size: int = 16,
        # RNN processing steps. Paper shows that 5-10 is good.
        processing_steps: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.set_element_size = set_element_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps
        self.dropout = dropout

        self.reading_block = MLP(
            in_features=set_element_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        self.process_lstm = nn.LSTMCell(memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        # Empty sets are represented by a learnable vector
        self.empty_set_vector = nn.Parameter(torch.zeros(output_size))

    @override
    def forward(self, input_set: Tensor, lengths: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_set: a tensor of shape (batch_size, set_size, element_size). The valid elements have to be first in the sequence.
            lengths: a tensor of shape (batch_size). Number of valid elements in each sequence.

        Returns:
            output: a tensor of shape (batch_size, output_size)

        """
        assert input_set.ndim == 3, (
            f"Expected input_set to have shape (batch_size, set_size, element_size), instead got {input_set.shape}"
        )
        assert lengths.ndim == 1, (
            f"Expected lengths to have shape (batch_size,), instead got {lengths.shape}"
        )

        # Read: Map each set elements to a memory vector
        return self.process(self.reading_block(input_set), lengths)

    def forward_indexed(
        self, values: Tensor, indices: Tensor, lengths: Tensor
    ) -> Tensor:
        """Forward pass for sets of [value, onehot(index)] elements, without materializing the one-hot vectors."""
        first, *rest = self.reading_block.children()
        first = cast("nn.Linear", first)
        memories = (
            values.unsqueeze(-1) * first.weight[:, 0]
            + F.embedding(indices, first.weight[:, 1:].T)
            + first.bias
        )
        for layer in rest:
            memories = layer(memories)
        return self.process(memories, lengths)

    def process(self, memories: Tensor, lengths: Tensor) -> Tensor:
        """Attend over (batch_size, set_size, memory_size) memories, of which the first `lengths` are valid."""
        batch_size, set_size, _ = memories.shape
        nonempty_set_mask = (lengths > 0).unsqueeze(-1)
        valid = torch.arange(set_size, device=memories.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        valid = valid | ~nonempty_set_mask

        # Initialize lstm state
        q_t = torch.zeros(batch_size, self.memory_size, device=memories.device)
        c_t = torch.zeros(batch_size, self.memory_size, device=memories.device)
        # Initial input
        r_t = torch.zeros(batch_size, self.memory_size, device=memories.device)

        # Process: Iteratively refine state with attention over memories
        for _ in range(self.processing_steps):
            # LSTM update
            q_t, c_t = self.process_lstm(
                r_t, (q_t, c_t)
            )  # q_t: (batch_size, memory_size)

            # Attention between each memory vector and all memories, even the invalid ones, to produce logits
            e_t = torch.bmm(memories, q_t.unsqueeze(-1)).squeeze(
                -1
            )  # (batch_size, set_size)

            # Mask padding elements
            e_t.masked_fill_(~valid, float("-inf"))

            # Compute attention weights
            a_t = torch.softmax(e_t, dim=-1)  # (batch_size, set_size)

            # Read from memory with attention
            r_t = torch.bmm(a_t.unsqueeze(1), memories).squeeze(
                1
            )  # (batch_size, memory_size)

        # Write: Transform final state to output
        output = self.write_block(
            torch.cat((q_t, r_t), dim=1)
        )  # (batch_size, output_size)

        return torch.where(
            nonempty_set_mask,
            output,
            self.empty_set_vector.expand_as(output),
        )


@final
class JAFAEmbedder(Embedder):
    """Wrap a ReadProcessEmbedder to handle inputs consisting of features and their indices, using the representation described in "Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding."""

    def __init__(self, encoder: ReadProcessEncoder):
        super().__init__()
        self.encoder = encoder

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        # We currently assume a single batch dimension and single feature dimension
        assert masked_features.shape == feature_mask.shape, (
            "masked_features and feature_mask must have the same shape"
        )
        assert masked_features.dim() == 2, (
            "masked_features and feature_mask must have 2 dimensions (batch_size, n_features)"
        )
        # Shuffling set should improve training
        indices, lengths = observed_indices(feature_mask, shuffle=True)
        values = masked_features.gather(-1, indices)
        return self.encoder.forward_indexed(values, indices, lengths)


@final
class JAFAMLPClassifier(EmbeddingClassifier):
    def __init__(
        self, input_size: int, num_classes: int, num_cells: tuple[int, ...]
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_cells = num_cells

        self.mlp = MLP(
            input_size,
            num_classes,
            num_cells=num_cells,
            activation_class=nn.ReLU,
        )

    @override
    def forward(self, embedding: Embedding) -> Logits:
        return self.mlp(embedding)


@final
class LitJAFAEmbedderClassifier(pl.LightningModule):
    """A module that combines the ShimEncoder with a classifier for pretraining."""

    def __init__(
        self,
        embedder: JAFAEmbedder,
        classifier: JAFAMLPClassifier,
        class_probabilities: Float[Tensor, "n_classes"],  # noqa: F821
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier
        class_weight = 1 / class_probabilities
        self.class_weight = class_weight / class_weight.sum()
        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[Embedding, Logits]:
        """
        Forward pass.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            embedding: the embedding of the input features
            logits: the output of the classifier

        """
        assert masked_features.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {masked_features.ndim}"
        )

        embedding = self.embedder(masked_features, feature_mask)
        logits = self.classifier(embedding)

        return embedding, logits

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):  # noqa: ANN201
        features: Features = batch[0]
        label: Label = batch[1]

        assert features.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {features.ndim}"
        )
        assert label.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {label.ndim}"
        )

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
        _, logits = self(masked_features, feature_mask)
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
        _, logits = self(masked_features, feature_mask)
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

        assert feature_values.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {feature_values.ndim}"
        )
        assert y.ndim == 2, (
            f"Expected a single batch dimension and single feature dimension, got {y.ndim}"
        )

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
    def configure_optimizers(self):  # noqa: ANN201
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class JAFAAFAPredictFn(AFAPredictFn):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAPredictFn interface."""

    def __init__(self, embedder_and_classifier: LitJAFAEmbedderClassifier):
        super().__init__()
        self.embedder_and_classifier = embedder_and_classifier

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

        _, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits.softmax(dim=-1)


@final
class JAFAAFAClassifier(AFAClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        pretrained_model: LitJAFAEmbedderClassifier,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.pretrained_model = pretrained_model.to(self._device)

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

        _, logits = self.pretrained_model(masked_features, feature_mask)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.pretrained_model.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        embedder_and_classifier = torch.load(
            path, weights_only=False, map_location=device
        )
        return cls(embedder_and_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.pretrained_model = self.pretrained_model.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "probabilities"
