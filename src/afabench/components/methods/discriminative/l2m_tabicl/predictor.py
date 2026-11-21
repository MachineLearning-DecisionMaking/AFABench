"""
Wrap pretrained TabICL v2 as a frozen, differentiable predictor.

TabICL weights remain fixed. Gradients through the input features are
preserved so that an acquisition policy can later learn from its loss.

Inputs must already be numerically preprocessed. Context rows come first,
followed by query rows. Only context labels are supplied to the model.
"""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from tabicl._model.tabicl import TabICL
from torch import nn

type TableFeatures = Float[torch.Tensor, "batch rows features"]
type ContextLabels = Int[torch.Tensor, "batch context"]
type QueryLogits = Float[torch.Tensor, "batch query classes"]
type QueryRepresentations = Float[torch.Tensor, "batch query representation"]


class FrozenTabICLPredictor(nn.Module):
    """Load frozen TabICL weights and expose query classification logits."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()

        # Load configuration and pretrained weights without using the GPU.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = checkpoint["config"]
        self.config: dict[str, object] = dict(config)

        # This first wrapper supports classification checkpoints only.
        if config["max_classes"] < 2:
            msg = (
                f"max_classes={config['max_classes']} requires "
                "a classification checkpoint"
            )
            raise ValueError(msg)

        # The differentiable path uses training mode. Require zero dropout
        # so that this does not introduce random dropout during prediction.
        if config["dropout"] != 0.0:
            msg = (
                f"dropout={config['dropout']} must be zero for "
                "this initial predictor wrapper"
            )
            raise ValueError(msg)

        # Use the checkpoint's architecture, then load its learned weights.
        self.model: TabICL = TabICL(**config)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

        # Freeze model parameters while retaining gradients through inputs.
        self.model.requires_grad_(False)

    @override
    def forward(
        self,
        features: TableFeatures,
        context_labels: ContextLabels,
        *,
        n_classes: int,
    ) -> QueryLogits:
        """
        Predict query rows using the labeled context.

        features: [batch, context_rows + query_rows, features]
        context_labels: [batch, context_rows], containing class IDs
        result: [batch, query_rows, n_classes]
        """
        if not 2 <= n_classes <= self.model.max_classes:
            msg = (
                f"n_classes={n_classes} must be between 2 and "
                f"{self.model.max_classes}"
            )
            raise ValueError(msg)

        # TabICL's evaluation path disables autograd internally.
        # Training mode selects its differentiable path; weights stay frozen.
        self.model.train()
        logits = self.model(
            features,
            context_labels,
            embed_with_test=False,
        )

        # Keep only the class outputs needed by the current task.
        return logits[..., :n_classes]

    def query_representations(
        self,
        features: TableFeatures,
        context_labels: ContextLabels,
        *,
        n_classes: int,
    ) -> QueryRepresentations:
        """Read detached query states entering the prediction decoder."""
        captured: list[torch.Tensor] = []
        context_size = context_labels.shape[1]

        def capture(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            captured.append(inputs[0][:, context_size:].detach().clone())

        handle = self.model.icl_predictor.decoder.register_forward_pre_hook(
            capture
        )
        try:
            with torch.no_grad():
                self(
                    features,
                    context_labels,
                    n_classes=n_classes,
                )
        finally:
            handle.remove()

        if len(captured) != 1:
            message = f"Captured {len(captured)} representations; expected one"
            raise RuntimeError(message)

        return captured[0]
