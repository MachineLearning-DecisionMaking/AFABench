"""L2M selector with frozen TabICL states and explicit observation masks."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import nn

from afabench.components.methods.discriminative.l2m.upstream.utils import (
    MLP,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    QueryRepresentations,
)

type SelectionLogits = Float[torch.Tensor, "batch query features"]
type PolicyObservationMask = (
    Bool[torch.Tensor, "batch query features"]
    | Float[torch.Tensor, "batch query features"]
)


class AcquisitionPolicy(nn.Module):
    """Score features from TabICL states and the query observation mask."""

    def __init__(
        self,
        n_features: int,
        *,
        representation_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(n_features, representation_dim) < 1:
            message = (
                f"Dimensions must be positive: "
                f"{n_features=}, {representation_dim=}"
            )
            raise ValueError(message)
        self.n_features: int = n_features
        self.representation_dim: int = representation_dim
        self.network: nn.Sequential = MLP(
            input_dim=representation_dim + n_features,
            output_dim=n_features,
            hidden_dim=512,
            num_layers=1,
            policy_head=True,
        )

    @override
    def forward(
        self,
        representations: QueryRepresentations,
        observation_mask: PolicyObservationMask,
    ) -> SelectionLogits:
        if (
            representations.ndim != 3
            or representations.shape[-1] != self.representation_dim
        ):
            message = (
                f"representations shape={representations.shape} must be "
                f"[batch, query, {self.representation_dim}]"
            )
            raise ValueError(message)
        expected_shape = (*representations.shape[:2], self.n_features)
        if observation_mask.shape != expected_shape:
            message = (
                f"observation_mask shape={observation_mask.shape} must be "
                f"{expected_shape}"
            )
            raise ValueError(message)
        if observation_mask.device != representations.device:
            message = (
                "observation_mask and representations must share a device"
            )
            raise ValueError(message)
        if observation_mask.dtype != torch.bool and not (
            observation_mask.is_floating_point()
        ):
            message = "observation_mask must be Boolean or floating-point"
            raise ValueError(message)
        if not ((observation_mask == 0) | (observation_mask == 1)).all():
            message = "observation_mask must contain only zero and one"
            raise ValueError(message)
        # The encoder stays unchanged; the mask enters only the selector.
        inputs = torch.cat(
            (representations, observation_mask.to(representations.dtype)),
            dim=-1,
        )
        return self.network(inputs)
