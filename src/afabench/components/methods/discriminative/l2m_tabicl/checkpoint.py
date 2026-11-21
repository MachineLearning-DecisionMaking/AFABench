"""Save and restore frozen TabICL and its trainable L2M policy together."""

from pathlib import Path

import torch
from torch import nn

from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)


def save_models(
    path: Path,
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
) -> None:
    """Write model weights and reconstruction settings to a directory."""
    first_layer = policy.network[0]
    last_layer = policy.network[-1]

    if not isinstance(first_layer, nn.Linear) or not isinstance(
        last_layer, nn.Linear
    ):
        message = "The L2M policy must start and end with linear layers"
        raise TypeError(message)

    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": predictor.config,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in predictor.model.state_dict().items()
            },
        },
        path / "predictor.pt",
    )

    torch.save(
        {
            "n_features": last_layer.out_features,
            "representation_dim": policy.representation_dim,
            "policy_inputs": "representation_and_observation_mask",
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in policy.state_dict().items()
            },
        },
        path / "policy.pt",
    )


def load_models(
    path: Path,
    device: torch.device,
) -> tuple[FrozenTabICLPredictor, AcquisitionPolicy]:
    """Rebuild the models without needing the original TabICL file."""
    checkpoint = torch.load(
        path / "policy.pt",
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("policy_inputs")
        != "representation_and_observation_mask"
    ):
        message = (
            "Checkpoint must use representation and observation mask inputs"
        )
        raise ValueError(message)

    policy = AcquisitionPolicy(
        n_features=checkpoint["n_features"],
        representation_dim=checkpoint["representation_dim"],
    )

    policy.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    predictor = FrozenTabICLPredictor(path / "predictor.pt").to(device)
    return predictor.eval(), policy.to(device).eval()
