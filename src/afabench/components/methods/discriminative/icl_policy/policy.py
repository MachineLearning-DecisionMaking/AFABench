"""
Copy of TabICL v2's decoder structure, ready for policy design decisions.

Source: tabicl/_model/learning.py, ICLearning.__init__, self.decoder.
The head receives ONLY the shared representation and outputs feature logits.
Masks are applied externally to numerical inputs and selection validity. model.py connects it to TabICL; train.py defines the acquisition loss.
"""

from typing import override

import torch
from jaxtyping import Float
from torch import nn

type QueryRepresentations = Float[torch.Tensor, "batch query representation"]
type HeadScores = Float[torch.Tensor, "batch query features"]


class PolicyHead(nn.Module):
    """Decoder-shaped MLP with embedding-only input and feature-logit output."""

    def __init__(
        self,
        *,
        n_features: int,
        representation_dim: int = 512,
    ) -> None:
        super().__init__()
        if representation_dim < 1 or n_features < 1:
            message = (
                "Dimensions must be positive: "
                f"{representation_dim=}, {n_features=}"
            )
            raise ValueError(message)
        self.representation_dim: int = representation_dim
        self.n_features: int = n_features

        # P03 OUTPUTS: one logit per feature, fixed feature count/order.
        # TODO: [P03 OUTPUTS] Compare groups/stop/cross-table scoring later.
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(representation_dim, representation_dim * 2),
            nn.GELU(),
            nn.Linear(representation_dim * 2, n_features),
        )

        # TODO: [P04 INITIALIZATION] Choose initialization separately from the architecture.
        # This uses fresh PyTorch weights. Options: keep fresh weights; copy
        # the classifier's first Linear as an experiment; use L2M's small
        # orthogonal final-layer initialization. Class-output weights have
        # no established correspondence to feature selections.

    @override
    def forward(
        self,
        representations: QueryRepresentations,
    ) -> HeadScores:
        if representations.ndim != 3 or (
            representations.shape[-1] != self.representation_dim
        ):
            message = (
                f"representations shape={representations.shape} must be "
                f"[batch, query, {self.representation_dim}]"
            )
            raise ValueError(message)

        # P02 EXTERNAL MASK: do not append masks or modify this representation.
        # The exact same query states are supplied to the classifier decoder.
        return self.network(representations)


# P01 MISSING VALUES: build_masked_table hides values with NaN. The model
# interface calls official TabICL preprocessing, fitted on context only.
# TODO: [P01 MISSING VALUES] Extend beyond the verified numeric single-view
# profile only with new official-pipeline parity and reveal-gradient tests.

# P05 VALID SELECTIONS: train.select_feature excludes unavailable/acquired
# features and safely returns zeros for exhausted instances.
# TODO: [P05 VALID SELECTIONS] Extend unit-cost feature selections to groups,
# nonuniform costs or learned stopping if those become experiment targets.

# P06 LEARNING: train.afa_training_step now uses straight-through
# Gumbel-softmax, post-reveal cross-entropy and random-prefix states.
# TODO: [P06 LEARNING] Compare estimators, temperature, costs and lookahead.
# It uses fixed labeled context, not L2M's autoregressive earlier-query labels.

# P07 UPDATES: joint mode trains policy at 1e-4 and all pretrained
# components at 1e-5. Frozen mode is retained as a controlled baseline.
# TODO: [P07 UPDATES] Validate retained prediction quality after joint updates.
# Both heads share a representation, but joint training changes that space.

# TODO: [P08 EVALUATION] Compare choices on fixed validation tasks/budgets/seeds.
# Include random acquisition with the same classifier; keep test labels out
# of training and model selection. Useful gradients do not prove useful AFA.
