from typing import final, override

import torch

from afabench.core.types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class RandomInitializer(AFAInitializer):
    """Select different random features for each episode."""

    def __init__(self, num_initial_features: int):
        self.num_initial_features = num_initial_features
        self.rng = torch.Generator()

    @override
    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng = torch.Generator()

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for RandomInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        batch_size = int(torch.prod(torch.tensor(batch_shape)))

        num_features = feature_shape.numel()
        num_features_to_unmask = self.num_initial_features
        assert 0 <= num_features_to_unmask <= num_features, (
            f"num_initial_features={num_features_to_unmask} must be within "
            f"[0, {num_features}]"
        )

        if num_features_to_unmask == 0:
            return torch.zeros(
                batch_shape + feature_shape,
                dtype=torch.bool,
                device=features.device,
            )

        # Create different random masks for each batch element
        masks = []
        for _ in range(batch_size):
            flat_indices = torch.randperm(num_features, generator=self.rng)[
                :num_features_to_unmask
            ]
            multi_indices = torch.unravel_index(flat_indices, feature_shape)
            mask = torch.zeros(feature_shape, dtype=torch.bool)
            mask[multi_indices] = True
            masks.append(mask)

        # Stack masks and reshape to match batch dimensions
        stacked_masks = torch.stack(masks, dim=0)
        return stacked_masks.view(batch_shape + feature_shape)
