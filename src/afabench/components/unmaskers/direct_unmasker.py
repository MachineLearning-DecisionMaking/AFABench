from typing import override

import torch

from afabench.core.types import (
    AFASelection,
    AFAUnmasker,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class DirectUnmasker(AFAUnmasker):
    @override
    def set_seed(self, seed: int | None) -> None:
        # This unmasker is deterministic
        pass

    @override
    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        """Features correspond to selections, and therefore the costs do as well."""
        return feature_costs

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        # The number of selections is a flattened view of the feature shape
        return feature_shape.numel()

    @override
    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Unmasks the features assuming `afa_selection` are 0-based indices of the features to uncover.

        afa_selection < 0 (e.g., -1 from stop action) is ignored.
        """
        assert feature_shape is not None, "feature_shape must be provided"
        assert afa_selection.shape[-1] == 1, (
            "AFASelection must have shape [..., 1]"
        )

        # Get batch shape by removing the last dimension (which is 1)
        batch_shape = afa_selection.shape[:-1]
        selection = afa_selection.reshape(-1, 1)
        new_feature_mask_flat = feature_mask.reshape(
            len(selection), feature_shape.numel()
        ).clone()
        index = selection.clamp(min=0)
        current = new_feature_mask_flat.gather(1, index)
        new_feature_mask_flat.scatter_(
            1,
            index,
            torch.where(selection >= 0, torch.ones_like(current), current),
        )

        return new_feature_mask_flat.view(batch_shape + feature_shape)
