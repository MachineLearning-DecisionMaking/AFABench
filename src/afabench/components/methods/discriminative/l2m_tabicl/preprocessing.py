"""
Context-fitted preprocessing and differentiable feature masking.

Preprocessing statistics come only from labeled context features.
Unknown query features use the transformed context-imputation baseline.
"""

import torch
from jaxtyping import Float
from tabicl._sklearn.preprocessing import (
    PreprocessingPipeline,
    TransformToNumerical,
)

type RowFeatures = Float[torch.Tensor, "rows features"]
type ObservationMask = Float[torch.Tensor, "rows features"]
type MissingBaseline = Float[torch.Tensor, " features"]


class ContextPreprocessor:
    """Fit numerical imputation and scaling once on context rows."""

    def __init__(self, context_features: RowFeatures) -> None:
        if context_features.ndim != 2 or len(context_features) < 2:
            msg = (
                f"context_features shape={context_features.shape} "
                "must contain at least two rows"
            )
            raise ValueError(msg)
        if not torch.isfinite(context_features).all():
            message = (
                "context_features contains nonfinite values; "
                "this initial implementation requires finite context"
            )
            raise ValueError(message)

        # These fixed sklearn operations run before mask construction.
        context_array = context_features.detach().cpu().numpy()

        # Use TabICL's numerical imputation, fitted only on context.
        self.numerical: TransformToNumerical = TransformToNumerical()
        numerical_context = self.numerical.fit_transform(context_array)

        # "none" skips optional normalization, but still scales and clips.
        self.pipeline: PreprocessingPipeline = PreprocessingPipeline(
            normalization_method="none",
            random_state=17,
        )
        self.pipeline.fit(numerical_context)

        # Find the processed value that represents an unknown cell.
        missing_row = torch.full_like(context_features[:1], float("nan"))
        self.missing_baseline: MissingBaseline = self.transform(missing_row)[0]

    def transform(self, features: RowFeatures) -> RowFeatures:
        """Apply fixed transforms without fitting on query features."""
        feature_array = features.detach().cpu().numpy()
        numerical_features = self.numerical.transform(feature_array)
        transformed = self.pipeline.transform(numerical_features)

        return torch.as_tensor(
            transformed,
            dtype=features.dtype,
            device=features.device,
        )


def apply_observation_mask(
    features: RowFeatures,
    observation_mask: ObservationMask,
    missing_baseline: MissingBaseline,
) -> RowFeatures:
    """Use imputation at mask=0 and the observed value at mask=1."""
    if observation_mask.shape != features.shape:
        msg = (
            f"observation_mask shape={observation_mask.shape} "
            f"must match features shape={features.shape}"
        )
        raise ValueError(msg)

    # Keep the mask floating-point so acquisition gradients survive.
    return missing_baseline + observation_mask * (features - missing_baseline)
