"""
Context-fitted TabICL 'none' preprocessing, with torch query transforms.

Keep feature order fixed. Fit mean imputation, scaling and outlier bounds
with the installed package; reproduce its numerical transforms in torch.
No query values participate in fitting and no query conversion to NumPy
occurs. Missing measurements are sanitized before differentiable reveals.
"""

from typing import override

import torch
from jaxtyping import Float
from sklearn.impute import SimpleImputer
from tabicl._sklearn.preprocessing import (
    PreprocessingPipeline,
    TransformToNumerical,
)
from torch import nn

type Matrix = Float[torch.Tensor, "rows features"]


class NumericalTransform(nn.Module):
    fill: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def __init__(self, context: Matrix) -> None:
        super().__init__()
        if context.ndim != 2 or len(context) < 2:
            msg = "context must have at least two rows"
            raise ValueError(msg)
        if torch.isinf(context).any():
            msg = "context contains infinite values"
            raise ValueError(msg)
        numerical = TransformToNumerical()
        values = numerical.fit_transform(context.detach().cpu().numpy())
        pipeline = PreprocessingPipeline(normalization_method="none")
        pipeline.fit(values)
        if not isinstance(numerical.tfm_, SimpleImputer):
            msg = "numerical input must use mean imputation"
            raise TypeError(msg)
        for name, array in {
            "fill": numerical.tfm_.statistics_,
            "mean": pipeline.standard_scaler_.mean_,
            "scale": pipeline.standard_scaler_.scale_,
            "lower": pipeline.outlier_remover_.lower_bounds_,
            "upper": pipeline.outlier_remover_.upper_bounds_,
        }.items():
            self.register_buffer(name, torch.as_tensor(array).to(context))
        self.fill = self.get_buffer("fill")
        self.mean = self.get_buffer("mean")
        self.scale = self.get_buffer("scale")
        self.lower = self.get_buffer("lower")
        self.upper = self.get_buffer("upper")

    @override
    def forward(self, values: Matrix) -> Matrix:
        values = torch.where(torch.isfinite(values), values, self.fill)
        scaled = ((values - self.mean) / self.scale).clamp(-100, 100)
        clipped = torch.maximum(
            -torch.log1p(scaled.abs()) + self.lower, scaled
        )
        return torch.minimum(torch.log1p(clipped.abs()) + self.upper, clipped)

    def reveal(self, values: Matrix, mask: Matrix) -> Matrix:
        # Both branches are finite before multiplying by the relaxed mask.
        safe = torch.where(torch.isfinite(values), values, self.fill)
        return self(self.fill + mask * (safe - self.fill))
