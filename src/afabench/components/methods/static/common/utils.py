from collections.abc import Callable, Sequence
from typing import cast, final, override

import torch
from torch import nn
from torch.distributions import RelaxedOneHotCategorical
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import default_collate

from afabench.core.types import AFADataset


def transform_dataset(
    dataset: AFADataset,
    selected_features: Sequence[int] | torch.Tensor | object,
) -> TensorDataset:
    x, y = dataset.get_all_data()
    feature_indices = cast("Sequence[int] | torch.Tensor", selected_features)
    x_selected = x[:, feature_indices]
    return TensorDataset(x_selected, y)


def restore_parameters(model: nn.Module, best_model: nn.Module) -> None:
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param


def make_masked_collate(
    mask: torch.Tensor,
) -> Callable[
    [list[tuple[torch.Tensor, torch.Tensor]]],
    tuple[torch.Tensor, torch.Tensor],
]:
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = default_collate(batch)
        mask_b = mask.to(dtype=x.dtype, device=x.device)
        return x * mask_b, y

    return collate


@final
class ConcreteMask(nn.Module):
    """
    For differentiable global feature selection.

    Args:
      num_features:
      num_select:
      group_matrix:
      append:
      gamma:

    """

    def __init__(
        self,
        num_features: int,
        num_select: int,
        group_matrix: torch.Tensor | None = None,
        *,
        append: bool = False,
        gamma: float = 0.2,
    ):
        super().__init__()
        self.logits: nn.Parameter = nn.Parameter(
            torch.randn(num_select, num_features, dtype=torch.float32)
        )
        self.append: bool = append
        self.gamma: float = gamma
        self.group_matrix: torch.Tensor | None
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer("group_matrix", group_matrix.float())

    @override
    def forward(self, x: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        dist = RelaxedOneHotCategorical(
            temp, logits=self.logits / self.gamma, validate_args=False
        )
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
