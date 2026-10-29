from pathlib import Path
from typing import override

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, RelaxedOneHotCategorical

from afabench.components.initializers.config import InitializerConfig
from afabench.components.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.components.unmaskers.utils import (
    get_afa_unmasker_from_config,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import (
    AFADataset,
    AFAInitializer,
    AFAUnmasker,
)
from afabench.core.utils import get_class_frequencies


def restore_parameters(model: nn.Module, best_model: nn.Module) -> None:
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param


def make_onehot(x: torch.Tensor) -> torch.Tensor:
    """Make an approximately one-hot vector one-hot."""
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot


def get_entropy(pred: torch.Tensor) -> torch.Tensor:
    """Calculate entropy, assuming logit predictions."""
    return Categorical(logits=pred, validate_args=False).entropy()


def ind_to_onehot(inds: torch.Tensor, n: int) -> torch.Tensor:
    """Convert index to one-hot encoding."""
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot


def afa_discriminative_training_prep(
    train_dataset_bundle_path: Path,
    val_dataset_bundle_path: Path,
    initializer_cfg: InitializerConfig,
    unmasker_cfg: UnmaskerConfig,
    seed: int | None = None,
) -> tuple[AFADataset, AFADataset, AFAInitializer, AFAUnmasker, torch.Tensor]:
    train_dataset = load_bundle(
        train_dataset_bundle_path, expect=AFADataset
    ).obj
    val_dataset = load_bundle(val_dataset_bundle_path, expect=AFADataset).obj

    initializer = get_afa_initializer_from_config(initializer_cfg)
    unmasker = get_afa_unmasker_from_config(unmasker_cfg)
    if seed is not None:
        initializer.set_seed(seed)
        unmasker.set_seed(seed)

    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    class_weights = 1 / train_class_probabilities
    class_weights = class_weights / class_weights.sum()

    return train_dataset, val_dataset, initializer, unmasker, class_weights


def patch_soft_to_feature_soft(
    soft_patch: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Convert soft patch mask (B, mask_size) into soft feature mask."""
    # if len(x.shape) == 4:
    B, C, H, W = x.shape
    mask_size = soft_patch.shape[1]
    mask_width = int(mask_size**0.5)
    m = soft_patch.view(B, 1, mask_width, mask_width)
    patch_size_h = H // mask_width
    patch_size_w = W // mask_width
    m = F.interpolate(
        m, scale_factor=(patch_size_h, patch_size_w), mode="nearest"
    )
    return m.expand(B, C, H, W)

    # assert len(x.shape) == 2
    # B, D = x.shape
    # mask_size = soft_patch.shape[1]

    # if D == mask_size:
    #     return soft_patch

    # # In case we use patch mask for tabular data
    # assert D % mask_size == 0
    # patch_len = D // mask_size
    # return soft_patch.repeat_interleave(patch_len, dim=1)


def selection_soft_to_feature_soft(
    soft_sel: torch.Tensor,  # [B, n_selections]
    mask_size: int,
    n_contexts: int,
) -> torch.Tensor:
    out = torch.zeros(
        soft_sel.shape[0],
        mask_size,
        device=soft_sel.device,
        dtype=soft_sel.dtype,
    )
    out[:, :n_contexts] = soft_sel[:, [0]]
    if mask_size > n_contexts:
        out[:, n_contexts:] = soft_sel[:, 1:]
    return out


def tie_first_k_linears_by_module(
    classifier: nn.Module, value_network: nn.Module, k: int = 2
) -> None:
    pred_linears = [
        m for m in classifier.modules() if isinstance(m, nn.Linear)
    ]
    val_linears = [
        m for m in value_network.modules() if isinstance(m, nn.Linear)
    ]
    if len(pred_linears) < k or len(val_linears) < k:
        message = (
            f"Need at least {k} Linear layers in each model. "
            f"Got classifier={len(pred_linears)}, value_network={len(val_linears)}"
        )
        raise ValueError(message)
    for i in range(k):
        _replace_module(value_network, val_linears[i], pred_linears[i])


def _replace_module(root: nn.Module, old: nn.Module, new: nn.Module) -> None:
    for name, child in root.named_children():
        if child is old:
            setattr(root, name, new)
            return
        _replace_module(child, old, new)


class MaskLayer(nn.Module):
    """
    Mask layer for tabular data.

    Args:
      append:
      mask_size:

    """

    def __init__(self, append: bool, mask_size: int | None = None):
        super().__init__()
        self.append: bool = append
        self.mask_size: int | None = mask_size

    @override
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = x * m
        if self.append:
            if out.dim() > 2:
                out = out.flatten(start_dim=1)
                m = m.flatten(start_dim=1)
            out: torch.Tensor = torch.cat([out, m], dim=1)
        return out


class MaskLayer2d(nn.Module):
    """
    Mask layer for zeroing out 2d image data.

    Args:
      mask_width: width of the mask, or the number of patches.
      patch_size: upsampling factor for the mask, or number of pixels along
        the side of each patch.
      append: whether to append mask to the output.
    """

    def __init__(self, mask_width: int, patch_size: int, append: bool):
        super().__init__()
        self.append: bool = append
        self.mask_width: int = mask_width
        self.mask_size: int = mask_width**2

        # Set up upsampling.
        self.patch_size: int = patch_size
        self.upsample: nn.Module
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            msg = "patch_size should be int >= 1"
            raise ValueError(msg)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Reshape if necessary.
        if len(mask.shape) == 2:
            mask = mask.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(mask.shape) != 4:
            msg = f"cannot determine how to reshape mask with shape = {
                mask.shape
            }"
            raise ValueError(msg)

        # Apply mask.
        mask = self.upsample(mask)
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class ConcreteSelector(nn.Module):
    """Output layer for selector models."""

    def __init__(self, gamma: float = 0.2) -> None:
        super().__init__()
        self.gamma: float = gamma

    @override
    def forward(
        self,
        logits: torch.Tensor,
        temp: float,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        dist = RelaxedOneHotCategorical(
            temp, logits=logits / self.gamma, validate_args=False
        )
        return dist.rsample()
