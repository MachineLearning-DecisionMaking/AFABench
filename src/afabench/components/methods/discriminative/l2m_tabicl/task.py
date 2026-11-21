from dataclasses import dataclass

from torch import Tensor


@dataclass
class Task:
    """Features, one-hot labels, and observed-feature mask for one training task."""

    features: Tensor
    labels: Tensor
    feature_mask: Tensor
