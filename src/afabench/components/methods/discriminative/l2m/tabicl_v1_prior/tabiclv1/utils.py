import random

import numpy as np
import torch
from torch import nn

from .distributions import zipf_sampler_f


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(n_samples)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(n_samples, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(n_samples, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    # x: seq, batch_size
    classes = torch.arange(0, num_classes, device=x.device)  # shape (num_classes, )
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


def permute_classes(input: torch.Tensor) -> torch.Tensor:
    """Label-encode and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape ``(T,)`` containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels of shape ``(T,)``.
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()  # 0, 1, 2..
    mapped = indices[torch.searchsorted(unique_vals, input)]  # to get index for each value

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device)
    permuted = perm[mapped]

    return permuted


def filter_unique_features(xi):
    """Filters features with only one unique value from a single dataset."""
    num_features = xi.shape[-1]
    if isinstance(xi, torch.Tensor):
        unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(num_features)]
    else:
        unique_mask = [len(np.unique(xi[:, j])) > 1 for j in range(num_features)]
    return xi[:, unique_mask]


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7, num_classes_sampler=None):
        if num_classes_sampler is None:
            num_classes_sampler = zipf_sampler_f(0.8, 1, 10)
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)
        num_classes = self.num_classes_sampler()

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        return x
