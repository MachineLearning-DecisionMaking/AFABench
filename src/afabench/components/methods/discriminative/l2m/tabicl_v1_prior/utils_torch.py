# PATCH(afabench): only the functions the vendored prior imports from the
# Mother-Tree module `model/utils_torch.py` (and the helpers they call), copied
# verbatim; the rest of that module needs wandb and training code. Source lines
# 62, 65-69, 72-116 and 127-154 of the module.
import random

import torch


default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def get_nan_value(set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return float('nan')
    else:
        return random.choice([-999, 0, 1, 999])


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1. - num / x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1, categorical_features=None):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    if categorical_features:
        categorical_mask = torch.zeros(X.shape[2], dtype=torch.bool, device=X.device)
        categorical_mask.scatter_(0, torch.tensor(categorical_features, device=X.device, dtype=int), 1.)

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    if categorical_features:
        X = torch.where(categorical_mask, X, torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X))
        X = torch.where(categorical_mask, X, torch.minimum(torch.log(1 + torch.abs(X)) + upper, X))
    else:
        X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
        X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X
