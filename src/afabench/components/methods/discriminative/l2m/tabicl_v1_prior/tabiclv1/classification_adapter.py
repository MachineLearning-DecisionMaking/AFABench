import random

import numpy as np
import torch

# PATCH(afabench): import path of the vendored copy.
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.utils_torch import (get_nan_value, normalize_data, remove_outliers)
from .distributions import sample_distributions, uniform_int_sampler_f, parse_distributions, safe_randint
from .utils import CategoricalActivation, randomize_classes, permute_classes


class BalancedBinarize:
    def __call__(self, x):
        return (x > torch.median(x)).float()


def class_sampler_f(min_, max_):
    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2

    return s


class MulticlassSteps:
    """"Sample piecewise constant functions with random number of steps and random class boundaries"""

    def __init__(self, num_classes, max_steps=10):
        self.num_classes = num_classes  # class_sampler_f(2, num_classes)()
        self.num_steps = np.random.randint(1, max_steps) if max_steps > 1 else 1

    def __call__(self, x):
        # x has shape (T,B,H) ?!
        # x has shape (samples, batch)
        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundary_indices = torch.randint(0, x.shape[0], ((self.num_classes - 1) * (self.num_steps - 1) + 1,), device=x.device)
        class_boundaries_sorted, _ = x[class_boundary_indices].sort(axis=0)
        step_assignments = torch.searchsorted(class_boundaries_sorted.T.contiguous(), x.T.contiguous()).T
        class_assignments = torch.randint(0, self.num_classes, (step_assignments.max() + 1, x.shape[1]), device=x.device)
        classes = torch.gather(class_assignments, 0, step_assignments)
        return classes


class MulticlassRank:
    def __init__(self, num_classes, ordered_p=0.5):
        self.num_classes = num_classes  # class_sampler_f(2, num_classes)()
        self.ordered_p = ordered_p

    def __call__(self, x):
        # x has shape (T,B,H)

        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)  # shape (num_classes-1, 1, batch_size, num_features)
        d = (x > class_boundaries).sum(axis=0)

        randomized_classes = torch.rand((d.shape[1],)) > self.ordered_p  # batch_size
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        # random switch class order
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d


class ClassificationAdapter:
    # This class samples the number of features actually use (num_features_used), the number of samples
    # adds NaN and potentially categorical features
    # and discretizes the classification output variable
    # It's instantiated anew for each batch that's created
    def __init__(self, base_prior, config, padding):
        self.h = sample_distributions(parse_distributions(config))
        self.base_prior = base_prior
        self.padding = padding
        if random.random() > 0.5:
            n = random.randint(2, 10)  # 50%: 2~n
        else:
            n = 2
        # L2M CHANGE (not in the original prior): optional fixed class count. `fixed_num_classes` is only set by
        # the afabench adapter (tabicl_task_sampler.py); without it the random draw above is used unchanged.
        if self.h.get('fixed_num_classes'):
            n = self.h['fixed_num_classes']
        self.h['num_classes'] = n

        if self.h['num_classes'] > 1 and not self.h['balanced']:
            if self.h['multiclass_type'] == 'rank':
                self.class_assigner = MulticlassRank(
                    self.h['num_classes'], ordered_p=self.h['output_multiclass_ordered_p']
                )
            elif self.h['multiclass_type'] == 'steps':
                self.class_assigner = MulticlassSteps(self.h['num_classes'], self.h['multiclass_max_steps'])
            else:
                raise ValueError("Unknow Multiclass type")
        elif self.h['num_classes'] == 2 and self.h['balanced']:
            self.class_assigner = BalancedBinarize()
        elif self.h['num_classes'] > 2 and self.h['balanced']:
            raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        nan_prob_sampler = CategoricalActivation(ordered_p=0.0, categorical_p=1.0, num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        # actually only half that probability but that's fine
        x[d < torch.rand((1, d.shape[1], d.shape[2]), device=x.device) * 20 * self.h['nan_prob_a_reason'] - 10] = v
        return x

    def drop_for_no_reason(self, x, v):
        x[torch.rand(x.shape, device=x.device) < random.random() * self.h['nan_prob_no_reason']] = v
        return x

    def __call__(self, batch_size, n_samples, num_features, device, epoch=None, single_eval_pos=None):
        info = {}
        # num_features is constant for all batches, num_features_used is passed down to wrapped priors to change number of features
        if self.h['feature_curriculum']:
            num_features = min(num_features, epoch + 1)
        if self.h['num_features_sampler'] == 'uniform':
            num_features_used = safe_randint(1, num_features)
        elif self.h['num_features_sampler'] == 'double_sample':
            num_features_used = safe_randint(1, safe_randint(1, num_features))
        elif self.h['num_features_sampler'] == 'fixed':
            # L2M CHANGE (not in the original prior): always use all `num_features` features. Only selected by
            # the afabench adapter (tabicl_task_sampler.py) (prior_fix_num_features).
            num_features_used = num_features
        else:
            raise ValueError(f"Unknown num_features_sampler: {self.h['num_features_sampler']}")
        # print(f"[ClassificationAdapter] num_features={num_features}, num_features_used={num_features_used}, padding={self.padding}", flush=True)
        args = {'device': device, 'n_samples': n_samples, 'num_features': num_features_used,
                'batch_size': batch_size, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
        x, y = self.base_prior.get_batch(**args)  # seq, batch, features
        # print(x.shape, y.shape, self.base_prior.__class__.__name__)
        assert x.shape[2] == num_features_used

        if self.h['nan_prob_no_reason'] + self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5:  # Only one out of two datasets should have nans
            if random.random() < self.h['nan_prob_no_reason']:  # Missing for no reason
                x = self.drop_for_no_reason(x, get_nan_value(self.h['set_value_to_nan']))

            if random.random() < self.h['nan_prob_a_reason']:  # Missing for a reason
                x = self.drop_for_reason(x, get_nan_value(self.h['set_value_to_nan']))

        # Categorical features
        categorical_features = []
        if random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    categorical_features.append(col)
                    x[:, :, col] = m(x[:, :, col])
        info['categorical_features'] = categorical_features
        x = remove_outliers(x, categorical_features=categorical_features)  # find the outliers
        x, y = normalize_data(x), normalize_data(y)
        # Cast to classification if enabled
        # In case of regression two normalizations.
        y = self.class_assigner(y).float()
        y = permute_classes(y).float()  # Randomly permute class labels to avoid any ordering bias in the model
        if self.h['max_num_classes'] == 0:
            # Inpute potential nan values after normalization
            y[y.isnan()] = 0

        # Append empty features if enabled
        if self.padding:
            x = torch.cat(
                [x, torch.zeros((x.shape[0], x.shape[1], num_features - num_features_used), device=device)], -1)
            # x = torch.stack([F.pad(xi, (0, num_features - xi.shape[1]), mode="constant", value=0) for xi in x], dim=1)
        # print(f"[ClassificationAdapter] after padding: x.shape={tuple(x.shape)}, padding_applied={self.padding}", flush=True)
        if torch.isnan(y).any():
            print('Nans in target!')
            if self.h['max_num_classes'] == 0:
                y[torch.isnan(y)] = 0
            else:
                y[torch.isnan(y)] = -100

        if self.h['max_num_classes'] != 0:
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch.unique(y[:single_eval_pos, b], sorted=True)
                    targets_in_eval = torch.unique(y[single_eval_pos:, b], sorted=True)

                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                            targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                    if not is_compatible:
                        randperm = torch.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    if self.h['max_num_classes'] != 0:
                        # todo check that it really does this and how many together
                        y[:, b] = -100  # Relies on CE having `ignore_index` set to -100 (default)
                    # todo check that it really does this and how many together
                    y[:, b] = -100  # Relies on CE having `ignore_index` set to -100 (default)
            for b in range(y.shape[1]):
                valid_labels = y[:, b] != -100
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

                if y[valid_labels, b].numel() != 0:
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(0, num_classes, (1,), device=device)
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes
        # print(x.shape, y.shape, 'final output')
        return x, y, info  # x.shape = (T,B,H)


class ClassificationAdapterPrior:
    def __init__(self, base_prior, padding=True, **config):
        self.base_prior = base_prior
        self.config = config
        self.padding = padding

    def get_batch(self, batch_size, n_samples, num_features, device, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
            x, y, info = ClassificationAdapter(self.base_prior, self.config, padding=self.padding)(batch_size=batch_size, **args)
            x, y = x.detach(), y.detach()

        return x, y, info
