from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from torch import nn

# PATCH(afabench): the vendored package's own import path.
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.utils_torch import default_device
from .distributions import parse_distributions, sample_distributions

# PATCH(afabench): threads per tree model, which the original sets to every available CPU
# (`n_jobs=-1`). The task sampler sets it to 1 so that parallelism comes from its worker processes.
N_JOBS = -1


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std)


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:

    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    n_samples : int
        Length of sequence to generate.

    num_features : int
        Number of features to generate.

    pre_stats : bool
        Whether to pre-generate statistics for the input features.

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform').

    device : str, default='cpu'
        Device to store tensors on.
    """

    def __init__(self, n_samples, num_features, pre_stats=False, sampling="mixed", device="cpu"):
        self.n_samples = n_samples
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(means, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.n_samples, 1)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.n_samples, 1)

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape ``(n_samples, num_features)``.
        """
        samplers = {"normal": self.sample_normal_all, "mixed": self.sample_mixed, "uniform": self.sample_uniform}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.n_samples, self.num_features), device=self.device).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.n_samples, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.n_samples,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.n_samples, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.n_samples,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        X = []
        zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.n_samples,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)


class TreeLayer(nn.Module):
    """A layer that transforms input features using a tree-based model.

    This layer fits a specified tree-based regression model (Decision Tree,
    Extra Trees, Random Forest, or XGBoost) to the input features using
    randomly generated target values. It then uses the trained model
    to predict the outputs.

    Parameters
    ----------
    tree_model : str
        The type of tree-based model to use. Options are "decision_tree",
        "extra_trees", "random_forest", "xgboost".

    max_depth : int
        The maximum depth allowed for the individual trees in the model.

    n_estimators : int
        The number of trees in the ensemble.

    out_dim : int
        The desired output dimension for the transformed features. This determines
        the number of target variables (``y_fake``) generated for fitting the
        multi-output regressor.

    device : str or torch.device
        The device ('cpu' or 'cuda') on which to place the output tensor.
    """

    def __init__(self, tree_model: str, max_depth: int, n_estimators: int, out_dim: int, device: str):
        super(TreeLayer, self).__init__()
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.out_dim = out_dim
        self.device = device

        # PATCH(afabench): every tree model below takes `n_jobs=N_JOBS` where the original has `n_jobs=-1`.
        if tree_model == "decision_tree":
            self.model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=max_depth, splitter="random"), n_jobs=N_JOBS)
        elif tree_model == "extra_trees":
            self.model = MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth), n_jobs=N_JOBS
            )
        elif tree_model == "random_forest":
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth), n_jobs=N_JOBS
            )
        elif tree_model == "xgboost":
            # PATCH(afabench) (not in the original prior): the module-level import was commented out although xgboost is the
            # default tree model, which made every TreeSCM raise NameError. Imported here so that only runs that
            # actually use the xgboost trees need the package.
            from xgboost import XGBRegressor

            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",
                multi_strategy="multi_output_tree",
                n_jobs=N_JOBS,
            )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")

    def forward(self, X):
        """Applies the fitted tree-based transformation to the input features.

        Fits the internal tree model using X and random targets, then predicts
        and returns the outputs.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Transformed features tensor of shape ``(n_samples, out_dim)``.
        """
        X = X.nan_to_num(0.0).cpu()
        y_fake = np.random.randn(X.shape[0], self.out_dim)
        self.model.fit(X, y_fake)
        y = self.model.predict(X)
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        if self.out_dim == 1:
            y = y.view(-1, 1)

        return y


class TreeSCM(nn.Module):
    """A Tree-based Structural Causal Model for generating synthetic datasets.

    Similar to MLP-based SCM but uses tree-based models (like Random Forests or XGBoost)
    for potentially non-linear feature transformations instead of linear layers.

    Parameters
    ----------
    n_samples : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate outputs of the tree transformations applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the tree layers becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final tree layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate tree outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate tree outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=5
        Number of tree transformation layers.

    hidden_dim : int, default=10
        Output dimension size for intermediate tree transformations.

    tree_model : str, default="xgboost"
        Type of tree model to use ("decision_tree", "extra_trees", "random_forest",
        "xgboost"). XGBoost is favored for performance as it supports multi-output
        regression natively.

    max_depth_lambda : float, default=0.5
        Lambda parameter for sampling the max depth for tree models from an
        exponential distribution.

    n_estimators_lambda : float, default=0.5
        Lambda parameter for sampling the number of estimators (trees) per layer
        from an exponential distribution.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each tree
        layer's transformation.

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.
        If `True`, the noise standard deviation for each output dimension of a layer
        is sampled from a normal distribution centered at 0 with `noise_std`.
        If `False`, a fixed `noise_std` is used for all dimensions.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
            self,
            n_samples: int = 1024,
            num_features: int = 100,
            num_outputs: int = 1,
            is_causal: bool = True,
            num_causes: int = 10,
            y_is_effect: bool = True,
            in_clique: bool = False,
            sort_features: bool = True,
            num_layers: int = 5,
            hidden_dim: int = 10,
            tree_model: str = "xgboost",
            max_depth_lambda: float = 0.5,
            n_estimators_lambda: float = 0.5,
            sampling: str = "normal",
            pre_sample_cause_stats: bool = False,
            noise_std: float = 0.01,
            pre_sample_noise_std: bool = False,
            device: str = "cpu",
            **kwargs,
    ):
        super(TreeSCM, self).__init__()
        # Tree models can be slow so we use less layers, smaller hidden dim, and non-causal mode
        is_causal = False
        num_layers = np.random.randint(1, 3)
        hidden_dim = np.random.randint(3, 10)

        # Data Generation Settings
        self.n_samples = n_samples
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tree_model = tree_model
        self.tree_depth_lambda = max_depth_lambda
        self.tree_n_estimators_lambda = n_estimators_lambda
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
        else:
            # In non-causal mode, features are the causes
            self.num_causes = self.num_features

        # Define the input sampler
        self.xsampler = XSampler(
            n_samples=self.n_samples,
            num_features=self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers
        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))  # lambda increase, distribution is sharp
        n_estimators = 1 + int(np.random.exponential(1 / self.tree_n_estimators_lambda))
        layers = [
            TreeLayer(
                tree_model=self.tree_model,
                max_depth=min(max_depth, 4),
                n_estimators=min(n_estimators, 4),
                out_dim=self.hidden_dim,
                device=self.device,
            )
        ]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, tree-based transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim

        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1 / self.tree_n_estimators_lambda))
        tree_layer = TreeLayer(
            tree_model=self.tree_model,
            max_depth=min(max_depth, 4),
            n_estimators=min(n_estimators, 4),
            out_dim=out_dim,
            device=self.device,
        )

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(tree_layer, noise_layer)

    def forward(self):
        """Generates synthetic data by sampling input features and applying tree-based transformations."""
        causes = self.xsampler.sample()  # (n_samples, num_causes)

        # Generate outputs through MLP layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]  # Start from 2 because the first layer is only linear without activation

        # Handle outputs based on causality
        X, y = self.handle_outputs(causes, outputs)

        # Check for NaNs and handle them by setting to default values
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def handle_outputs(self, causes, outputs):
        """Handle outputs based on whether causal or not.

        If causal, sample inputs and target from the graph.
        If not causal, directly use causes as inputs and last output as target.

        Parameters
        ----------
        causes : torch.Tensor
            Causes of shape ``(n_samples, num_causes)``.

        outputs : list of torch.Tensor
            List of output tensors from tree layers.

        Returns
        -------
        X : torch.Tensor
            Input features of shape ``(n_samples, num_features)``.

        y : torch.Tensor
            Target of shape ``(n_samples, num_outputs)``.
        """
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if self.in_clique:
                # When in_clique=True, features and targets are sampled as a block, ensuring that
                # selected variables may share dense dependencies.
                start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
                random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
            else:
                # Otherwise, features and targets are randomly and independently selected
                random_perm = torch.randperm(outputs_flat.shape[-1] - 1, device=self.device)

            indices_X = random_perm[self.num_outputs: self.num_outputs + self.num_features]
            if self.y_is_effect:
                # If targets are effects, take last output dims
                indices_y = list(range(-self.num_outputs, 0))
            else:
                # Otherwise, take from the beginning of the permuted list
                indices_y = random_perm[: self.num_outputs]

            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)

            # Select input features and targets from outputs
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            y = outputs[-1]

        return X, y


class TreeSCMPrior:
    def __init__(self, config=None, batch_size_per_subgroup=None):
        self.config = parse_distributions(config or {})
        self.batch_size_per_subgroup = batch_size_per_subgroup

    def get_batch(self, batch_size, n_samples, num_features, device=default_device,
                  epoch=None, single_eval_pos=None):
        with torch.no_grad():
            if self.batch_size_per_subgroup is not None:
                assert batch_size % self.batch_size_per_subgroup == 0, \
                    "Batch size must be divisible by batch_size_per_subgroup"
                x_list, y_list = [], []
                for i in range(0, batch_size, self.batch_size_per_subgroup):
                    sub_x, sub_y = self._generate_subgroup(
                        self.batch_size_per_subgroup, n_samples, num_features, device, epoch, single_eval_pos)
                    x_list.append(sub_x)
                    y_list.append(sub_y)
                x = torch.cat(x_list, dim=1)  # (T, batch_size, H)
                y = torch.cat(y_list, dim=1)
            else:
                x, y = self._generate_subgroup(batch_size, n_samples, num_features, device, epoch, single_eval_pos)
            return x, y

    def _generate_subgroup(self, sub_batch_size, n_samples, num_features, device, epoch, single_eval_pos):
        with torch.no_grad():
            model = TreeSCM(
                n_samples=n_samples,
                num_features=num_features,
                device=device,
                **sample_distributions(self.config),
            ).to(device)

            xs, ys = [], []
            for _ in range(sub_batch_size):
                x, y = model()  # same TreeSCM, different random noise each call
                xs.append(x.unsqueeze(1))  # (T, 1, H)
                ys.append(y.unsqueeze(1))  # (T, 1)
            x = torch.cat(xs, dim=1)  # (T, sub_batch_size, H)
            y = torch.cat(ys, dim=1)  # (T, sub_batch_size)
            return x, y
