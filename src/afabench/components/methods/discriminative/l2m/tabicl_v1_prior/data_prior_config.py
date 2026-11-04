import os
import sys
from typing import List  # was provided implicitly by the star import from prior.activation

import torch

# PATCH(afabench): the vendored package's own import path replaces the sys.path
# append of the repository the prior came from.
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.activation import *
from logging import warning


class RandomChoiceActivation(nn.Module):
    """Randomly selects and instantiates one activation function from a list.

    Parameters
    ----------
    act_list : list of nn.Module
        List of activation function constructors to choose from.

    Attributes
    ----------
    act : nn.Module
        The randomly selected activation function instance.
    """

    def __init__(self, act_list: List[nn.Module]):
        super().__init__()
        self.act = act_list[np.random.randint(len(act_list))]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class StdRandomScaleFactory:
    def __init__(self, act_class, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(StdScaleLayer(), RandomScaleLayer(individual=self.individual), self.act_class())


class RandomChoiceFactory:
    """Factory class for creating RandomChoice activations"""

    def __init__(self, act_classes):
        self.act_classes = act_classes

    def __call__(self):
        return RandomChoiceActivation(self.act_classes)


def get_activations(random: bool = True, scale: bool = True, diverse: bool = True):
    """Generate a list of activation functions with various configurations.

    This function creates a list of activation functions by combining simple activations
    with optional random functions, scaling, and diversity options.

    Parameters
    ----------
    random : bool, default=True
        If True, adds RandomFunctionActivation to the list and samples it multiple
        times to increase probability of selection.

    scale : bool, default=True
        If True, wraps activations with StdRandomScaleFactory to add standardization
        and random scaling.

    diverse : bool, default=True
        If True, adds RandomChoiceFactory instances to allow different activation
        functions in each layer.
    """
    # Start with a set of simple activations
    simple_activations = [
        nn.Tanh,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.SELU,
        nn.SiLU,
        nn.ReLU,
        nn.Softplus,
        nn.ReLU6,
        nn.Hardtanh,
        SignActivation,
        RBFActivation,
        ExpActivation,
        SqrtAbsActivation,
        UnitIntervalIndicator,
        SineActivation,
        SquareActivation,
        AbsActivation,
    ]
    activations = simple_activations
    if random:
        # Add random activation and sample it more often
        activations += [RandomFunctionActivation] * 10

    if scale:
        # Create scaled versions using StdRandomScaleFactory
        activations = [StdRandomScaleFactory(act) for act in activations]

    if diverse:
        # Add possibility to have different activation functions in each layer
        activations += [RandomChoiceFactory(activations)] * len(activations)

    return activations


# MotherNet MLP prior configuration
def mothernet_mlp_config():
    """
    Copy from original mothernet repo
    """
    mlp_prior_config = {"pre_sample_causes": True,
                        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
                        'prior_mlp_scale_weights_sqrt': True,
                        'random_feature_rotation': True,
                        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
                        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
                        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                        # This mustn't be too high since activations get too large otherwise
                        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
                        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
                        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                       'lower_bound': 2},
                        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
                            torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU
                        ]},
                        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}
    return mlp_prior_config


# TabICL v1 MLP prior configuration
def tabiclv1_mlp_config():
    mlp_prior_config = {"pre_sample_causes": True,
                        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed', 'uniform']},
                        'prior_mlp_scale_weights_sqrt': True,
                        'random_feature_rotation': True,
                        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                        "prior_mlp_hidden_dim": {
                            'distribution': 'meta_trunc_norm_log_scaled',
                            'min_mean': 5, 'max_mean': 130,
                            'round': True, 'lower_bound': 4
                        },

                        "num_layers": {
                            'distribution': 'meta_trunc_norm_log_scaled',
                            'min_mean': 1, 'max_mean': 6,
                            'round': True, 'lower_bound': 2
                        },
                        # This mustn't be too high since activations get too large otherwise
                        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
                        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
                        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                       'lower_bound': 2},
                        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
                            torch.nn.Tanh, torch.nn.Identity, torch.nn.LeakyReLU, torch.nn.ELU,
                            torch.nn.SELU, torch.nn.SiLU, torch.nn.ReLU, torch.nn.Softplus,
                            torch.nn.ReLU6, torch.nn.Hardtanh,
                            SignActivation, RBFActivation, ExpActivation, SqrtAbsActivation,
                            UnitIntervalIndicator, SineActivation, SquareActivation, AbsActivation,
                        ]},
                        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}
    return mlp_prior_config


def get_prior_config(max_features=100, n_samples=1024 + 128, mlp_prior_type='mothernet'):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """

    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        'heterogeneous_batches': False,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'prior_type': 'prior_bag',
        'prior_bag': {'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0}},
        'batch_size_per_subgroup': 4}

    if mlp_prior_type == 'mothernet':
        mlp_prior_config = mothernet_mlp_config()
    elif mlp_prior_type == 'tabiclv1' or mlp_prior_type == 'tabiclv2':
        mlp_prior_config = tabiclv1_mlp_config()
        if mlp_prior_type == 'tabiclv2':
            warning("Using tabiclv2 prior, but will run from another branch")
    else:
        raise ValueError(f"Unknown MLP prior type: {mlp_prior_type}")

    prior['mlp'] = mlp_prior_config

    gp_prior_config = {
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]},
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
    }

    prior['gp'] = gp_prior_config

    tree_prior_config = {
        # TreeSCM internally hardcodes: is_causal=False, num_layers=[1,2], hidden_dim=[3,9]
        "tree_model": "xgboost",
        "tree_depth_lambda": 0.5,
        "tree_n_estimators_lambda": 0.5,
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed', 'uniform']},
        "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
        "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
    }

    prior['tree'] = tree_prior_config

    prior['step_function'] = {
        'max_steps': 1,
        'sampling': 'uniform',
    }

    max_num_classes = 10
    classsification_prior = {
        "max_num_classes": max_num_classes,
        "num_classes": {'distribution': 'uniform_int', 'min': 2, 'max': max_num_classes},
        # "noise_type": "Gaussian",  # NN unused?!
        "balanced": False,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        "multiclass_type": 'rank',
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': .9,
        'num_features_sampler': 'uniform',
        'pad_zeros': True,
        'feature_curriculum': False,
    }
    prior['classification'] = classsification_prior

    dataloader = {
        "batch_size": 8,
        "total_steps": None,
        'min_eval_pos': 2,
        'random_n_samples': 0,
        'n_test_samples': 0,
    }

    openmlloader = {
        'valid_data': 'old',
        'max_samples': float('inf'),
        'pca': False,
        'val_csv_path': os.path.normpath(os.path.join(os.path.dirname(__file__), 'val_dataset', 'openml_old_datasets_val.csv')),
        'val_pkl_path': os.path.normpath(os.path.join(os.path.dirname(__file__), 'val_dataset', 'openml_old_datasets_val.pkl')),
    }

    prior['boolean'] = {
        'max_fraction_uninformative': 0.5,
        'p_uninformative': 0.5
    }

    return {'prior': prior, 'dataloader': dataloader, 'openmlloader': openmlloader}
