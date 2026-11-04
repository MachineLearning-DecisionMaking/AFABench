import time

import gpytorch
import torch

# PATCH(afabench): import path of the vendored copy.
from afabench.components.methods.discriminative.l2m.tabicl_v1_prior.utils_torch import default_device
from .distributions import parse_distributions, sample_distributions


# We will use the simplest form of GP model, exact inference


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(x, y, hyperparameters):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.e-9))
    model = ExactGPModel(x, y, likelihood)
    model.likelihood.noise = torch.ones_like(model.likelihood.noise) * hyperparameters["noise"]
    model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale) * hyperparameters["outputscale"]
    model.covar_module.base_kernel.lengthscale = torch.ones_like(model.covar_module.base_kernel.lengthscale) * \
                                                 hyperparameters["lengthscale"]
    return model, likelihood


class GPPrior:
    def __init__(self, config=None, batch_size_per_subgroup=None):
        self.config = parse_distributions(config or {})
        self.batch_size_per_subgroup = batch_size_per_subgroup

    def get_batch(self, batch_size, n_samples, num_features, device=default_device,
                  equidistant_x=False, fix_x=None, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            if self.batch_size_per_subgroup is not None:
                assert batch_size % self.batch_size_per_subgroup == 0, \
                    "Batch size must be divisible by batch_size_per_subgroup"
                x_list, y_list = [], []
                for i in range(0, batch_size, self.batch_size_per_subgroup):
                    sub_x, sub_y = self._generate_subgroup(
                        self.batch_size_per_subgroup, n_samples, num_features, device,
                        equidistant_x, fix_x)
                    x_list.append(sub_x)
                    y_list.append(sub_y)
                x = torch.cat(x_list, dim=1)  # (T, batch_size, H)
                y = torch.cat(y_list, dim=1)
                return x, y
            else:
                return self._generate_subgroup(
                    batch_size, n_samples, num_features, device, equidistant_x, fix_x)

    def _generate_subgroup(self, sub_batch_size, n_samples, num_features, device,
                           equidistant_x=False, fix_x=None):
        with torch.no_grad():
            assert not (equidistant_x and (fix_x is not None))
            is_fitted = False
            while not is_fitted:
                hypers = sample_distributions(self.config)
                with gpytorch.settings.fast_computations(*hypers.get('fast_computations', (True, True, True))):
                    if equidistant_x:
                        assert num_features == 1
                        x = torch.linspace(0, 1., n_samples).unsqueeze(0).repeat(sub_batch_size, 1).unsqueeze(-1)
                    elif fix_x is not None:
                        assert fix_x.shape == (n_samples, num_features)
                        x = fix_x.unsqueeze(0).repeat(sub_batch_size, 1, 1).to(device)
                    else:
                        if hypers['sampling'] == 'uniform':
                            x = torch.rand(sub_batch_size, n_samples, num_features, device=device)
                        else:
                            x = torch.randn(sub_batch_size, n_samples, num_features, device=device)
                    model, likelihood = get_model(x, torch.Tensor(), hypers)
                    model.to(device)

                    error_times = 5
                    try:
                        with gpytorch.settings.prior_mode(True):
                            model, likelihood = get_model(x, torch.Tensor(), hypers)
                            model.to(device)

                            d = model(x)
                            d = likelihood(d)
                            sample = d.sample().transpose(0, 1)
                            is_fitted = True
                    except RuntimeError:  # This can happen when torch.linalg.eigh fails. Restart with new init resolves this.
                        print('GP Fitting unsuccessful, retrying.. ')
                        print(x)
                        print(self.config)
                        # clear the memory
                        torch.cuda.empty_cache()
                        del model, likelihood, d
                        time.sleep(1)
                        error_times -= 1
                        assert error_times == 0

            if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()):
                print({"noise": hypers['noise'], "outputscale": hypers['outputscale'],
                       "lengthscale": hypers['lengthscale'], 'batch_size': sub_batch_size})

            # TODO: Multi output
            return x.transpose(0, 1), sample  # (T, B, H), (T, B)
