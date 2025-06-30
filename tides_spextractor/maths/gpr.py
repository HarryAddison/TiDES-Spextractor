'''
Author: Harry Addison
Created: 13/05/2025
'''


# I was testing the spextractor gpr code. I tried it on the template spectrum
# and it did not model it too well. I have not yet tried it on the 4most-ified spectrum.
# My pgytorch gpr also did not do too well at modelling the template spectrum, but it did better
# than the spextractor gpr.
# The template spectrum has errors of 0.0 but I noticed that the processed spectrum
# has non zero errors, which is incorrect. I think this might be affecting the gpr models.
# It also suggests that there is a bug in the preprocessing code as its making up errors somehow.


import torch
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import linear_operator
import numpy as np
from astropy.table import QTable
from astropy import units as u
from tides_spextractor.util.conversions import convert_np_to_tensor
from tides_spextractor.util.preprocessing import remove_masked_rows


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        kernel_const = gpytorch.kernels.ConstantKernel(constant_constraint=gpytorch.constraints.Interval(0, 1))
        kernel_rbf = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(100, 500))
        kernel_mat12 = gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=gpytorch.constraints.Interval(20, 100))
        kernel_mat52 = gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=gpytorch.constraints.Interval(20, 100))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveKernel(gpytorch.kernels.ScaleKernel(kernel_const),
                                                                                         gpytorch.kernels.ScaleKernel(kernel_rbf),
                                                                                         gpytorch.kernels.ScaleKernel(kernel_mat52),
                                                                                         gpytorch.kernels.ScaleKernel(kernel_mat12)))


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_model(data):

    data = remove_masked_rows(data)

    x = convert_np_to_tensor(data["wave"].value)
    y = convert_np_to_tensor(data["flux"].value)
    y_err = np.array(data["flux_err"].value)
    var = convert_np_to_tensor(y_err * y_err)
    # make model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=var,
                                                                   learn_additional_noise=True)
    model = ExactGPModel(x, y, likelihood)
    
    # Train
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.max_cg_iterations(5000):
        for i in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
    return model, likelihood


def model_values(model, likelihood, data):

    x = convert_np_to_tensor(data["wave"].value)

    # Predict
    with torch.no_grad():
        with gpytorch.settings.max_cg_iterations(5000):
            model.eval()
            likelihood.eval()
            pred = likelihood(model(x))

    y = np.asarray(pred.mean.detach().cpu().numpy()).flatten() * data["flux"].unit
    y_err = np.asarray(pred.stddev.detach().cpu().numpy()).flatten()  * data["flux"].unit

    model_spec = QTable(data=[data["wave"], y, y_err],
                        names=["wave", "flux", "flux_err"])
    return model_spec
