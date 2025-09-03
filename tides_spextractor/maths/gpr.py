'''
Author: Harry Addison
Created: 13/05/2025
'''


import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
from tides_spextractor.util.conversions import convert_np_to_tensor
from tides_spextractor.util.preprocessing import remove_masked_rows


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        kernel_rbf = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(100, 200))
        kernel_mat52 = gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=gpytorch.constraints.Interval(5, 10))

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveKernel(gpytorch.kernels.ScaleKernel(kernel_mat52),
                                                                                         gpytorch.kernels.ScaleKernel(kernel_rbf)))


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_model(data, keys=["x", "y", "y_err"], gp_training_iterations=100,
               patience=20, min_delta_loss=1e-4, plot_loss=False, **kwargs):

    # Use GPU if available else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = remove_masked_rows(data)

    x = convert_np_to_tensor(data[keys[0]].value)
    y = convert_np_to_tensor(data[keys[1]].value)
    y_err = np.array(data[keys[2]].value)
    var = convert_np_to_tensor(y_err * y_err)
    # make model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=var,
                                                                   learn_additional_noise=True)
    model = ExactGPModel(x, y, likelihood)

    # Move the needed variables to the "device"
    x.to(device)
    y.to(device)
    var.to(device)
    likelihood.to(device)
    model.to(device)

    # Train
    model.train()
    likelihood.train()

    # Optimise
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    wait = 0
    best_loss = np.inf

    with gpytorch.settings.max_cg_iterations(5000):
        for i in range(gp_training_iterations):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            # Early stop if loss is not improving much or is nan value
            if type(loss.item()) != float and type(loss.item()) != int:
                print("non float/int value. Early stop")
                break
            if best_loss - loss.item() > min_delta_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if plot_loss == True:
        plt.figure()
        plt.plot(losses, c="k")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        # plt.show()

    return model, likelihood


def model_values(model, likelihood, data, keys=["x", "y", "y_err"], **kwargs):

    device = next(model.parameters()).device  # Get the model's device (GPU/CPU)

    x = convert_np_to_tensor(data[keys[0]].value).to(device)  # Move to device

    # Predict
    with torch.no_grad():
        with gpytorch.settings.max_cg_iterations(5000):
            model.eval()
            likelihood.eval()
            pred = likelihood(model(x))

    y = np.asarray(pred.mean.detach().cpu().numpy()).flatten() * data[keys[1]].unit
    y_err = np.asarray(pred.stddev.detach().cpu().numpy()).flatten() * data[keys[2]].unit

    model_spec = QTable(data=[data[keys[0]], y, y_err],
                        names=[keys[0], keys[1], keys[2]])
    return model_spec
