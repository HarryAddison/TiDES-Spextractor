'''
Author: Harry Addison
Created: 13/05/2025
'''

import GPy
import numpy as np
from astropy.table import QTable
from astropy import units as u


def make_model(data):
    
    '''
    Model the spectrum using Gaussian process regression.

    Inputs:
    > "data" = Data of the spectrum. Must be in a dataframe with the
               column names "WAVE" (wavelength), "FLUX", "ERR_FLUX"
               (flux error).
    > "ds_factor" = Downsampling factor.

    Outputs:
    > "model" and "kernal" of the GPR model
    > "ds_data" = downsampled data
    '''

    x = np.array(data["wave"].value)
    y = np.array(data["flux"].value)
    y_err = np.array(data["flux_err"].value)

    # Using brownian kernel.
    kernel = GPy.kern.Matern32(1, variance=0.001, lengthscale=1)

    # Adding noise to kernel using flux errors.
    var_mat = y_err * y_err * np.eye(len(y_err))
    kern_err = GPy.kern.Fixed(1, var_mat)
    kern = kernel + kern_err

    # make the model
    model = GPy.models.GPRegression(x[:, np.newaxis], y[:, np.newaxis], kern)
    model["Gaussian.noise.variance"][0] = 0.01

    model[".*fixed.variance"].constrain_fixed()
    model.Gaussian_noise.fix(1e-6)
    model.optimize(optimizer="bfgs")

    kernel.variance = kern.Mat32.variance
    kernel.lengthscale = kern.Mat32.lengthscale

    return model, kernel


def model_values(model, kernel, wl):
    '''
    Obtain the data for the model spectrum for the provided wavelengths.

    Inputs:
    > "model" and "kernal" = model and kernal from the GPR model. These
                             are outputs of the function "make_model".
    > "wl" = wavelengths to obtain the model flux for. Is an array.

    Outputs:
    > "model_spec" = Table of the "wl" with its corresponding flux and
                     flux error from the input model.
                     Table/dataframe has the column names: "WAVE"
                     (wavelength), "FLUX", "ERR_FLUX" (flux err).
    '''

    mean, var = model.predict(wl.value[:, np.newaxis], kern=kernel.copy())

    flux_unit = u.erg / (u.cm**2 * u.s)
    model_flux = mean.squeeze() * flux_unit
    model_flux_var = var.squeeze()

    model_flux_err = np.sqrt(model_flux_var) * flux_unit

    model_spec = QTable(data=[wl, model_flux, model_flux_err],
                        names=["wave", "flux", "flux_err"])

    return model_spec
