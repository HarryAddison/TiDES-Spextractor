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
    var = y_err * y_err

    # Kernel
    mat32_kern = GPy.kern.Matern32(1, variance=1, lengthscale=5)
    kernel = mat32_kern

    # make the model and put some contraints on hyper-parameters
    model = GPy.models.GPHeteroscedasticRegression(x[:, np.newaxis], y[:, np.newaxis], kernel)
    model['.*het_Gauss.variance'] = var[:, np.newaxis]
    model['.*het_Gauss.variance'].constrain_bounded(1e-6, 0.2)
    
    # Optimise
    model.optimize(optimizer="bfgs")
    print(model)

    # obtain the optimised variance and lengthscales
    print(kernel)

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
