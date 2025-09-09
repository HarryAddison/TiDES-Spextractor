'''
Author: Harry Addison
Created: 03/06/25
'''


from scipy.integrate import simpson
from tides_spextractor.maths.integration import *
from tides_spextractor.util.conversions import convert_np_to_tensor
import torch


def calc_pew(data, continuum_data, model, keys=["x", "y", "y_err"], **kwargs):
    integrand = 1 - (data[keys[1]] / continuum_data[keys[1]])

    pew = simpson(integrand, data[keys[0]].value) * data[keys[0]].unit
    pew_err = _calc_pew_error(data, continuum_data, model, keys, **kwargs)  # Has unit already
    return pew, pew_err


def _calc_integration_pew_err(data, continuum_data, keys=["x", "y", "y_err"], **kwargs):

    integrand_err = data[keys[2]] / continuum_data[keys[1]]
    n_intervals = len(integrand_err) - 1

    if len(data) < 2:
        raise ValueError("Not enough values provided to calculate simpson's error."
                         "Atleast 2 data points are required.")
    elif n_intervals % 2 == 1:
        pew_err = calc_simpson_odd_interval_error(data[keys[0]], integrand_err)
    elif n_intervals % 2 == 0:
        if len(data) == 2:
            pew_err = calc_simpson_2_points_error(data[keys[0]], integrand_err)
        else:
            pew_err = calc_simspon_even_interval_error(data[keys[0]], integrand_err)
    return pew_err


def _calc_gpr_model_pew_err(data, continuum_data, model, keys=["x", "y", "y_err"], n_samples=1000, **kwargs):
    with torch.no_grad():
        x = convert_np_to_tensor(data[keys[0]].value)
        posterior = model(x)
        samples = posterior.sample(torch.Size([n_samples]))
        samples = samples.detach().cpu().numpy()

    integrands = 1 - (samples / continuum_data[keys[1]].value)
    pews = simpson(integrands, data[keys[0]].value)
    pew_err = np.std(pews) * data[keys[0]].unit

    return pew_err


def _calc_pew_error(data, continuum_data, model, keys=["x", "y", "y_err"], n_samples=1000, **kwargs):

    # There are three main errors in the pEW: integration error, GPR model error,
    # and continuum error.
    # The first two are easy to deal with but the error in the continuum is not.
    # TODO One way I can add continuum error is to produce many spectra from the GPR model,
    # fit the continuum for each, and then calculate the pEW, then take the std. dev. of all
    # the measurements. I'm not sure how feasible this method is in terms of computation efficiency
    # or if the error would be significant compared to the other errors. 

    integration_err = _calc_integration_pew_err(data, continuum_data, keys, **kwargs)
    gpr_model_error = _calc_gpr_model_pew_err(data, continuum_data, model, keys, n_samples, **kwargs)

    err = integration_err + gpr_model_error

    return err
