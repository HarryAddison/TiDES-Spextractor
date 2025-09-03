'''
Author: Harry Addison
Created: 02/06/25
'''
import numpy as np
import torch
from tides_spextractor.physics.doppler import calc_doppler_vel, calc_doppler_vel_err
from tides_spextractor.util.conversions import convert_np_to_tensor


def calc_vel(data, rest_wl, model, keys=["x", "y", "y_err"], **kwargs):
    '''
    calc vel
    '''

    min_flux_ind = np.argmin(data[keys[1]])
    min_wl = data[keys[0]][min_flux_ind]
    min_wl_err = _estimate_min_wl_err(data, model, keys, **kwargs)

    vel = calc_doppler_vel(min_wl, rest_wl)
    vel_err = calc_doppler_vel_err(min_wl, rest_wl, min_wl_err)
    return vel, vel_err


def _estimate_min_wl_err(data, model, keys=["x", "y", "y_err"], n_samples=1000, **kwargs):

    with torch.no_grad():
        x = convert_np_to_tensor(data[keys[0]].value)
        posterior = model(x)
        samples = posterior.sample(torch.Size([n_samples]))
        samples = samples.detach().cpu().numpy()

    min_samples_ind = np.argmin(samples, axis=1)
    min_wl_err = np.std(data[keys[0]][min_samples_ind])
    return min_wl_err
