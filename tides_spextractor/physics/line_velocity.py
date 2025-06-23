'''
Author: Harry Addison
Created: 02/06/25
'''

import numpy as np
from tides_spextractor.physics.doppler import calc_doppler_vel, calc_doppler_vel_err


def calc_vel(data, rest_wl, model, kernel, keys=["x", "y", "y_err"], **kwargs):
    '''
    calc vel
    '''

    min_flux_ind = np.argmin(data[keys[1]])
    min_wl = data[keys[0]][min_flux_ind]
    min_wl_err = _estimate_min_wl_err(data, model, kernel, keys, **kwargs)
    
    vel = calc_doppler_vel(min_wl, rest_wl)
    vel_err = calc_doppler_vel_err(min_wl, rest_wl, min_wl_err)

    return vel, vel_err


def _estimate_min_wl_err(data, model, kernel, keys=["x", "y", "y_err"], n_samples=1000, **kwargs):

    samples = model.posterior_samples_f(data[keys[0]].value[:, np.newaxis],
                                        n_samples, kern=kernel.copy())
    samples = samples.squeeze()
    min_samples_ind = np.argmin(samples, axis=0)
    min_wl_err = np.std(data[keys[0]][min_samples_ind]) * data[keys[0]].unit
    return min_wl_err
