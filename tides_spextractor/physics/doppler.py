'''
Author: Harry Addison
Created: 17/02/2025
'''


c = 299792.458 # km/s


def deredshift(wls, z):
    wls /= (z + 1)
    return wls


def calc_doppler_vel(obs_wl, rest_wl):
    wl_ratio_2 = (obs_wl / rest_wl)**2
    vel = -c * (wl_ratio_2 - 1) / (wl_ratio_2 + 1)
    return vel


def calc_doppler_vel_err(obs_wl, rest_wl, obs_wl_err):
    vel_err = (((4 * c * rest_wl**2 * obs_wl) / (rest_wl**2 + obs_wl**2)**2) * obs_wl_err)
    return vel_err