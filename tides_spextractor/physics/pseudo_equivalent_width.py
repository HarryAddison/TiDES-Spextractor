'''
Author: Harry Addison
Created: 03/06/25
'''


from tides_spextractor.maths.integration import * 
from scipy.integrate import simpson


def calc_pew(data, keys=["x", "y", "y_err"], **kwargs):
    pew = _calc_pew(data, keys=keys)
    pew_err = _calc_pew_error(data, keys=keys)
    return pew, pew_err


def _calc_pew(data, keys=["x", "y"]):
    integrand = 1 - data[keys[1]].value
    pew = simpson(integrand, data[keys[0]].value)
    return pew


def _calc_pew_error(data, keys=["x", "y", "y_err"]):

    n_intervals = len(data) - 1

    if n_intervals == 2:
        pew_err = calc_simpson_2_points_error(data[keys[0]].value, data[keys[2]].value)
    elif n_intervals % 2 == 1:
        pew_err = calc_simpson_odd_interval_error(data[keys[0]].value, data[keys[2]].value)
    elif n_intervals % 2 == 0:
        pew_err = calc_simspon_even_interval_error(data[keys[0]].value, data[keys[2]].value)
    return pew_err
