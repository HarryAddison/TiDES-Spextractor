'''
Author: Harry Addison
Created: 19/06/25
'''

import numpy as np


def calc_simpson_2_points_error(x, y_err):
    '''
    Simpsons with 2 points: I = 0.5 x (x2 - x1) * (y1 + y2)
    The error of this is given by using the formula: 
        sigma_I = sqrt[sum(dI/dy_err_n)**2 * sigma_y_err_n)]
    where the differential is partial and n is the number of data points
    '''

    error = (0.5 * x[0] * x[1]) * np.sqrt((y_err[0]**2 + y_err[1]**2))
    return error


def calc_simspon_even_interval_error(x, y_err):

    end_ind = len(x) - 1  # set to last index
    slice_1 = slice(0, end_ind - 1, 2)  # 2i where i = N/2 - 1
    slice_2 = slice(1, end_ind, 2)  # 2i + 1
    slice_3 = slice(2, end_ind + 1, 2)  # 2i + 2, end_ind + 1 as "stop" ind is not inclusive but want it to be.

    h = np.diff(x)
    h_frac = h[slice_1] / h[slice_2]
    h_prod = h[slice_1] * h[slice_2]
    h_sum = h[slice_1] + h[slice_2]

    term_com = (h_sum) / 6
    term_1 = 2 - (1 / h_frac)
    term_2 = h_sum**2 / h_prod
    term_3 = 2 - h_frac

    error = np.sqrt(sum(term_com**2 * (term_1**2 * y_err[slice_1]**2 +
                                       term_2**2 * y_err[slice_2]**2 +
                                       term_3**2 * y_err[slice_3]**2)))

    return error


def calc_simpson_odd_interval_error(x, y_err):

    # first N - 1 intervals
    result = calc_simspon_even_interval_error(x, y_err)**2  # Want the error^2

    # last interval
    last_ind = len(x) - 1
    h = np.diff(x[-3:])  # last two intervals
    h_prod = h[0] * h[1]
    h_sum = h[0] + h[1]

    term_1 = ((2 * h[1]**2) + (3 * h_prod)) / (6 * h_sum)
    term_2 = (h[1]**2 + 3 * h_prod) / (6 * h[0])
    term_3 = (h[1]**3) / (6 * h[0] * h_sum)

    result += (term_1**2 * y_err[last_ind]**2 + term_2**2 * y_err[last_ind - 1]**2 +
               term_3**2 * y_err[last_ind - 2]**2)

    error = np.sqrt(result)
    return error
