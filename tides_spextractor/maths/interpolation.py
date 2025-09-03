'''
Author: Harry Addison
Created: 13/05/2025
'''

import numpy as np
from astropy.table import QTable, MaskedColumn


def find_neighbours(data, val):
    if val <= data[0]:
        return None, 0
    if val >= data[-1]:
        return -1, None
    else:
        # Find the position that the val would be placed in the array
        ind = np.searchsorted(data, val)
        lower_neigh_ind = ind - 1
        upper_neigh_ind = ind

        return lower_neigh_ind, upper_neigh_ind


def interpolate_linear(x_data, y_data, x_new):

    m = (y_data[0] - y_data[1]) / (x_data[0] - x_data[1])
    y_new = (x_new - x_data[0]) * m + y_data[0]
    return y_new


def downsample_spec_data(data, ds_factor, keys=["x", "y", "y_err"], **kwargs):

    n_bins = int(len(data[keys[0]]) / ds_factor)
    bin_edges= np.linspace(data[keys[0]].min(), data[keys[0]].max(),
                           (n_bins + 1), endpoint=True)

    downsampled_data = QTable()
    downsampled_data[keys[0]] = MaskedColumn([], dtype=float, unit=data[keys[0]].unit)
    downsampled_data[keys[1]] = MaskedColumn([], dtype=float, unit=data[keys[1]].unit)
    downsampled_data[keys[2]] = MaskedColumn([], dtype=float, unit=data[keys[2]].unit)

    for i in range(n_bins):
        if i == n_bins - 1:  # last bin
            mask = (data[keys[0]] >= bin_edges[i]) & (data[keys[0]] <= bin_edges[i+1])
        else:
            mask = (data[keys[0]] >= bin_edges[i]) & (data[keys[0]] < bin_edges[i+1])
        bin_data = data[mask]

        if len(bin_data) > 1:
            bin_wl = np.mean(bin_data[keys[0]])
            bin_flux = np.mean(bin_data[keys[1]]*bin_data[keys[0]]) / bin_wl
            bin_flux_err = np.sqrt((((bin_data[keys[0]] * bin_data[keys[2]]) / bin_wl)**2).sum())
            downsampled_data.add_row([bin_wl, bin_flux, bin_flux_err])
        elif len(bin_data) == 1:
            downsampled_data.add_row([bin_data[keys[0]][0], bin_data[keys[1]][0],
                                      bin_data[keys[2]][0]])

    return downsampled_data


def convolveTo15A(x,nBins=60):
    return np.convolve(x, np.ones(nBins), 'valid') / nBins


def convolveNoiseTo15A(x,nBins=60):
    for m in range(len(x)-(nBins-1)):
        yield np.sqrt(sum((np.ones(nBins) * x[m:m+nBins])**2)) / nBins


def bin_spec_data(data, bin_width, keys=["x", "y", "y_err"], **kwargs):

    # convert data to dimensionless quantities
    wave = data[keys[0]].value
    signal = data[keys[1]].value
    noise = data[keys[2]].value

    n_bins = int((max(wave) - min(wave)) / bin_width) + 1

    if n_bins != None:
        # rebin the data
        # Convert fluxes to flux * wave before binning then convert back.
        wave15 = convolveTo15A(wave, n_bins)
        signal = convolveTo15A(signal * wave, n_bins) / wave15
        noise = np.array(list(convolveNoiseTo15A(noise * wave, n_bins))) / wave15

    data_binned = QTable(data=[wave15 * data[keys[0]].unit,
                               signal * data[keys[1]].unit,
                               noise * data[keys[2]].unit], names=keys)
    return data_binned