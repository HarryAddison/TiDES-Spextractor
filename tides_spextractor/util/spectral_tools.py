'''
Author: Harry Addison
Created: 15/05/2025
'''

from tides_spextractor.maths.interpolation import interpolate_linear
from astropy.table import QTable
import numpy as np


def identify_present_features(features, spec_wls, **kwargs):
    within_wl_range_mask = ((features["lo_range_lo"]>= spec_wls.min()) &
                            (features["up_range_up"]<= spec_wls.max()))
    features["measure_flag"][~within_wl_range_mask] = False
    features["comment"][~within_wl_range_mask] = "Feature outside of spectrum wavelengths"
    return features


def identify_features_coincident_with_tellurics(features, telluric_regions, **kwargs):
    for telluric_region in telluric_regions:
        coincide_telluric_mask = ((telluric_region["lower_wl"] <= features["up_range_up"]) &
                                         (telluric_region["upper_wl"] >= features["lo_range_lo"]))
        features["comment"][coincide_telluric_mask] = "Feature coincides with a telluric region"
    return features


def find_suitable_continuum_region_bounds(spec, feature, keys=["x", "y"], **kwargs):
    lower_region_mask = ((spec[keys[0]] < feature["lo_range_up"]) &
                         (spec[keys[0]] > feature["lo_range_lo"]))
    upper_region_mask = ((spec[keys[0]] < feature["up_range_up"]) &
                         (spec[keys[0]] > feature["up_range_lo"]))
    lower_region_inds = np.where(lower_region_mask)[0]
    upper_region_inds = np.where(upper_region_mask)[0]

    return lower_region_inds, upper_region_inds


def find_best_continuum(spec, lower_region_inds, upper_region_inds, keys=["x", "y"], **kwargs):

    wl = spec[keys[0]]
    flux = spec[keys[1]]

    # Wavelengths and flux data seperated into lower and upper continuum bound regions
    wls_low = wl[lower_region_inds]
    wls_up = wl[upper_region_inds]
    fluxes_low = flux[lower_region_inds]
    fluxes_up = flux[upper_region_inds]

    # Search the pairs of points that have "best" continuum.
    # Best = continuum that is above spectrum and maximises wavelength span.
    best_continuum_wl = None
    best_continuum_flux = None
    best_width = -np.inf

    # loop points in upper and lower continuum bound regions, covering all pairs
    for i, wl_low in enumerate(wls_low):
        for j, wl_up in enumerate(wls_up):
            if wl_up == wl_low:
                continue  # avoid divide by zero

            # spectrum between the two wl points
            mask = (wl >= wl_low) & (wl <= wl_up)
            wl_segment = wl[mask]
            flux_segment = flux[mask]

            flux_continuum = interpolate_linear([wl_low, wl_up],
                                                [fluxes_low[i], fluxes_up[j]],
                                                wl_segment)

            # check if continuum is above the spectrum
            if np.all(flux_continuum >= flux_segment):
                width = abs(wl_up - wl_low)
                if width > best_width:
                    best_width = width
                    best_continuum_wl = wl_segment
                    best_continuum_flux = flux_continuum

    if best_continuum_flux is not None:
        return QTable({keys[0]: best_continuum_wl, keys[1]: best_continuum_flux})
    else:
        return None


def get_continuum(spec, feature, **kwargs):
    '''
    '''
    lo_region_inds, up_region_inds = find_suitable_continuum_region_bounds(spec, feature, **kwargs)
    continuum = find_best_continuum(spec, lo_region_inds, up_region_inds, **kwargs)

    return continuum


def locate_spectral_features(spec, features, **kwargs):
    for i, feature in enumerate(features):
        if feature["measure_flag"]:
            continuum = get_continuum(spec, feature, **kwargs)
            if continuum:
                features[i]["continuum"] = continuum
            else:
                features[i]["comment"] = "A valid continuum could not be found"

    return features


def get_feature_data(data, continuum_data, keys=["x", "y"], **kwargs):

    feature_wl_mask = ((data[keys[0]] >= min(continuum_data[keys[0]])) &
                       (data[keys[0]] <= max(continuum_data[keys[0]])))
    feature_data = data[feature_wl_mask]
    return feature_data


def normalise_by_continuum(data, continuum_data, keys=["x", "y"], **kwargs):

    norm_flux = data[keys[1]] - continuum_data[keys[1]]
    data[keys[1]] = norm_flux
    return data
