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


def evaluate_continuum(spec, **kwargs):

    continuum_flux = interpolate_linear([spec["wave"][0], spec["wave"][-1]],
                                        [spec["flux"][0], spec["flux"][-1]], spec["wave"])
    continuum = QTable({"wave": spec["wave"], "flux": continuum_flux})

    residuals = continuum["flux"] - spec["flux"]

    del continuum
    del spec

    if min(residuals) >= 0:
        return True
    else:
        return False


def get_continuum(spec, feature, **kwargs):
    '''
    Locate the maxima in the feature's lower bound. Continuum must be fitted
    to this point or a point at a higher wavelength.
    '''

    lower_region_mask = ((spec["wave"] < feature["lo_range_up"]) &
                         (spec["wave"] > feature["lo_range_lo"]))
    upper_region_mask = ((spec["wave"] < feature["up_range_up"]) &
                         (spec["wave"] > feature["up_range_lo"]))
    lower_region_mask_inds = np.where(lower_region_mask)[0]
    upper_region_mask_inds = np.where(upper_region_mask)[0]

    lower_region_maxima_ind = lower_region_mask_inds[spec["flux"][lower_region_mask].argmax()]
    upper_region_maxima_ind = upper_region_mask_inds[spec["flux"][upper_region_mask].argmax()]

    spec = spec[lower_region_maxima_ind:upper_region_maxima_ind]

    continuum_flux = interpolate_linear([spec["wave"][0], spec["wave"][-1]],
                                        [spec["flux"][0], spec["flux"][-1]],
                                        spec["wave"])
    continuum = QTable({"wave": spec["wave"], "flux": continuum_flux})
    return continuum


def locate_spectral_features(spec, features, **kwargs):
    for i, feature in enumerate(features):
        if feature["measure_flag"]:
            continuum = get_continuum(spec, feature)
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
