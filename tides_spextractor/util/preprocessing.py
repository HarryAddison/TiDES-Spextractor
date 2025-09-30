'''
Author: Harry Addison
Created: 17/02/2025

Functions for preprocessing the spectrum.
'''

import importlib.resources
import numpy as np
from tides_spextractor.physics import extinction
from tides_spextractor.physics import doppler
from tides_spextractor.util.conversions import normalise_data
from tides_spextractor.util.input_output import load_telluric_regions
from tides_spextractor.maths.interpolation import find_neighbours, interpolate_linear


from astropy.table import MaskedColumn, QTable
from warnings import warn


def remove_nans(spec, keys=["x", "y", "y_err"], **kwargs):
    mask = ~np.isnan(spec[keys[0]]) * ~np.isnan(spec[keys[1]]) * ~np.isnan(spec[keys[2]])
    return spec[mask]


def remove_masked_values(spec):
    if spec.mask is None:
        return spec
    if spec.mask is not None:
        mask = [False if any(row_mask) else True for row_mask in spec.mask]
        return spec[mask]


def remove_negative_flux(spec, keys=["x", "y", "y_err"], **kwargs):
    mask = spec[keys[1]] >= 0
    return spec[mask]


def remove_zero_flux(spec, keys=["x", "y", "y_err"], **kwargs):
    mask = spec[keys[1]] != 0
    return spec[mask]


def clean_spectrum(spec, remove_negative_fluxes=True, remove_zero_fluxes=True, **kwargs):
    spec = remove_nans(spec, **kwargs)
    spec = remove_masked_values(spec)

    if remove_negative_fluxes is True:
        spec = remove_negative_flux(spec, **kwargs)
    if remove_zero_fluxes is True:
        spec = remove_zero_flux(spec, **kwargs)

    return spec


def remove_tellurics(spec, telluric_path=None, z=None, **kwargs):

    # TODO Add the option to use different methods:
    #   - No removal/processing
    #   - Remove the spectrum data in the telluric regions

    tellurics = load_telluric_regions(telluric_path, z)

    masks = []
    for telluric in tellurics:
        # Remove the data points
        # Invert the mask so that when the masks are multiplied into one mask,
        # the masked regions from each individual mask remain.
        masks.append(~((spec["wave"] >= telluric["lower_wl"]) & (spec["wave"] <= telluric["upper_wl"])))

    mask = np.prod(np.array(masks), axis=0, dtype=bool)
    spec["flux"] = MaskedColumn(spec["flux"])
    spec["flux_err"] = MaskedColumn(spec["flux_err"])
    spec["flux"].mask = ~mask  # Invert the mask from previous inversion
    spec["flux_err"].mask = ~mask
    return spec


def deredden_spectrum(spec, mwebv, ebv, rv=3.1, **kwargs):

    if mwebv is not None:
        spec = extinction.deredden(spec, mwebv, rv, **kwargs)

    if ebv is not None:
        spec = extinction.deredden(spec, ebv, rv, **kwargs)

    return spec


def deredshift_spectrum(spec, z, keys=["x"], **kwargs):
    if z is not None:
        spec[keys[0]] = doppler.deredshift(spec[keys[0]], z, **kwargs)
    if z is None:
        warn("No redshift provided. Spectrum could not be converted to rest frame. "
             "Assuming spectrum is already in its rest frame.")
    return spec


def prune_spectrum(spec, min_wavelength=None, max_wavelength=None, 
                   keys=["x", "y", "y_err"], **kwargs):
    if min_wavelength is None:
        min_wavelength = min(spec[keys[0]])
    if max_wavelength is None:
        max_wavelength = max(spec[keys[0]])
    mask = (spec[keys[0]] >= min_wavelength) & (spec[keys[0]] <= max_wavelength)
    return spec[mask]


def remove_outliers(spec):

    raise NotImplementedError("'remove_outliers' not implemented.")

    ds_spec = downsample()

    gpr_spec = gpr_model()

    mask = spec["flux"] < gpr_spec["mean"] + x * np.sqrt(gpr_spec["var"])

    return spec[mask] 
    

def normalise_spectrum(spec, normalisation_method="max", normalisation_wavelength=None,
                       keys=["x", "y", "y_err"], **kwargs):
    if normalisation_method == "max":
        val = max(spec[keys[1]])
    elif normalisation_method == "flux_at_wl":
        if normalisation_wavelength is None:
            warn("No normalisation wavelength provided. Normalisation skipped!")
            return spec
        else:
            # Case where the desired wavelength is in the data array
            mask = spec[keys[0]] == normalisation_wavelength
            if np.sum(mask) == 1:
                val = spec[keys[1]][mask][0]
            # Case where desired wavelength is in the data array more than once
            elif np.sum(mask) > 1:
                raise ValueError("Duplicate wavelengths exist in the spectrum data.")
            # Case where the desired wavelength isnt in the data array.
            # Interpolate the flux from the nearest neighbours.
            else:
                lower_ind, upper_ind = find_neighbours(spec[keys[0]], normalisation_wavelength)
                if lower_ind is not None and upper_ind is not None:
                    val = interpolate_linear(spec[keys[0]][[lower_ind, upper_ind]],
                                             spec[keys[1]][[lower_ind, upper_ind]],
                                             normalisation_wavelength)
                else:
                    warn("Normalisation wavelength provided is not within the wavelength range of the spectrum. "
                         "Normalisation skipped!")
                    return spec
    else:
        warn(f"Normalisation method '{normalisation_method}' is not a valid method. "
             "Must be either 'max' or 'flux_at_wl'. Normalisation skipped!")

    spec[keys[1]] = normalise_data(spec[keys[1]], val.value)
    spec[keys[2]] = normalise_data(spec[keys[2]], val.value)

    return spec


def remove_masked_rows(table):
    # TODO Move to a different file/folder

    # If no columns are masked at all, return table as-is
    if not any(hasattr(col, 'mask') for col in table.itercols()):
        return table

    # Build a boolean mask for all rows: True if the row has *no* masked values
    unmasked_mask = np.ones(len(table), dtype=bool)

    for col in table.itercols():
        if hasattr(col, 'mask'):
            # col.mask could be False (scalar) or array — normalize to array
            col_mask = np.array(col.mask, dtype=bool)
            if col_mask.shape == ():  # scalar mask
                if col_mask:
                    unmasked_mask[:] = False  # entire column is masked
            else:
                unmasked_mask &= ~col_mask  # keep only unmasked rows

    return table[unmasked_mask]