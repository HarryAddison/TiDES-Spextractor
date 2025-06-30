'''
Author: Harry Addison
Created: 17/02/2025

Functions for preprocessing the spectrum.
'''

import importlib_resources
import numpy as np
from tides_spextractor.physics import deredden
from tides_spextractor.physics import doppler
from tides_spextractor.util.conversions import normalise_data
from tides_spextractor.util.input_output import load_telluric_regions
from tides_spextractor.maths.interpolation import find_neighbours, interpolate_linear


from astropy.table import QTable
from warnings import warn


def remove_nans(spec):
    mask = ~np.isnan(spec["wave"]) * ~np.isnan(spec["flux"]) * ~np.isnan(spec["flux_err"])
    return spec[mask]


def remove_masked_values(spec):
    if spec.mask is None:
        return spec
    if spec.mask is not None:
        mask = [False if any(row_mask) else True for row_mask in spec.mask]
        return spec[mask]


def remove_negative_flux(spec):
    mask = spec["flux"] >= 0
    return spec[mask]


def remove_zero_flux(spec):
    mask = spec["flux"] != 0
    return spec[mask]


def clean_spectrum(spec, remove_negative_fluxes=True, remove_zero_fluxes=True, **kwargs):
    spec = remove_nans(spec)
    spec = remove_masked_values(spec)

    if remove_negative_fluxes is True:
        spec = remove_negative_flux(spec)
    if remove_zero_fluxes is True:
        spec = remove_zero_flux(spec)

    return spec


def remove_tellurics(spec, telluric_path=None, z=None, **kwargs):

    # TODO Decide if this way of dealing with the tellurics is good.
    # Improvements/other ideas include:
    # - Taking an average flux value and using that for the interpolation
    #   rather than the point at the edge of the region.
    # - Removing the fluxes in the telluric regions and letting the GPR model
    #   fill in the blank fluxes.

    tellurics = load_telluric_regions(telluric_path, z)

    # TODO Make the functionaliy in the for loop a different function(s)
    for telluric in tellurics:
        # check telluric is within spectrum.
        # TODO Add the edge case of when the telluric boundary wavelengths
        # are equal to the max/min wavelength of the spectrum.
        min_wl = min(spec["wave"])
        max_wl = max(spec["wave"])

        if min_wl > telluric["lower_wl"]:
            if min_wl > telluric["upper_wl"]:
                # Telluric at lower wavelengths than spectrum
                continue
            elif min_wl < telluric["upper_wl"]:
                # Telluric partly overlaps with the lower wavelength spectrum
                # Remove the data overlapping with the telluric region
                mask = spec["wave"] > telluric["upper_wl"]
                spec = spec[mask]
        if max_wl < telluric["upper_wl"]:
            if max_wl < telluric["lower_wl"]:
                # Telluric at higher wavelengths than spectrum
                continue
            if max_wl > telluric["lower_wl"]:
                # Telluric partly overlaps with the higher wavelength spectrum
                # Remove the data overlapping with the telluric region
                mask = spec["wave"] < telluric["lower_wl"]
                spec = spec[mask]
        else:
            # Telluric within the spectrum
            # Replace the fluxes with interpolated values
            mask = (spec["wave"] >= telluric["lower_wl"]) & (spec["wave"] <= telluric["upper_wl"])
            ind = np.nonzero(mask)[0]

            # Interpolate between the points either side of the masked region.
            spec["flux"][ind] = interpolate_linear(spec["wave"][[ind[0] - 1, ind[-1] + 1]],
                                                   spec["flux"][[ind[0] - 1, ind[-1] + 1]],
                                                   spec["wave"][ind])

    return spec


def deredden_spectrum(spec, mwebv, ebv, rv=3.1):

    if mwebv is not None:
        spec = deredden.deredden(spec, mwebv, rv)

    if ebv is not None:
        spec = deredden.deredden(spec, ebv, rv)

    return spec


def deredshift_spectrum(spec, z, keys=["x"], **kwargs):
    if z is not None:
        spec[keys[0]] = doppler.deredshift(spec[keys[0]], z, **kwargs)
    if z is None:
        warn("No redshift provided. Spectrum could not be converted to rest frame. "
             "Assuming spectrum is already in its rest frame.")
    return spec


def prune_spectrum(spec, min_wavelength=None, max_wavelength=None):
    if min_wavelength is None:
        min_wavelength = min(spec["wave"])
    if max_wavelength is None:
        max_wavelength = max(spec["wave"])

    mask = (spec["wave"] >= min_wavelength) & (spec["wave"] <= max_wavelength)
    return spec[mask]


def remove_outliers(spec):

    raise NotImplementedError("'remove_outliers' not implemented.")

    ds_spec = downsample()

    gpr_spec = gpr_model()

    mask = spec["flux"] < gpr_spec["mean"] + x * np.sqrt(gpr_spec["var"])

    return spec[mask] 
    



def normalise_spectrum(spec, normalisation_method="max", normalisation_wavelength=None, **kwargs):
    if normalisation_method == "max":
        val = max(spec["flux"])
    elif normalisation_method == "flux_at_wl":
        if normalisation_wavelength is None:
            warn("No normalisation wavelength provided. Normalisation skipped!")
            return spec
        else:
            # Case where the desired wavelength is in the data array
            mask = spec["wave"] == normalisation_wavelength
            if np.sum(mask) == 1:
                val = spec["flux"][mask][0]
            # Case where desired wavelength is in the data array more than once
            elif np.sum(mask) > 1:
                raise ValueError("Duplicate wavelengths exist in the spectrum data.")
            # Case where the desired wavelength isnt in the data array.
            # Interpolate the flux from the nearest neighbours.
            else:
                lower_ind, upper_ind = find_neighbours(spec["wave"], normalisation_wavelength)
                if lower_ind is not None and upper_ind is not None: 
                    val = interpolate_linear(spec["wave"][[lower_ind, upper_ind]],
                                             spec["flux"][[lower_ind, upper_ind]],
                                             normalisation_wavelength)
                else:
                    warn("Normalisation wavelength provided is not within the wavelength range of the spectrum. "
                         "Normalisation skipped!")
                    return spec
    else:
        warn(f"Normalisation method '{normalisation_method}' is not a valid method. "
             "Must be either 'max' or 'flux_at_wl'. Normalisation skipped!")

    spec["flux"] = normalise_data(spec["flux"], val.value)
    spec["flux_err"] = normalise_data(spec["flux_err"], val.value)

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