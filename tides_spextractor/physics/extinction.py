'''
Author: Harry Addison
Created: 17/02/2025
'''

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from extinction import fitzpatrick99, remove


def deredden(data, ebv, rv, keys=["x", "y", "y_err"], **kwargs):
    if data[keys[0]].unit in ["Angstrom", "aa", "A"]:
        unit = "aa"
    else:
        raise ValueError("Wavelength unit should be in angstroms")
    # wavelength dtype has to be float64 otherwise fitzpatrick99 errors. 
    wave = np.array(data[keys[0]].value, dtype=np.float64)
    wl_dependent_ebv = fitzpatrick99(wave, a_v=(rv * ebv.value),
                                     r_v=rv, unit=unit)

    data[keys[1]] = remove(wl_dependent_ebv, data[keys[1]])

    return data


def get_mwebv(ra, dec, frame='icrs', **kwargs):
    '''
    Query the Schlegel, Finkbeiner & Davis dust map at the given
    coords for the MW Galaxy extinction ebv.
    '''
    coord = SkyCoord(ra, dec, frame=frame)
    mwebv = float(SFDQuery()(coord)) * u.mag

    return mwebv
