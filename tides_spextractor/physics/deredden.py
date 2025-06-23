'''
Author: Harry Addison
Created: 17/02/2025
'''

import numpy as np
from extinction import fitzpatrick99, remove


def deredden(data, ebv, rv):
    if data["wave"].unit in ["Angstrom", "aa", "A"]:
        unit = "aa"
    else:
        raise ValueError("Wavelength unit should be in angstroms")
    # wavelength dtype has to be float64 otherwise fitzpatrick99 errors. 
    wave = np.array(data["wave"].value, dtype=np.float64)
    wl_dependent_ebv = fitzpatrick99(wave, a_v=(rv * ebv.value),
                                     r_v=rv, unit=unit)

    data["flux"] = remove(wl_dependent_ebv, data["flux"])

    return data
