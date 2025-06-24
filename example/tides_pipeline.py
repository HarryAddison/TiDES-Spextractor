import os
import sys



import tides_spextractor
from astropy.table import QTable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/spectrum_parameters.fits")
    sn = tides_spextractor.SN("Ia", z=a["redshift"][0], ra=a["ra"][0], dec=a["dec"][0], mwebv=a["mwebv"][0], ebv=None, phase=0, rest_phase=0)
    sn.add_spectrum(fn="/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/mock-4most-spectra/spectrum_1.fits")

    keys = ["wave", "flux", "flux_err"]
    for spec in sn.spectra:

        spec.preprocess(**{"keys": keys})
        spec.create_model()
        spec.plot()
        spec.measure_properties(kwargs={"n_cpu":48})
        spec.features.pprint_all()
    exit()






























import numpy as np
import astropy.units as u

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    pass

from tides_spextractor import SN, load_config


# This is main function that will be run
def analyse_spectrum(sn_info, spec_info, **config):

    # Initialise the SN class
    sn = SN(**sn_info)

    # Initialise the spectrum
    sn.add_spectrum(**spec_info, **config)


    sn.spectra[0].data["flux"][0] = np.nan * u.erg / (u.Angstrom * u.s * u.cm * u.cm)
    sn.spectra[0].data["flux"][1] = 0 * u.erg / (u.Angstrom * u.s * u.cm * u.cm)
    sn.spectra[0].data["flux"][2] = -10 * u.erg / (u.Angstrom * u.s * u.cm * u.cm)
    sn.spectra[0].data["flux"][3] = 99999 * u.erg / (u.Angstrom * u.s * u.cm * u.cm)
    sn.spectra[0].data["flux"][4] = np.ma.masked
    
    # Perform preprocessing
    sn.spectra[0].preprocess(**config)
    # Produce GPR model
    
    # Analyse the features

    # Make plots


if __name__ == "__main__":

    # Load the config from the yaml file
    config = load_config()

    # Run a database query for recently updated SNe.
    # These updated SNe will be analysed using tides_spextractor


    # For each of the SNe run spectral analysis ("analyse_spectrum"). Run it as its own process.
    # dask fire_and_forget might be useful for this.

    sn_info = {"sn_type": "Ia",
               "rest_phase": 0}
    spec_info = {"fn": "/vol/ph/astro_data/haddison/TiDES-spectral-analysis/ozdes-Ia-testing-2025/data/spectra/l1_spectrum_1.fits",
                 "file_format": "fits",
                 "wl_col": "WAVE",
                 "flux_col": "FLUX",
                 "flux_err_col": "ERR_FLUX"}

    analyse_spectrum(sn_info, spec_info, **config)