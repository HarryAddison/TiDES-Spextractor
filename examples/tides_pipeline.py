import os
import sys

import tides_spextractor
from astropy.table import QTable
import matplotlib.pyplot as plt
from tides_spextractor.util.input_output import load_config


if __name__ == "__main__":

    #TODO Clean up this example!
    # (Currently its used for various testing during development)

    ts_config = load_config("tides_pipeline_config.yaml")

    # a = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/test-data/simple-test-data/spectrum_parameters.fits")
    # i = 0  # There are 3 SNe in the sample, use python indexing
    # sn = tides_spextractor.SN("Ia", z=a["redshift"][i], ra=a["ra"][i], dec=a["dec"][i], mwebv=a["mwebv"][i], ebv=None, phase=0, rest_phase=0, config=ts_config)
    # sn.add_spectrum(fn=f"/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/test-data/simple-test-data/mock-4most-spectra/spectrum_{i+1}.fits")
    # sn.add_spectrum(fn=f"/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/mock-4most-spectra/template_{i+1}.fits")
    
    
    # Data from SNR testing
    # sne_info = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/SNR-testing-with-host-subtraction/data/mock-spectra/spectrum_parameters.fits")
    # id = 1
    # row = sne_info[sne_info["id"]==id]
    # row.pprint_all()
    # exit()
    # sn = tides_spextractor.SN("Ia", z=row["z"], ra=row["ra"], dec=row["dec"], mwebv=row["mwebv"], ebv=row["hostebv"], phase=row["observer_phase"][0], rest_phase=row["rest_phase"][0], config=ts_config)
    # sn.add_spectrum(fn=f"/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/SNR-testing-with-host-subtraction/data/mock-spectra/l1-spectra/l1_spectrum_{id}.fits")

    

    # Data from host extraction testing
    sne_info = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/host-contamination-testing/data/mock-spectra/sn_spectrum_parameters.fits")
    # sne_info.pprint_all()
    id = 148
    row = sne_info[sne_info["id"]==id]
    row.pprint_all()
    sn = tides_spextractor.SN(sn_id=row["id"][0], sn_type="Ia", z=row["z"], ra=row["ra"], dec=row["dec"], mwebv=row["mwebv"], ebv=row["hostebv"], phase=row["observer_phase"][0], rest_phase=row["rest_phase"][0], config=ts_config)
    sn.add_spectrum(spec_id=1, fn=f"/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/host-contamination-testing/data/mock-spectra/l1-spectra/l1_spectrum_{id}.fits")




    # # CF spectra
    # import astropy.units as u

    # name = 69247744
    # visit_id = 663
    # phase = 5 * u.day

    # # name = 69531257
    # # visit_id = 14948
    # # phase = 25 * u.day

    # sne_info = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/data-CF-spectra/observedSNeAndGalaxies_joinedSNR.csv")
    # mask = (sne_info["name"] == name) & (sne_info["visit_id"] == visit_id)
    # row = sne_info[mask]
    # print(row)
    # sn = tides_spextractor.SN("Ia", z=row["redshift_estimate"][0], ra=row["ra"]*u.deg, dec=row["dec"]*u.deg, phase=phase, config=ts_config)
    # sn.add_spectrum(fn=f"/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/data-CF-spectra/justSNSpectra/spec_{row['subsurvey'].value[0]}_{row['name'].value[0]}_{row['visit_id'].value[0]}_spectrum.fits")







    for spec in sn.spectra:
        spec.preprocess()
        spec.create_model()
        spec.measure_properties()
        spec.features.pprint_all()
        print("Galaxy eigenvalues:", spec.gal_model_eigenvals)
        spec.plot()
        plt.show()
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