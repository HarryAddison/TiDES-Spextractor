'''
Author: Harry Addison
Created: 17/02/2025
'''

from tides_spextractor.maths.gpr import *
from tides_spextractor.maths.interpolation import downsample_spec_data, bin_spec_data
from tides_spextractor.physics.extinction import get_mwebv
from tides_spextractor.physics.line_velocity import calc_vel
from tides_spextractor.physics.pseudo_equivalent_width import calc_pew
from tides_spextractor.util.conversions import *
from tides_spextractor.util.input_output import *
from tides_spextractor.util.plotting import *
from tides_spextractor.util.preprocessing import *
from tides_spextractor.util.spectral_tools import *
from host_removal import HostGalaxyRemoval


class SN:

    def __init__(self, sn_type, z=None, ra=None, dec=None, mwebv=None, ebv=None,
                 phase=None, rest_phase=None, config=None):

        self.sn_type = sn_type
        self.z = z
        self.ra = ra
        self.dec = dec
        self.mwebv = mwebv
        self.ebv = ebv
        self.phase = phase
        self.rest_phase = rest_phase
        self.config = config
        self.spectra = []

        # TODO Checks to make sure the provided values above are suitable.

        if self.mwebv is None and self.ra is not None and self.dec is not None:
            self.mwebv = get_mwebv(self.ra, self.dec)

        if self.rest_phase is None:
            if self.phase is not None:
                self.rest_phase = self.phase / (1 + self.z)


    def add_spectrum(self, fn, **kwargs):
        '''
        '''
        self.spectra.append(Spectrum(self, fn, **kwargs))



class Spectrum:

    def __init__(self, sn_instance, fn, file_format="fits", min_wavelength=None,
                 max_wavelength=None, **kwargs):

        self.sn_instance = sn_instance
        self.data = obtain_spec(fn, file_format, **self.config)

        if min_wavelength is None:
            self.min_wl = min(self.data[self.config["keys"][0]])
        else:
            self.min_wl = min_wavelength
        if max_wavelength is None:
            self.max_wl = max(self.data[self.config["keys"][0]])
        else:
            self.max_wl = max_wavelength
        if self.min_wl.unit == u.nm:
            self.min_wl = nm_to_A(self.min_wl)
        if self.max_wl.unit == u.nm:
            self.max_wl = nm_to_A(self.max_wl)
        if self.min_wl.unit not in [u.Angstrom, u.nm] or self.max_wl.unit not in [u.Angstrom, u.nm]:
            raise u.core.UnitTypeError("Incompatible wavelength units. Wavelength range must be in nm or Angstrom.")

        self.min_rest_wl = None
        self.max_rest_wl = None
        self.model_data = None
        self.features = None
        self.gpr_model = None
        self.gpr_kernel = None
        self.ds_data = None
        self.gal_model_eigenvals = None
        

        #TODO Add checks for the right data structure/keys


    def __getattr__(self, attr):
        return getattr(self.sn_instance, attr)


    def preprocess(self, **kwargs):
        '''
        Preprocess the spectrum. This includes the following:
            - Removing zeros, nan, masked values
            - Sorting into ascending wavelength
            - Remove data outside of desired observed wavelength range
            - Removing tellurics
            - Remove extinction
            - Convert wavelengths to rest-frame (redshift correction)
            - Remove outliers
            - Normalisation
        '''
        # Overwrite/combine the config kwargs with those defined in the function call.
        kwargs = {**{"keys": ["x", "y", "y_err"]}, **self.config, **kwargs}

        self.data = clean_spectrum(self.data, **kwargs)
        self.data.sort(keys=kwargs["keys"])
        self.data = prune_spectrum(self.data, self.min_wl, self.max_wl, **kwargs)
        # self.data = remove_tellurics(self.data, **kwargs)
        self.data = deredshift_spectrum(self.data, self.z, **kwargs)
        self.data = deredden_spectrum(self.data, self.mwebv, self.ebv, **kwargs)
        self.min_rest_wl, self.max_rest_wl = min(self.data[kwargs["keys"][0]]), max(self.data[kwargs["keys"][0]])
        # self.data = remove_outliers(self.data)
        if kwargs["preprocess_binning"] == True:
            self.data = bin_spec_data(self.data, kwargs["preprocess_binning_width"], **kwargs)
        self.data = normalise_spectrum(self.data, **kwargs)


    def create_model(self, **kwargs):
        # Overwrite/combine the config kwargs with those defined in the function call.
        kwargs = {**self.config, **kwargs}

        self.ds_data = self._check_spec_size(self.data, **kwargs)
        self.gpr_model, self.gpr_kernel = make_model(self.ds_data, **kwargs)
        self.model_data = model_values(self.gpr_model, self.gpr_kernel, self.ds_data, **kwargs)

        if kwargs["host_gal_removal"]:
            self._remove_host_galaxy(**kwargs)


    def measure_properties(self, **kwargs):
        '''
        Measure the properties of the spectral properties.
        '''
        # Overwrite/combine the config kwargs with those defined in the function call.
        kwargs = {**self.config, **kwargs}

        if self.model_data is None:
            self.create_model()
        if self.features is None:
            self._setup_spectral_features(**kwargs)
        self._measure_properties(**kwargs)


    def plot(self, **kwargs):
        # Overwrite/combine the config kwargs with those defined in the function call.
        kwargs = {**self.config, **kwargs}

        plt.figure(figsize=(16,9))

        if kwargs["plot_tellurics"]:
            plot_telluric_regions([self.min_wl, self.max_wl], self.z, **kwargs)

        plot_spec_with_err(self.data, alphas=[0.4, 0.3], color="k",
                           label="Processed Spectrum", z_orders=[2, 1], **kwargs)
        plot_spec_with_err(self.ds_data, alphas=[0.2, 0.1], color="blue",
                           label="Downsampled Spectrum", z_orders=[4, 3], **kwargs)
        plot_spec_with_err(self.model_data, plot_type="line", alphas=[1, 0.3], color="red",
                           label="Model Spectrum", z_orders=[6, 5], **kwargs)

        plot_features(self.features, "k", 7, **kwargs)

        plt.legend()
        plt.xlabel(r"$\rm{Rest Frame Wavelength}~(\AA)$")
        plt.ylabel("Normalised flux")
        plt.xlim((self.min_rest_wl.value - 100), (self.max_rest_wl.value + 100))
        plt.ylim(0, 1.1)


    def _setup_spectral_features(self, **kwargs):
        self._load_features(**kwargs)
        self._identify_features_coincident_with_tellurics(**kwargs)
        self.features = identify_present_features(self.features, self.model_data[kwargs["keys"][0]], **kwargs)
        self.features = locate_spectral_features(self.model_data, self.features, **kwargs)


    def _load_features(self, **kwargs):
        self.features = load_spectral_feature_definitions(self.sn_type, **kwargs)
        self.features["present_flag"] = True
        self.features["measure_flag"] = True
        self.features["continuum"] = np.full(len(self.features), None, dtype=object)
        self.features["vel"] = np.nan * u.km / u.s
        self.features["vel_err"] = np.nan * u.km / u.s
        self.features["pew"] = np.nan * u.angstrom
        self.features["pew_err"] = np.nan * u.angstrom
        self.features["comment"] = np.full(len(self.features), None, dtype="U100")


    def _identify_features_coincident_with_tellurics(self, **kwargs):
        telluric_regions = load_telluric_regions(z=self.z, **kwargs)
        self.features = identify_features_coincident_with_tellurics(self.features, telluric_regions, **kwargs)


    def _check_spec_size(self, data, points_limit=4000, **kwargs):
        if len(data) > points_limit:
            ds_factor = int(len(data) / points_limit)
            return downsample_spec_data(data, ds_factor, **kwargs)
        else:
            return data


    def _measure_properties(self, keys=["x", "y"], **kwargs):
        for i, feature in enumerate(self.features):
            if feature["measure_flag"]:
                if feature["continuum"]:
                    spec_feature_data = get_feature_data(self.model_data, feature["continuum"], keys=keys, **kwargs)
                    spec_feature_data_continuum_normalised = normalise_by_continuum(spec_feature_data.copy(),
                                                                                    feature["continuum"], keys=keys)

                    vel, vel_err = calc_vel(spec_feature_data_continuum_normalised, feature["rest_wl"], self.gpr_model, keys=keys, **kwargs)
                    pew, pew_err = calc_pew(spec_feature_data, feature["continuum"], self.gpr_model, keys=keys, **kwargs)

                    self.features["vel"][i] = vel
                    self.features["vel_err"][i] = vel_err
                    self.features["pew"][i] = pew
                    self.features["pew_err"][i] = pew_err


    def _remove_host_galaxy(self, host_gal_removal_plot=False, **kwargs):
        hgr = HostGalaxyRemoval(self.model_data, self.rest_phase.value, **kwargs)
        hgr.fit_spectrum()
        hgr.remove_galaxy_contamination()
        if host_gal_removal_plot:
            hgr.plot_fit(show=True)

        self.model_data = hgr.sn_spec_no_host
        self.gal_model_eigenvals = hgr.gal_eigenvals
