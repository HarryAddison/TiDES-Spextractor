'''
Author: Harry Addison
Created: 17/02/2025
'''

from tides_spextractor.maths.gpr import *
from tides_spextractor.maths.interpolation import downsample_spec_data
from tides_spextractor.physics.line_velocity import calc_vel
from tides_spextractor.physics.pseudo_equivalent_width import calc_pew
from tides_spextractor.util.conversions import *
from tides_spextractor.util.input_output import *
from tides_spextractor.util.preprocessing import *
from tides_spextractor.util.spectral_tools import *



class SN:

    def __init__(self, sn_type, z=None, ra=None, dec=None, mwebv=None, ebv=None,
                 phase=None, rest_phase=None, **kwargs):

        self.sn_type = sn_type
        self.z = z
        self.ra = ra
        self.dec = dec
        self.mwebv = mwebv
        self.ebv = ebv
        self.phase = phase
        self.rest_phase = rest_phase

        self.spectra = []

        # TODO Checks to make sure the provided values above are suitable.


        if mwebv is None and self.ra is not None and self.dec is not None:
            raise NotImplementedError()
            self.mwebv = get_mwebv()  #TODO

        if rest_phase is None:
            raise NotImplementedError("Conversion of the phase to the rest phase "
                                      "is not yet implemented. Please provide the rest phase")


    def add_spectrum(self, fn, **kwargs):
        '''
        '''
        self.spectra.append(Spectrum(self, fn, **kwargs))



class Spectrum:

    def __init__(self, sn_instance, fn, file_format="fits", min_wavelength=None,
                 max_wavelength=None, **kwargs):
        self.sn_instance = sn_instance

        self.data = obtain_spec(fn, file_format, **kwargs)
        if min_wavelength is None:
            self.min_wl = min(self.data["wave"])
        else:
            self.min_wl = min_wavelength
        if max_wavelength is None:
            self.max_wl = max(self.data["wave"])
        else:
            self.max_wl = max_wavelength
        if self.min_wl.unit == u.nm:
            self.min_wl = nm_to_A(self.min_wl)
        if self.max_wl.unit == u.nm:
            self.max_wl = nm_to_A(self.max_wl)
        if self.min_wl.unit not in [u.Angstrom, u.nm] or self.max_wl.unit not in [u.Angstrom, u.nm]:
            raise u.core.UnitTypeError("Incompatible wavelength units. Wavelength range must be in nm or Angstrom.")
        self.model_data = None
        self.features = None
        self.gpr_model = None
        self.gpr_kernel = None

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

        self.data = clean_spectrum(self.data, **kwargs)
        self.data.sort(keys="wave")
        self.data = prune_spectrum(self.data, self.min_wl, self.max_wl)
        self.data = remove_tellurics(self.data, **kwargs)
        self.data = deredden_spectrum(self.data, self.mwebv, self.ebv)
        self.data = deredshift_spectrum(self.data, self.z, **kwargs)
        # self.data = remove_outliers(self.data)
        self.data = normalise_spectrum(self.data, **kwargs)


    def create_model(self):
        data = self._check_spec_size(self.data)
        self.gpr_model, self.gpr_kernel = make_model(data)
        self.model_data = model_values(self.gpr_model, self.gpr_kernel, self.data["wave"])


    def measure_properties(self, **kwargs):
        '''
        Measure the properties of the spectral properties.
        '''

        if self.model_data is None:
            self.create_model()

        if self.features is None:
            self._setup_spectral_features()
        self.features.pprint_all()
        self._measure_properties()


    def plot(self):
        # TODO Separate this code into different functions/files.
        import matplotlib.pyplot as plt
        plt.figure()

        plt.plot(self.data["wave"].value, self.data["flux"].value,
                 color="k", zorder=2, label="Processed Spectrum")
        plt.fill_between(self.data["wave"].value,
                         self.data["flux"].value - self.data["flux_err"].value,
                         self.data["flux"].value + self.data["flux_err"].value,
                         alpha=0.3, color="k", zorder=1,
                         label="Processed Spectrum Error")

        plt.plot(self.model_data["wave"].value, self.model_data["flux"].value,
                 color="red", zorder=4, label="Model Spectrum")
        plt.fill_between(self.model_data["wave"].value,
                         self.model_data["flux"].value - self.model_data["flux_err"].value,
                         self.model_data["flux"].value + self.model_data["flux_err"].value,
                         alpha=0.3, color="red", zorder=3,
                         label="Model Spectrum Error")

        # for feature in self.features:
        #     plt.plot(feature["continuum"]["wave"].value, feature["continuum"]["flux"].value)
        
        plt.show()
        

        # call to plot pre-processed spectrum
        # call to plot model
        # call to plot the features
        # call to plot the telluric regions
        # call to save plot


    def _setup_spectral_features(self, **kwargs):
        self._load_features(**kwargs)
        self._identify_features_coincident_with_tellurics(**kwargs)
        self.features = identify_present_features(self.features, self.model_data["wave"])
        self.features = locate_spectral_features(self.model_data, self.features, **kwargs)


    def _load_features(self, **kwargs):
        self.features = load_spectral_feature_definitions(self.sn_type, **kwargs)[:2]  #TODO Remove index
        self.features["present_flag"] = True
        self.features["measure_flag"] = True
        self.features["continuum"] = np.full(len(self.features), None, dtype=object)
        self.features["vel"] = None
        self.features["vel_err"] = None
        self.features["pew"] = None
        self.features["pew_err"] = None
        self.features["comment"] = np.full(len(self.features), None, dtype="U100")


    def _identify_features_coincident_with_tellurics(self, **kwargs):
        telluric_regions = load_telluric_regions(z=self.z, **kwargs)
        self.features = identify_features_coincident_with_tellurics(self.features, telluric_regions)


    def _check_spec_size(self, data, limit=4000):
        if len(data) > limit:
            ds_factor = int(len(data) / limit)
            return downsample_spec_data(data, ds_factor, data.keys())
        else:
            return data


    def _measure_properties(self):
        for i, feature in enumerate(self.features):
            if feature["measure_flag"]:
                if feature["continuum"]:
                    spec_feature_data = get_continuum_subtracted_feature_data(self.model_data, feature["continuum"], feature["continuum"].keys())
                    vel, vel_err = calc_vel(spec_feature_data, feature["rest_wl"],
                                                            self.gpr_model, self.gpr_kernel,
                                                            spec_feature_data.keys())
                    pew, pew_err = calc_pew(spec_feature_data, spec_feature_data.keys())
                    self.features["vel"][i] = vel
                    self.features["vel_err"][i] = vel_err
                    self.features["pew"][i] = pew
                    self.features["pew_err"][i] = pew_err
                    # TODO Add units to the measurements
