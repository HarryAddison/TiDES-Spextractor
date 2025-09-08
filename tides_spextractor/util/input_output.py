'''
Author: Harry Addison
Created: 17/02/2025

Functions for input and output operations.
'''

from tides_spextractor.util.conversions import *
from tides_spextractor.physics.doppler import deredshift

import argparse
import astropy.units as u
import importlib
import textwrap
import yaml

from astropy.table import QTable


def parser():
    '''
    Function that allows the user to parse a initfile path via the
    command line.
    '''

    help_msg = "A help message and description of the code"  #TODO

    parser = argparse.ArgumentParser(prog="TiDES Spectral Analysis",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(help_msg))

    parser.add_argument("-i",
                        "--initfile",
                        default=None,
                        help=("yaml file containing input frames "
                              "and other procedure arguments"))
    return parser


def load_config(initfile=None):
    '''
    Obtain the config parameters from the given yaml file. If no file path provided
    try to see if a yaml file was parsed via the command line.
    '''

    if initfile == None:
        arg_parser = parser()
        args = arg_parser.parse_args()
        initfile = args.initfile
        print(f"\nInput parameter file: {initfile}\n\n")

        # If no parameter file was provided print help message
        if initfile is None:
            arg_parser.print_help()
            exit()

    # Open the initfile and read in the parameters
    with open(str(initfile), "r") as file:
        config = yaml.safe_load(file)

    config = convert_config_quantities(config)
    return config


def read_spec(fn, file_format, keys=["x", "y", "y_err"], **kwargs):
    if isinstance(fn, str) and isinstance(file_format, str):
        spec = QTable.read(fn, format=file_format)
        spec.rename_columns([keys[0], keys[1], keys[2]], ["wave", "flux", "flux_err"])
        return spec
    elif isinstance(fn, str) is False:
        raise TypeError("The filename provided is not a string.")
    elif isinstance(file_format, str) is False:
        raise TypeError("The filename provided is not a string.")


def obtain_spec(fn, file_format, **kwargs):
    spec = read_spec(fn, file_format, **kwargs)

    if spec["wave"].unit == u.nm:
        spec["wave"] = nm_to_A(spec["wave"])
    elif spec["wave"].unit != u.Angstrom:
        raise u.core.UnitTypeError("Incompatible wavelength units. Spectrum wavelengths must be in nm or Angstrom.")

    flux_unit = u.erg / (u.Angstrom * u.s * u.cm * u.cm)
    if spec["flux"].unit != flux_unit:
        raise u.core.UnitTypeError(f"Incompatiple flux units. Spectrum flux must be in {flux_unit}.")
    if spec["flux_err"].unit != flux_unit:
        raise u.core.UnitTypeError(f"Incompatiple flux error units. Spectrum flux error must be in {flux_unit}.")

    return spec


def load_spectral_feature_definitions(sn_type, features_dir=None, **kwargs):
    '''
    Load in the spectral features relevant to the given SN type.

    Inputs:
    > "sn_type" = type of SN as a sting. e.g "Ia", "II", "Ib" ...

    > "features_path" = string of the path to the directory where the
                        spectral features files are located.
                        default = "Spectral_Features/"
    
    Outputs:
    > "features" = table containing the spectral feature information
                   including: rest wavelength, lower bound wavelength
                   range, and upper bound wavelength range.
    '''

    if features_dir is None:
        features_dir = importlib.resources.files("tides_spextractor.data").joinpath("spectral-regions")

    fn = "%s/SN_%s_features.ecsv"%(features_dir, sn_type)

    features = QTable.read(fn, format="ascii.ecsv")

    return features


def load_telluric_regions(file_path=None, z=None, **kwargs):
    '''
    Load in the telluric absorption regions. Apply a de-redshift to
    convert the wavelength regions into "rest" frame (optional).

    Input:
    > "file_path" = path to the directory where the telluric absorption
                    region wavelength ranges are saved.
    > "z" = Redshift to use to convert wavelength regions to "rest"
            frame. If "z" = None then the wavelengths are not
            de-redshifted into "rest" frame.

    Output:
    > "regions" = table containing the telluric absorption feature
                  blue-ward and red-ward wavelength bounds.
    '''
    if file_path is None:
        file_path = importlib.resources.files("tides_spextractor.data").joinpath("spectral-regions/tellurics.ecsv")
    #TODO add in a check for the file path being ecsv.
    regions = QTable.read(file_path, format="ascii.ecsv")

    # TODO add check/conversion for wavelength units in the tellurics table.

    # De-redshift wavelengths to "rest" frame.
    if z is not None:
        regions["lower_wl"] = deredshift(regions["lower_wl"], z)
        regions["upper_wl"] = deredshift(regions["upper_wl"], z)

    return regions
