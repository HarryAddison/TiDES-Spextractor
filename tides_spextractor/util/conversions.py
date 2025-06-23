'''
Author: Harry Addison
Created: 17/02/2025
'''

import astropy.units as u 


def nm_to_A(wave):
    return wave.to(u.Angstrom)


def normalise_data(data, val):
    data = data / val
    return data


def convert_config_quantities(config):

    for key, value in config.items():
        if isinstance(value, (str)):
            parts = value.split()
            if len(parts) == 2:
                try:
                    val, unit = parts
                    config[key] = float(val) * u.Unit(unit)
                except ValueError:
                    continue

    return config
