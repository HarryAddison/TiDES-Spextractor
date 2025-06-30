'''
Author: Harry Addison
Created: 17/02/2025
'''

import astropy.units as u
import numpy as np
import torch

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


def convert_np_to_tensor(array, dtype=torch.float32):

    # If the byte order is not native, swap bytes and set new byte order to native
    # '=' means native, '|' means not applicable (e.g. for bytes)
    if array.dtype.byteorder not in ('=', '|'):
        array = array.byteswap().newbyteorder()
    array = np.ascontiguousarray(array)
    tensor = torch.tensor(array, dtype=dtype)
    return tensor
