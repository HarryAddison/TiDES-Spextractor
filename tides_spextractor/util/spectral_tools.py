'''
Author: Harry Addison
Created: 15/05/2025
'''

from tides_spextractor.maths.interpolation import interpolate_linear
from tides_spextractor.util.input_output import load_spectral_feature_definitions, load_telluric_regions
from tides_spextractor.util.memory_size import estimate_table_size
from astropy.table import QTable
from multiprocessing import  cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import numpy as np
import psutil


def identify_present_features(features, spec_wls):
    within_wl_range_mask = ((features["lo_range_lo"]>= spec_wls.min()) &
                            (features["up_range_up"]<= spec_wls.max()))
    features["measure_flag"][~within_wl_range_mask] = False
    features["comment"][~within_wl_range_mask] = "Feature outside of spectrum wavelengths"
    return features


def identify_features_coincident_with_tellurics(features, telluric_regions):
    for telluric_region in telluric_regions:
        coincide_telluric_mask = ((telluric_region["lower_wl"] <= features["up_range_up"]) &
                                         (telluric_region["upper_wl"] >= features["lo_range_lo"]))
        features["comment"][coincide_telluric_mask] = "Feature coincides with a telluric region"
    return features


def evaluate_continuum(spec):

    continuum_flux = interpolate_linear([spec["wave"][0], spec["wave"][-1]],
                                        [spec["flux"][0], spec["flux"][-1]], spec["wave"])
    continuum = QTable({"wave": spec["wave"], "flux": continuum_flux})

    residuals = continuum["flux"] - spec["flux"]

    del continuum
    del spec

    if min(residuals) >= 0:
        return True
    else:
        return False


def calc_best_continuum(specs_to_check, n_cpu=None, **kwargs):
    # table_sizes = []
    # for spec in specs_to_check:
    #     table_sizes.append(estimate_table_size(spec))
    # available_ram = psutil.virtual_memory().available
    # print(available_ram, available_ram / 1024**3)
    # print(np.mean(table_sizes) / 1024**3, max(table_sizes) / 1024**3, min(table_sizes) / 1024**3)
    # max_cpu = int(available_ram / (2 * np.mean(table_sizes)))
    if n_cpu is None:
        n_cpu = 40
    # elif max_cpu < n_cpu:
    #     n_cpu = max_cpu

    print(f"Starting continuum finding with {n_cpu} CPUs")

    # # Method 1: Process the continua in decending width to find first valid one.
    # with ProcessPoolExecutor(max_workers=n_cpu) as executor:
    #     futures = [executor.submit(evaluate_continuum, spec) for spec in specs_to_check]
    #     futures_set = set(futures)

    #     while futures_set:
    #         done_now, _ = wait(futures_set, return_when=FIRST_COMPLETED)

    #         for future in done_now:
    #             result = future.result()
    #             if result != None:
    #                 print("RESULT")
    #                 for f in futures_set:
    #                     f.cancel()
    #                 return result
    #         futures_set -= done_now  # remove finished futures
    # return None

    # Method 2: Using dask
    cluster = LocalCluster(n_workers=n_cpu, memory_limit="auto")
    client = Client(cluster)
    print(f"Dask cluster started with {len(client.nthreads())} workers.")

    # Create delayed tasks
    tasks = [delayed(evaluate_continuum)(spec) for spec in specs_to_check]

    # Compute all tasks in parallel
    results = compute(*tasks)

    # Find the first valid result
    for i, result in enumerate(results):
        if result is True:
            client.shutdown()
            print("RESULT\n", result)
            spec = specs_to_check[i]

            continuum_flux = interpolate_linear([spec["wave"][0], spec["wave"][-1]],
                                                [spec["flux"][0], spec["flux"][-1]],
                                                spec["wave"])
            continuum = QTable({"wave": spec["wave"], "flux": continuum_flux})
            print(continuum)
            return continuum

    client.shutdown()
    return None




def get_continuum(spec, feature, **kwargs):
    '''
    Locate the maxima in the feature's lower bound. Continuum must be fitted
    to this point or a point at a higher wavelength.
    '''

    lower_region_mask = ((spec["wave"] < feature["lo_range_up"]) &
                         (spec["wave"] > feature["lo_range_lo"]))
    upper_region_mask = ((spec["wave"] < feature["up_range_up"]) &
                         (spec["wave"] > feature["up_range_lo"]))
    lower_region_mask_inds = np.where(lower_region_mask)[0]
    upper_region_mask_inds = np.where(upper_region_mask)[0]

    lower_region_maxima_ind = lower_region_mask_inds[spec["flux"][lower_region_mask].argmax()]
    upper_region_maxima_ind = upper_region_mask_inds[spec["flux"][upper_region_mask].argmax()]

    lower_region_search_inds = np.arange(lower_region_maxima_ind, (lower_region_mask_inds[-1] + 1), 1)
    upper_region_search_inds = np.arange(upper_region_mask_inds[0], (upper_region_maxima_ind + 1), 1)
    print("starting generator conversion")
    specs_to_check = list((spec[lo_ind : up_ind]) for lo_ind in lower_region_search_inds
                          for up_ind in upper_region_search_inds)
    print("Sorting")
    specs_to_check = sorted(specs_to_check, key=lambda tbl: tbl['wave'].max() - tbl['wave'].min(), reverse=True)
    print("starting calculation of best continuum")
    continuum = calc_best_continuum(specs_to_check, **kwargs)
    print("continuum found or not found")
    return continuum


def locate_spectral_features(spec, features, **kwargs):
    for i, feature in enumerate(features):
        print(i, feature)
        if feature["measure_flag"]:
            continuum = get_continuum(spec, feature)
            if continuum:
                features[i]["continuum"] = continuum
            else:
                features[i]["comment"] = "A valid continuum could not be found"
                
    return features


def get_continuum_subtracted_feature_data(data, continuum_data, keys=["x", "y"]):

    feature_wl_mask = ((data[keys[0]] >= min(continuum_data[keys[0]])) &
                       (data[keys[0]] <= max(continuum_data[keys[0]])))
    feature_data = data[feature_wl_mask]
    feature_data[keys[1]] -= continuum_data[keys[1]]
    return feature_data
