import matplotlib.pyplot as plt
from tides_spextractor.util.input_output import load_telluric_regions


def plot_spec_with_err(spec, keys=["x", "y", "z"], plot_type="scatter",
                       alphas=[0.5, 0.25], color="k", label="Spectrum",
                       marker_area=9, z_orders=[2,1],  **kwargs):
    if plot_type == "scatter":
        plt.scatter(spec[keys[0]].value, spec[keys[1]].value, alpha=alphas[0],
                    color=color, label=label, s=marker_area, zorder=z_orders[0], **kwargs)        
    elif plot_type == "line":
        plt.scatter(spec[keys[0]].value, spec[keys[1]].value, alpha=alphas[0],
                    color=color, label=label, zorder=z_orders[0], **kwargs)      

    plt.fill_between(spec[keys[0]].value,
                     spec[keys[1]].value - spec[keys[2]].value,
                     spec[keys[1]].value + spec[keys[2]].value,
                     alpha=alphas[1], color=color, label=f"{label} Error",
                     zorder=z_orders[1], **kwargs)


def plot_telluric_regions(wl_range, z, **kwargs):

    tellurics = load_telluric_regions(z=z)

    for telluric in tellurics:
        if telluric["lower_wl"] >= wl_range[0] and telluric["upper_wl"] <= wl_range[1]:
            plt.fill_between([telluric["lower_wl"].value, telluric["upper_wl"].value],
                             1, 0, color="k", alpha=0.4, zorder=0)

            plt.text(telluric["lower_wl"], 1.02, telluric["feature"],
                     rotation=90, fontsize=9)


def plot_features(features, color, z_order, keys=["x", "y", "z"], **kwargs):
    for feature in features:
        try:
            plt.plot(feature["continuum"][keys[0]].value,
                     feature["continuum"][keys[1]].value,
                     c=color, zorder=z_order, label="Continuum")

            mid_ind = int(len(feature["continuum"]) / 2)
            x = feature["continuum"][keys[0]][mid_ind]
            y = feature["continuum"][keys[1]][mid_ind] + 0.025
            plt.text(x, y, feature["feature"], rotation=90, fontsize=7, zorder=z_order)
        except:
            pass
