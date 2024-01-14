import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define import directories (d = 0.1)
file_paths = {
    "h0_t1e" : "/Users/richie.lee/Desktop/ps_ground_truth_H0_prior_H0.xlsx",
    "h1_t1e" : "/Users/richie.lee/Desktop/ps_ground_truth_H0_prior_H1.xlsx",
    "h0_p" : "/Users/richie.lee/Desktop/ps_ground_truth_H1_prior_H0.xlsx",
    "h1_p" : "/Users/richie.lee/Desktop/ps_ground_truth_H1_prior_H1.xlsx"
    }

# # Define import directories (d = 0.01)
# file_paths = {
#     "h0_t1e" : "/Users/richie.lee/Desktop/ps_h0_t1e.xlsx",
#     "h1_t1e" : "/Users/richie.lee/Desktop/ps_h1_t1e.xlsx",
#     "h0_p" : "/Users/richie.lee/Desktop/ps_h0_p.xlsx",
#     "h1_p" : "/Users/richie.lee/Desktop/ps_h1_p.xlsx"
#     }

# # Define import directories (real)
# file_paths = {
#     "h0_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/visualisation code/real_ps_ground_truth_H0_prior_H0.xlsx",
#     "h1_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/visualisation code/real_ps_ground_truth_H0_prior_H1.xlsx",
#     "h0_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/visualisation code/real_ps_ground_truth_H1_prior_H0.xlsx",
#     "h1_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/visualisation code/real_ps_ground_truth_H1_prior_H1.xlsx"
#     }


# # Define import directories (d = 0.01) - priors corrected
# file_paths = {
#     "h0_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/tuesday_ps_ground_truth_H0_prior_H0.xlsx",
#     "h1_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/tuesday_ps_ground_truth_H0_prior_H1.xlsx",
#     "h0_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/tuesday_ps_ground_truth_H1_prior_H0.xlsx",
#     "h1_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/tuesday_ps_ground_truth_H1_prior_H1.xlsx"
#     }

# # Define import directories (d = 0.01)
# file_paths = {
#     "h0_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/prior_sensitivity_d_0_01_H0_type1_error.xlsx",
#     "h1_t1e" : "/Users/richie.lee/Desktop/results/prior_sensitivity/prior_sensitivity_d_0_01_H1_type1_error.xlsx",
#     "h0_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/prior_sensitivity_d_0_01_H0_power.xlsx",
#     "h1_p" : "/Users/richie.lee/Desktop/results/prior_sensitivity/prior_sensitivity_d_0_01_H1_power.xlsx"
#     }

# Import sheets (and preprocess for correct format)
data = dict()
for file, path in file_paths.items():
    
    # Sample sizes only relevant for Power Curves
    data_i = {
        "mean" : pd.read_excel(path, engine='openpyxl', sheet_name = "prior_means"),
        "variance" : pd.read_excel(path, engine='openpyxl', sheet_name = "prior_variances"),
        "rejections" : pd.read_excel(path, engine='openpyxl', sheet_name = "rejections"),
        "sample" : pd.read_excel(path, engine='openpyxl', sheet_name = "sample")
    } if file in ["h0_p", "h1_p"] else {
        "mean" : pd.read_excel(path, engine='openpyxl', sheet_name = "prior_means"),
        "variance" : pd.read_excel(path, engine='openpyxl', sheet_name = "prior_variances"),
        "rejections" : pd.read_excel(path, engine='openpyxl', sheet_name = "rejections")
    }
    
    # preprocessing
    for key, df in data_i.items():
        df.drop(columns=['Unnamed: 0'], inplace = True) # drop index column
        data_i[key] = df.to_numpy() # df -> np array
    
    # Add manual (maximally) poor value in there as quick and dirty solution to 0-100 scaling in heatmap
    if file == "h1_p":
        data_i["rejections"][0, 0] = 0
    elif file == "h0_t1e":
        data_i["rejections"][0, 0] = 100
        
    data[file] = data_i

# 3D plots
def plot_result_surface(x_nodes, y_nodes, z_values, color_map, title):
    # Rescale s.t. we get uniform comparisons -> Normalize z_values to a 0-1 range
    rescale_to_0_100 = False
    if rescale_to_0_100:
        z_min, z_max = np.nanmin(z_values), np.nanmax(z_values) # Due to Nan-Censoring -> Use np.nanmin and np.nanmax instead of z_values.nanmin() and z_values.nanmax()
        z_values_normalized = (z_values - z_min) / (z_max - z_min)

        # Scale to 0-100 range
        z_values = z_values_normalized * 100

    # color: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    z_values = np.reshape(z_values, x_nodes.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(x_nodes, y_nodes, z_values, levels=20, cmap=color_map)
    plt.colorbar()
    plt.contour(x_nodes, y_nodes, z_values, levels=20, cmap=color_map, linewidths=0.5)
    plt.xlabel(r'Prior mean')
    plt.ylabel('Prior variance')
    plt.axvline(0.01, color = "black")
    plt.axhline(2, color = "black", label = "true_distribution")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(title)
    plt.show()

def average_prior_effect(power, type1_error, weight_type1_error):
    return (1 - weight_type1_error) * power + (weight_type1_error) * (100 - type1_error)

def penalise_samples(power, sample, min_power):
    """
    - disqualifies all empirical power <= min_power (0, 100), by setting to None
    - penalises observed power with corresponding avg_sample size (higher sample = worse)
    """
    return np.array([n if power >= min_power else np.nan for power, n in zip(power.flatten(), sample.flatten())])

def valid_power(power, min_power):
    """
    - disqualifies all empirical power <= min_power (0, 100), by setting to None
    """
    return np.array([power if power >= min_power else np.nan for power in power.flatten()])

def valid_type1_error(type1_error, max_type1_error):
    """
    - disqualifies all empirical type_1_errors >= max_type1_error (alpha/k) by setting to None
    """
    return np.array([t1e if t1e <= max_type1_error else np.nan for t1e in type1_error.flatten()])


def valid_power_and_type1_error(power, type1_error, min_power, max_type1_error):
    """
    - "Valid" refers to with a proper type-I error control
    """
    filtered_power = valid_power(power, min_power)
    filtered_t1e = valid_type1_error(power, max_type1_error)
    
    # Formula interpretation: x = power IF p > min_p && t1e < max_t1e ELSE None
    filtered_jointly = filtered_power * (filtered_t1e / filtered_t1e) # fraction to obtain indicator for t1e-validity
    
    return filtered_jointly



POWER, ALPHA = 80, 5

mean_nodes, variance_nodes = data["h0_p"]["mean"], data["h0_p"]["variance"]


plot_result_surface(mean_nodes, variance_nodes, data['h0_t1e']["rejections"], color_map = "Reds", title = "H0 (mis)specified: Type-I error")
plot_result_surface(mean_nodes, variance_nodes, data['h0_p']["rejections"], color_map = "Greens", title = "H0 (mis)specified: Power")
plot_result_surface(mean_nodes, variance_nodes, data['h1_t1e']["rejections"], color_map = "Reds", title = "H1 (mis)specified: Type-I error")
plot_result_surface(mean_nodes, variance_nodes, data['h1_p']["rejections"], color_map = "Greens", title = "H1 (mis)specified: Power")



# Plot valid values
h0_t1e = valid_type1_error(data["h0_t1e"]["rejections"], ALPHA)
h1_t1e = valid_type1_error(data["h1_t1e"]["rejections"], ALPHA)
h0_p = valid_power(data["h0_p"]["rejections"], POWER)
h1_p = valid_power(data["h1_p"]["rejections"], POWER)

# Add manual (maximally) poor value in there as quick and dirty solution to 0-100 scaling in heatmap
h0_t1e[0], h1_t1e[0] = 100, 100 
h0_p[0], h1_p[0] = 0, 0 


# plot_result_surface(mean_nodes, variance_nodes, h0_t1e, color_map = "Greens_r", title = "valid type-I errors (H0)")
# plot_result_surface(mean_nodes, variance_nodes, h1_t1e, color_map = "Greens_r", title = "valid type-I errors (H1)")
# plot_result_surface(mean_nodes, variance_nodes, h0_p, color_map = "Greens", title = "valid power (H0)")
# plot_result_surface(mean_nodes, variance_nodes, h1_p, color_map = "Greens", title = "valid power (H1)")

h0_avg_power = average_prior_effect(data["h0_p"]["rejections"], data["h0_t1e"]["rejections"], weight_type1_error = 0.95)
h1_avg_power = average_prior_effect(data["h1_p"]["rejections"], data["h1_t1e"]["rejections"], weight_type1_error = 0.95)

# plot_result_surface(mean_nodes, variance_nodes, h0_avg_power, color_map = "RdYlGn", title = "H0: utility")
# plot_result_surface(mean_nodes, variance_nodes, h1_avg_power, color_map = "RdYlGn", title = "H1: utility")

h0_avg_power_valid = average_prior_effect(h0_p, h0_t1e, weight_type1_error = 0.9)
h1_avg_power_valid = average_prior_effect(h1_p, h1_t1e, weight_type1_error = 0.9)

plot_result_surface(mean_nodes, variance_nodes, h0_avg_power_valid, color_map = "Greens", title = "H0: Power (Type-I error < 0.05, Power > 0.8)")
plot_result_surface(mean_nodes, variance_nodes, h1_avg_power_valid, color_map = "Greens", title = "H1: Power (Type-I error < 0.05, Power > 0.8)")


# Sample size weighted
h0_p_weighted = penalise_samples(h0_avg_power_valid, data["h0_p"]["sample"], 0)
h1_p_weighted = penalise_samples(h1_avg_power_valid, data["h1_p"]["sample"], 0)

plot_result_surface(mean_nodes, variance_nodes, h0_p_weighted, color_map = "YlGn_r", title = f"H0: Power (Type-I error < {ALPHA/100}, Power > {POWER/100})")
plot_result_surface(mean_nodes, variance_nodes, h1_p_weighted, color_map = "YlGn_r", title = f"H1: Power (Type-I error < {ALPHA/100}, Power > {POWER/100})")


# plot_result_surface(mean_nodes, variance_nodes, data["h0_p"]["sample"], color_map = "RdYlGn_r", title = "H0: utility (Valid)")
# plot_result_surface(mean_nodes, variance_nodes, data["h1_p"]["sample"], color_map = "RdYlGn_r", title = "H1: utility (Valid)")

