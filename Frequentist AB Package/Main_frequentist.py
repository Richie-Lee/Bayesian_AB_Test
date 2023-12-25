import numpy as np
import random

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import part_1_dgp as p1_dgp
import part_3_p_values as p3_p
import part_4_inference as p4_metrics
import part_5_repeat as p5_repeat
import part_6_visualisation as p6_plot


# Specify test type: {naive t-test, alpha spending}
test_type = "alpha spending"
data_type = "real" 
 

"""
Part 1: DGP
"""
# Define Control & Treatment DGP
if data_type == "binary": # H0: C > T, H1: C < T
    C = {"n": 50000, "true_prob": 0.41}
    T = {"n": 50000, "true_prob": 0.4}
elif data_type == "continuous": # H0: C > T, H1: C < T
    C = {"n": 10000, "true_mean": 20.05, "true_variance": 3}
    T = {"n": 10000, "true_mean": 20, "true_variance": 3}
elif data_type == "real": # H0: C > T, H1: C < T
    data_config = {
        "import_directory": "/Users/richie.lee/Downloads/uk_orders_21_10_2023.csv",
        "voi": "order_food_total",
        "time_variable": "order_datetime_local",
        "start_time_hour": 0, "start_time_minute": 0,
        "n": 10000,
        }
    # Choose 1 way to apply simulated treatment effect (other value should be None)
    simulated_treatment_effect = {
        "relative_treatment_effect": 0.95, # format as multiplier, e.g. 5% lift should be "1.05" (H0 true if multiplier < 1)
        "absolute_treatment_effect": None, 
        }

# Part 1: Generate data
if data_type == "binary":
    # Bernoulli distributed Binary Data (Conversions)
    C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
    T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
elif data_type == "continuous":
    # Continuous data
    C["sample"] = p1_dgp.get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
    T["sample"] = p1_dgp.get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])
elif data_type == "real":
    real_data_collector = p1_dgp.get_real_data(data_config, simulated_treatment_effect, SEED)
    C, T, real_data, voi = real_data_collector.get_values()


""" 
Part 3: p-value (skip part 2 - priors)
"""
early_stopping_settings = {
    "alpha" : 0.05,
    "interim_test_interval" : 50,
    "minimum_sample" : 500
    }

p_value_calculator = p3_p.get_p_value(T, C, early_stopping_settings, test_type)
p_value_fh, p_value_es, interim_tests, sample_size, alpha = p_value_calculator.get_values()

"""
Part 4: Posterior & Inference
"""
metrics_calculator = p4_metrics.get_metrics(T, C, data_type)
metrics = metrics_calculator.get_values()

"""
Part 5: Repeat
"""
n_test = 100 # number of iterations
print_progress = True 
results, results_interim_tests = p5_repeat.multiple_iterations(T, C, data_type, test_type, early_stopping_settings, n_test, print_progress, data_config, voi)

"""
Part 6: Visualisation
"""
_visualisation = p6_plot.visualisation_frequentist(T, C, early_stopping_settings, results, results_interim_tests, test_type, data_type, data_config)
power_curve_data = _visualisation.get_results()


# Print true label & sample average
print(f"avg n = {results['sample_size'].mean()}")
if data_type == "binary":
    print(f"H0: C = T -> {True if C['true_prob'] > T['true_prob'] else False}")
if data_type == "continuous":
    print(f"H0: C = T -> {True if C['true_mean'] > T['true_mean'] else False}")

# Count H0 rejections TO DO -> CORRECT DUE TO ALPHA SPENDING
n_reject_es = (results['p_value'] < results["alpha"]).sum()
n_reject_fh = (results['p_value_fh'] < early_stopping_settings["alpha"]).sum()
print(f"ES: {n_reject_es}/{n_test} rejected\nFH: {n_reject_fh}/{n_test} rejected")


