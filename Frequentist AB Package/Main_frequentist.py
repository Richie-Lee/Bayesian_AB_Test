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


# Specify test type: {naive t-test}
test_type = "naive t-test"
data_type = "continuous" 
 

"""
Part 1: DGP
"""
# Define Control & Treatment DGP
if data_type == "binary": # H0: C = T, H1: C != T
    C = {"n": 100_000, "true_prob": 0.4}
    T = {"n": 100_000, "true_prob": 0.41}
elif data_type == "continuous": # H0: C > T, H1: C < T
    C = {"n": 5000, "true_mean": 20.1, "true_variance": 3}
    T = {"n": 5000, "true_mean": 20, "true_variance": 3}

# Part 1: Generate data
if data_type == "binary":
    # Bernoulli distributed Binary Data (Conversions)
    C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
    T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
elif data_type == "continuous":
    # Continuous data
    C["sample"] = p1_dgp.get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
    T["sample"] = p1_dgp.get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])

""" 
Part 3: p-value (skip part 2 - priors)
"""
early_stopping_settings = {
    "alpha" : 0.05,
    "interim_test_interval" : 10,
    "minimum_sample" : 500
    }

if test_type == "naive t-test":
    p_value_calculator = p3_p.get_p_value(T, C, early_stopping_settings)
    p_value_fh, p_value_es, interim_tests, sample_size = p_value_calculator.get_values()

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
results, results_interim_tests = p5_repeat.multiple_iterations(T, C, data_type, test_type, early_stopping_settings, n_test, print_progress)

"""
Part 6: Visualisation
"""
_visualisation = p6_plot.visualisation_frequentist(T, C, early_stopping_settings, results, results_interim_tests)
_visualisation.get_results()


# Print true label & sample average
print(f"avg n = {results['sample_size'].mean()}")
if data_type == "binary":
    print(f"H0: C > T -> {True if C['true_prob'] > T['true_prob'] else False}")
if data_type == "continuous":
    print(f"H0: C > T -> {True if C['true_mean'] > T['true_mean'] else False}")

# Count H0 rejections
n_reject_es = (results['p_value'] < early_stopping_settings["alpha"]).sum()
n_reject_fh = (results['p_value_fh'] < early_stopping_settings["alpha"]).sum()
print(f"ES: {n_reject_es}/{n_test} \nFH: {n_reject_fh}/{n_test}")


