import numpy as np
import random

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import part_1_dgp as p1_dgp
import part_2_prior as p2_prior
import part_3_bayes_factors as p3_bf
import part_4_inference as p4_metrics
import part_5_repeat as p5_repeat
import part_6_visualisation as p6_plot

# Specify prior type: {beta, normal}
prior_type = "normal"

# Specify data type: {binary (bernoulli), continuous (normal), real}
data_type = "real"

"""
Part 1: DGP
"""
# Define Control & Treatment DGP
if data_type == "binary": # H0: C = T, H1: C != T
    C = {"n": 100_000, "true_prob": 0.4}
    T = {"n": 100_000, "true_prob": 0.39}
elif data_type == "continuous": # H0: C > T, H1: C < T
    C = {"n": 20000, "true_mean": 20, "true_variance": 3}
    T = {"n": 20000, "true_mean": 20.05, "true_variance": 3}
elif data_type == "real":
    data_config = {
        "import_directory": "/Users/richie.lee/Downloads/uk_orders_21_10_2023.csv",
        "voi": "order_food_total",
        "time_variable": "order_datetime_local",
        "start_time_hour": 22, "start_time_minute": 0,
        "n": 10000,
        }
    # Choose 1 way to apply simulated treatment effect (other value should be None)
    simulated_treatment_effect = {
        "relative_treatment_effect": None, # format as multiplier, e.g. 5% lift should be "1.05"
        "absolute_treatment_effect": 0, 
        }

# Part 1: Generate data
if data_type == "binary":
    # Bernoulli distributed Binary Data (e.g. Conversions)
    C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
    T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
elif data_type == "continuous":
    # Normal distributed continuous data
    C["sample"] = p1_dgp.get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
    T["sample"] = p1_dgp.get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])
elif data_type == "real":
    real_data_collector = p1_dgp.get_real_data(data_config, simulated_treatment_effect, SEED)
    C, T, real_data, voi = real_data_collector.get_values()

""" 
Part 2: Prior
""" 
# Prior odds (hypothesis prior)
prior_odds = 1 

# Marginal likelihood prior (parameter prior)
prior_parameters = {
    "beta": {
        "T_prior_prob": 0.39, "T_weight": 1000,
        "C_prior_prob": 0.4, "C_weight": 1000
    },
    "normal": {
        "mean_H0": 0.05, "variance_H0": 1,
        "mean_H1": 0, "variance_H1": 1
    }
}

if prior_type == "beta":
    prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
    C_prior, T_prior = prior_calculator.get_values()  # For binary data
elif prior_type == "normal":
    prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
    H0_prior, H1_prior = prior_calculator.get_values()  # For continuous data



""" 
Part 3: Bayes Factor
"""
early_stopping_settings = {
    "prob_early_stopping" : 0.99,
    "interim_test_interval" : 50,
    "minimum_sample" : 500
}

if prior_type == "beta":
    bf_calculator = p3_bf.get_bayes_factor(T, C, prior_type, early_stopping_settings, T_prior=T_prior, C_prior=C_prior)
    bf_fh, bf, interim_tests, early_stopping_settings["k"], sample_size = bf_calculator.get_values()
elif prior_type == "normal":
    bf_calculator = p3_bf.get_bayes_factor(T, C, prior_type, early_stopping_settings, H0_prior=H0_prior, H1_prior=H1_prior)
    bf_fh, bf, interim_tests, early_stopping_settings["k"], sample_size = bf_calculator.get_values()



"""
Part 4: Posterior & Inference
"""
# FIX: data that goes inside is wrong data, but the logic of calculating the metrics seems ok - though the probabilities seem off.
if prior_type == "beta":
    metrics_calculator = p4_metrics.get_metrics(T, C, prior_type, bf, prior_odds, T_prior=T_prior, C_prior=C_prior)
elif prior_type == "normal":
    metrics_calculator = p4_metrics.get_metrics(T, C, prior_type, bf, prior_odds, H0_prior=H0_prior, H1_prior=H1_prior)

metrics = metrics_calculator.get_values()



"""
Part 5: Repeat
"""
n_test = 100 # number of iterations
print_progress = True 
results, results_interim_tests = p5_repeat.multiple_iterations(T, C, prior_odds, prior_type, prior_parameters, early_stopping_settings, n_test, print_progress, data_type, data_config, simulated_treatment_effect)

"""
Part 6: Visualisation
"""
if prior_type == "beta":
    _visualisation = p6_plot.visualisation_bayes(T, C, early_stopping_settings, results, results_interim_tests, prior_odds, prior_type, T_prior, C_prior, None, None)
elif prior_type == "normal":
    _visualisation = p6_plot.visualisation_bayes(T, C, early_stopping_settings, results, results_interim_tests, prior_odds, prior_type, None, None, H0_prior, H1_prior)

print(f"avg n = {results['sample_size'].mean()}")

# Count H0 rejections
n_reject_es = (results['bayes_factor'] > early_stopping_settings["k"]).sum()
n_accept_es = (results['bayes_factor'] < 1 / early_stopping_settings["k"]).sum()
n_reject_fh = (results['bayes_factor_fh'] > early_stopping_settings["k"]).sum()
n_accept_fh = (results['bayes_factor_fh'] < 1 / early_stopping_settings["k"]).sum()
print(f"ES: [H0: {n_accept_es}, H1: {n_reject_es}] \nFH: [H0: {n_accept_fh}, H1: {n_reject_fh}, Unconclusive {n_test - n_accept_fh - n_reject_fh}]")







