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
prior_type = "beta"  

"""
Part 1: DGP
"""
# Define Control & Treatment DGP
if prior_type == "beta": # H0: C = T, H1: C != T
    C = {"n": 100_000, "true_prob": 0.4}
    T = {"n": 100_000, "true_prob": 0.39}
elif prior_type == "normal": # H0: C > T, H1: C < T
    C = {"n": 100_000, "true_mean": 19.9, "true_variance": 3}
    T = {"n": 100_000, "true_mean": 20, "true_variance": 3}
    
# Part 1: Generate data
if prior_type == "beta":
    # Bernoulli distributed Binary Data (Conversions)
    C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
    T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
elif prior_type == "normal":
    # Continuous data
    C["sample"] = p1_dgp.get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
    T["sample"] = p1_dgp.get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])

""" 
Part 2: Prior
""" 
# Prior odds (hypothesis prior)
prior_odds = 1 

# Marginal likelihood prior (parameter prior)
prior_parameters = {
    "beta": {
        "T_prior_prob": 0.41, "T_weight": 1000,
        "C_prior_prob": 0.4, "C_weight": 1000
    },
    "normal": {
        "mean_H0": 0, "variance_H0": 3,
        "mean_H1": -0.3, "variance_H1": 3
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
    "prob_early_stopping" : 0.9,
    "interim_test_interval" : 100,
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
results, results_interim_tests = p5_repeat.multiple_iterations(T, C, prior_odds, prior_type, prior_parameters, early_stopping_settings, n_test, print_progress)

"""
Part 6: Visualisation
"""
if prior_type == "beta":
    _visualisation = p6_plot.visualisation_bayes(T, C, early_stopping_settings, results, results_interim_tests, prior_odds, prior_type, T_prior, C_prior, None, None)
elif prior_type == "normal":
    _visualisation = p6_plot.visualisation_bayes(T, C, early_stopping_settings, results, results_interim_tests, prior_odds, prior_type, None, None, H0_prior, H1_prior)

print(f"avg n = {results['sample_size'].mean()}")








