import random

import part_1_dgp as p1_dgp
import part_2_prior as p2_prior
import part_3_bayes_factors as p3_bf
import part_4_inference as p4_metrics
import part_5_repeat as p5_repeat
import part_6_visualisation as p6_plot

# Control randomness for reproducibility
random.seed(1)

# Define Control & Treatment DGP (Bernoulli distributed)
C = {"n": 10_000, "true_prob": 0.5}
T = {"n": 10_000, "true_prob": 0.51}

""" 
Part 1: Generate data
""" 
# Bernoulli distributed Binary Data (Conversions)
C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = T["true_prob"], n = T["n"])


""" 
Part 2: Prior
""" 
# Prior odds (hypothesis prior)
prior_odds = 1 

# Marginal likelihood prior (parameter prior)
prior_type = "beta"
prior_parameters = {
    "beta" : {"T_prior_prob" : 0.51, "T_weight" : 10000, "C_prior_prob" : 0.5,"C_weight" : 10000},
    "right haar" : {"param" : None}
    }

prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
C_prior, T_prior = prior_calculator.get_values()



""" 
Part 3: Bayes Factor
"""
early_stopping_settings = {
    "prob_early_stopping" : 0.95,
    "interim_test_interval" : 100,
    }

bf_calculator = p3_bf.get_bayes_factor(T, T_prior, C, C_prior, prior_type, early_stopping_settings)
bf_fh, bf, interim_tests, early_stopping_settings["k"], sample_size = bf_calculator.get_values()



"""
Part 4: Posterior & Inference
"""
metrics_calculator = p4_metrics.get_metrics(T, T_prior, C, C_prior, prior_type, bf, prior_odds)
metrics = metrics_calculator.get_values()



"""
Part 5: Repeat
"""
n_test = 500 # number of iterations
print_progress = True 
results, results_interim_tests = p5_repeat.multiple_iterations(T, C, prior_odds, prior_type, prior_parameters, early_stopping_settings, n_test, print_progress)



"""
Part 6: Visualisation
"""
_visualisation = p6_plot.visualisation_bayes(T, C, early_stopping_settings, results, results_interim_tests)











