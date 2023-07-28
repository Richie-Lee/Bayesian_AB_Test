import random
from datetime import datetime

import data_generation as dgp
import likelihood_and_early_stopping as likelihood
import prior_posterior_calculation as pp
import reporting as rpt
# import visualization as viz

# Track total runtime
_start_time = datetime.now()



"""
Specify Settings & Hyperparameters
"""
random.seed(2)

# H0: effect = 0, H1: effect = mde (note, not composite! though still practical for that purpose)
hypotheses = {"null": 0.6, "alt": 0.65, "mde": 0.05}
_relative_loss_theshold = 0.05 # Used for loss -> e.g. 0.05 = 5% of prior effect deviation is accepted 

# Define Control & Treatment DGP (Bernoulli distributed)
C = {"n": 1_000, "true_prob": 0.6} 
T = {"n": 1_000, "true_prob": 0.55}

# Define Prior (Beta distributed -> Conjugate)
prior = {"n": 1000, "weight": 25, "prior_control": 0.6, "prior_treatment": 0.7}

# Early Stopping parameters (criteria in % for intuitive use-cases)
early_stopping = {"enabled": False, "stopping_criteria_prob": 95, "interim_test_interval": 10}



"""
Module 1: Generate Data
"""
C["sample"], C["converted"], C["sample_conversion_rate"] = dgp.get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
T["sample"], T["converted"], T["sample_conversion_rate"] = dgp.get_bernoulli_sample(mean = T["true_prob"], n = T["n"])



"""
Module 2: Log Likelihoods & Early stopping (if enabled)
"""
T["bayes_factor"] = likelihood.log_likelihood_ratio_test(T["sample"], hypotheses)
T, C, early_stopping["k"] = likelihood.early_stopping_sampling(T, C, early_stopping, hypotheses)



"""
Module 3: Priors & corresponding Posterior 
"""
# Create an instance of the class
_prior_posterior_instance = pp.prior_posterior(prior, C, T, model_selection="beta")

# Get the results as a tuple
_C_prior, _C_prior_sample, _C_post, _C_post_sample, _T_prior, _T_prior_sample, _T_post, _T_post_sample = _prior_posterior_instance.get_results()

# Update the dictionary values with the calculated results
C["prior_dist"], C["prior_sample"], C["post_dist"], C["post_sample"] = _C_prior, _C_prior_sample, _C_post, _C_post_sample
T["prior_dist"], T["prior_sample"], T["post_dist"], T["post_sample"] = _T_prior, _T_prior_sample, _T_post, _T_post_sample



"""
Module 4: Reporting 
"""
treatment_effect = rpt.metrics(T, C, prior, hypotheses)



# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")