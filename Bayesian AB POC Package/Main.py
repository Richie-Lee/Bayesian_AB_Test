""" 
User input
""" 

# Specify DGP

# Specify Prior odds - [odds]

# Specify ML-Prior (Type) - [parameters]

# Specify Early stopping settings - [k, interval size, print_progress]



""" 
Modules
""" 

""" 
# Data generation
- in: n, TE (means)
- out: outcomes, converted, conversion_rate, n, true_prob


# Prior: visualisation?, (transformed) parameters
- in: prior-type, parameters
- out: formal parameters, visualisation

# BF (ES) + Posterior inference: 
- in: prior-type, prior-param, outcomes
- out: Interim_test_BF, posterior_prob (conclusion), ES_n
    
# Performance metrics: 
- in: True_prob, outcomes, prior-type, prior_param
- out: Uplift, Chance_T_beat_C, Loss


REPEAT

# Visualistion

""" 

import pandas as pd
import random
from datetime import datetime

import part_1_dgp as p1_dgp
import part_2_prior as p2_prior

# Define Control & Treatment DGP (Bernoulli distributed)
C = {"n": 10_000, "true_prob": 0.5}
T = {"n": 10_000, "true_prob": 0.53}

""" 
Part 1: Generate data
""" 

C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = T["true_prob"], n = T["n"])


""" 
Part 2: Prior
""" 

prior_odds = 1 


prior_type = "beta"
prior_parameters = {
    "beta" : {"T_prior_prob" : 0.5, "T_weight" : 1000, "C_prior_prob" : 0.52,"C_weight" : 1000},
    "right haar" : {"param":None}
    }

prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
C_prior, T_prior = prior_calculator.get_values()




