""" 
User input
""" 

# Specify DGP - [n, TE]

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