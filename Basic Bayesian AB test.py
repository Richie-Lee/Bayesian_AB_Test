import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
import random

from datetime import datetime

# Track total runtime
_start_time = datetime.now()

"""
Part 0: Settings & Hyperparameters
"""
random.seed(0)

# H0: effect = 0, H1: effect = mde (note, not composite! though still practical for that purpose)
hypotheses = {"null": 0.5, "alt": 0.51}
C = {"n": 10_000, "true_prob": 0.5} # control
T = {"n": 10_000, "true_prob": 0.51} # treatment




"""
Part 1: Generate Data
"""

def get_bernoulli_sample(mean, n):
    # Sample bernoulli distribution with relevant metrics
    samples = [1 if random.random() < mean else 0 for _ in range(n)]
    converted = sum(samples)
    mean = converted/n

    # Create a DataFrame
    data = {
        "userId": range(1, n + 1),
        "converted": samples
    }
    data = pd.DataFrame(data)
    
    return data, converted, mean 

C["sample"], C["converted"], C["sample_conversion_rate"] = get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
T["sample"], T["converted"], T["sample_conversion_rate"] = get_bernoulli_sample(mean = T["true_prob"], n = T["n"])



"""
Part 2: Log Likelihoods 
"""
# Log is important, to prevent "float division by zero" as data dimensions increase and likelihoods converge to 0

def bernoulli_log_likelihood(hypothesis, outcomes):
    log_likelihood = 0.0
    for y in outcomes:
        if y == 1:
            log_likelihood += np.log(hypothesis)
        elif y == 0:
            log_likelihood += np.log(1 - hypothesis)
        else:
            raise ValueError("Outcomes must contain non-negative integers")

    return log_likelihood

def log_likelihood_ratio_test(treatment):
    # Get likelihoods
    null_log_likelihood = bernoulli_log_likelihood(hypotheses["null"], treatment)
    alt_log_likelihood = bernoulli_log_likelihood(hypotheses["alt"], treatment)

    # Compute BF: H1|H0
    log_bayes_factor = alt_log_likelihood - null_log_likelihood
    bayes_factor = np.exp(log_bayes_factor)

    return bayes_factor, alt_log_likelihood, null_log_likelihood

T["l_bayes_factor"], T["l_likelihood_H1"], T["l_likelihood_H0"] = log_likelihood_ratio_test(T["sample"]["converted"])
log_Prob_H1 = round(T["l_bayes_factor"] / (T["l_bayes_factor"] + 1) * 100, 3)






# Print execution time
print(f"===============================\nTotal runtime:  {datetime.now() - _start_time}")

