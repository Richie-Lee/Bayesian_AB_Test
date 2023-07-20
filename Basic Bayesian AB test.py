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
# H0: effect = 0, H1: effect = mde (note, not composite! though still practical for that purpose)
hypotheses = {"null": 0.5, "alt": 0.52}
C = {"n": 1_000, "true_prob": 0.5} # control
T = {"n": 1_000, "true_prob": 0.55} # treatment




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
Part 2: Likelihoods
"""

def bernoulli_likelihood(hypothesis, outcomes):
    likelihood = 1.0
    for y in outcomes:
        if y == 1:
            likelihood *= hypothesis
        elif y == 0:
            likelihood *= (1 - hypothesis)
        else:
            raise ValueError("Outcomes must contain non-negative integers")

    return likelihood

def likelihood_ratio_test(treatment):
    # Get likelihoods
    null_likelihood = bernoulli_likelihood(hypotheses["null"], treatment)
    alt_likelihood = bernoulli_likelihood(hypotheses["alt"], treatment)

    # Compute Bayes Factor: p(data|H1) / p(data|H0)
    bayes_factor = alt_likelihood / null_likelihood

    return bayes_factor, alt_likelihood, null_likelihood

T["bayes_factor"], T["likelihood_H1"], T["likelihood_H0"] = likelihood_ratio_test(T["sample"]["converted"])

Prob_H1 = round(T["bayes_factor"] / (T["bayes_factor"] + 1) * 100, 1)






# Print execution time
print(f"===============================\nTotal runtime:  {datetime.now() - _start_time}")

