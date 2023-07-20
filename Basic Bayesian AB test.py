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
# random.seed(0)

# H0: effect = 0, H1: effect = mde (note, not composite! though still practical for that purpose)
hypotheses = {"null": 0.6, "alt": 0.65}
C = {"n": 1000, "true_prob": 0.6} # control
T = {"n": 1000, "true_prob": 0.64} # treatment
prior = {"n": 1000, "weight": 25, "prior_control": 0.6, "prior_treatment": 0.64}



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
    bayes_factor = round(np.exp(log_bayes_factor), 3)

    return bayes_factor, alt_log_likelihood, null_log_likelihood

T["bayes_factor"], T["log_likelihood_H1"], T["log_likelihood_H0"] = log_likelihood_ratio_test(T["sample"]["converted"])
prob_H1 = round(T["bayes_factor"] / (T["bayes_factor"] + 1), 3)
print(f"\nBayes Factor: {T['bayes_factor']} w/ Probability H1: {prob_H1}")


"""
Part 3: Priors (Conjugate)
"""

def beta_prior(prior_prob, weight, n):
    # Sample from Beta distribution: B(weight(prior belief) + 1, weight(1 - prior belief) + 1)
    a = round(prior_prob, 1) * weight + 1
    b = (1 - round(prior_prob, 1)) * weight + 1
    beta_prior = stats.beta(a, b)
    samples = beta_prior.rvs(size = n)    
    return beta_prior, samples, a, b

C["prior_dist"], C["prior_sample"], C["prior_beta_a"], C["prior_beta_b"] = beta_prior(prior["prior_control"], prior["weight"], prior["n"])
T["prior_dist"], T["prior_sample"], T["prior_beta_a"], T["prior_beta_b"] = beta_prior(prior["prior_treatment"], prior["weight"], prior["n"])



"""
Part 4: Posteriors
"""
def beta_posterior(prior_a, prior_b, converted, n):
    # Beta distribution because prior Beta distribution is conjugate
    beta_posterior = stats.beta(prior_a + converted, prior_b + (n - converted))
    samples = beta_posterior.rvs(size = n) 
    return beta_posterior, samples

C["post_dist"], C["post_sample"] = beta_posterior(C["prior_beta_a"], C["prior_beta_b"], C["converted"], C["n"])
T["post_dist"], T["post_sample"] = beta_posterior(T["prior_beta_a"], T["prior_beta_b"], T["converted"], T["n"])



"""
Part 5
"""

def metrics():
    # Evaluate how often treatment outperformes control
    treatment_won = [i <= j for i, j in zip(C["post_sample"], T["post_sample"])]
    chance_of_beating_control = np.mean(treatment_won)
    print(f"Posterior probability of beating control: {round(chance_of_beating_control, 2)} (Quick & Dirty)")
    
    # Get treatment effect measurement
    treatment_effect = { 
            "true": round(T["true_prob"] - C["true_prob"], 4),
            "observed": round(T["sample_conversion_rate"] - C["sample_conversion_rate"], 4),
            "estimated": round(T["post_sample"].mean() - C["post_sample"].mean(), 4)
        }
    
    print(f"Treatment effect:\n- true: {treatment_effect['true']}\n- observed: {treatment_effect['observed']}\n- posterior: {treatment_effect['estimated']}")
    
    return treatment_effect

treatment_effect = metrics()

def plot():
    # Kernel density
    sns.kdeplot(C["post_sample"], label='Control', fill=True)
    sns.kdeplot(T["post_sample"], label='Treatment', fill=True)
    plt.xlabel('Probability')
    plt.title("Sampling from posterior distributions")
    plt.legend()
    plt.show()
    
    # Plot the histogram 
    plt.hist(C["post_sample"], bins=30, alpha=0.5, label='Control', density=True)
    plt.hist(T["post_sample"], bins=30, alpha=0.5, label='Treatment', density=True)
    plt.xlabel('Probability')
    plt.legend()
    
plot()

# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")

