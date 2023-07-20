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
hypotheses = {"null": 0.6, "alt": 0.7, "mde": 0.05}
_relative_loss_theshold = 0.05 # Used for loss -> e.g. 0.05 = 5% of prior effect deviation is accepted 

# Define Control & Treatment DGP (Bernoulli distributed)
C = {"n": 1000, "true_prob": 0.6} 
T = {"n": 1000, "true_prob": 0.65}

# Define Prior (Beta distributed -> Conjugate)
prior = {"n": 1000, "weight": 5, "prior_control": 0.6, "prior_treatment": 0.7}


"""
Part 1: Generate Data
"""

def get_bernoulli_sample(mean, n):
    # Sample bernoulli distribution with relevant metrics
    samples = [1 if random.random() < mean else 0 for _ in range(n)]
    converted = sum(samples)
    mean = converted/n
    
    return samples, converted, mean 

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

    return bayes_factor

T["bayes_factor"] = log_likelihood_ratio_test(T["sample"])




"""
TO DO: warmup period + plot + cleanup code + sanity checks
Part 2.5: Early Stopping
"""
# Early Stopping criteria (give in %)
early_stopping = {"stopping_criteria_prob": 95, "interim_test_interval": 10}

def early_stopping_sampling(treatment):
    # Stopping criteria (symmetric) - computed using hyperparameter confidence %
    k =  early_stopping["stopping_criteria_prob"] / (100 - early_stopping["stopping_criteria_prob"])
    
    # Initialise
    bayes_factor, n_test = 0, 0
    early_stop = False
    all_bayes_factors = []
    
    while early_stop == False:
        # sample        
        n_test += 1
        n_observed = n_test * early_stopping["interim_test_interval"]
        
        # Full data set utilised
        if n_observed > T["n"]:
            break
        
        data_observed = treatment[:n_observed]
        bayes_factor = log_likelihood_ratio_test(data_observed)
        print(f"n: {n_observed}/{T['n']}, BF: {bayes_factor}")
        
        # Stopping criteria
        if bayes_factor > k or bayes_factor < 1/k:
            early_stop = True
        
        all_bayes_factors.append((n_observed, bayes_factor))
    
    T_ES = {
        "sample": data_observed,
        "converted": sum(data_observed),
        "sample_conversion_rate": round(sum(data_observed) / n_observed, 3),
        "bayes_factor": bayes_factor,
        "bayes_factor_full": all_bayes_factors,
        "early_stop": early_stop,
        "n": n_observed,
        "n_test": n_test,
        "true_prob": T["true_prob"]
        }
    
    C_ES = {
        "sample": C["sample"][:n_observed],
        "converted": sum(C["sample"][:n_observed]),
        "sample_conversion_rate": round(sum(C["sample"][:n_observed]) / n_observed, 3),
        "n": n_observed,
        "true_prob": T["true_prob"]
        }
        
        
    return T_ES, C_ES

T_ES, C_ES = early_stopping_sampling(T["sample"])

raise SystemExit(0)

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

# Print hypotheses:
print(f"\n===============================\nH0: y = {hypotheses['null']}, H1: y = {hypotheses['alt']}\nmde = {hypotheses['mde']} \n===============================")

def metrics():
    prob_H1 = round(T["bayes_factor"] / (T["bayes_factor"] + 1), 3)
    print(f"\nInformal probabilities: \nP[H1|BF]: {prob_H1}")
    
    # Evaluate how often treatment outperformes control
    treatment_won = [t - c >= hypotheses['mde'] for c, t in zip(C["post_sample"], T["post_sample"])]
    chance_of_beating_control = np.mean(treatment_won)
    print(f"P[T - C >= mde]: {round(chance_of_beating_control, 2)}")
    
    # Get treatment effect measurement
    treatment_effect = { 
            "true": round(T["true_prob"] - C["true_prob"], 4),
            "observed": round(T["sample_conversion_rate"] - C["sample_conversion_rate"], 4),
            "estimated": round(T["post_sample"].mean() - C["post_sample"].mean(), 4),
            "prior": round(prior["prior_treatment"] - prior["prior_control"], 4)
        }
    
    print(f"\nTreatment effect:\n- true: {treatment_effect['true']}\n- observed: {treatment_effect['observed']}\n- prior: {treatment_effect['prior']}\n- posterior: {treatment_effect['estimated']}")
    
    # Compute loss (Reward/Penalise not choosing probability closest to the truth, by difference |T-C|)
    loss_control = [max(j - i, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
    loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
    loss_control = round(np.mean(loss_control), 4)
    
    loss_treatment = [max(i - j, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
    loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]
    loss_treatment = round(np.mean(loss_treatment), 4)
    
    print(f"\nLoss (acceptable = {round(treatment_effect['prior'] * _relative_loss_theshold, 4)}):\n- Treatment: {loss_treatment}\n- Control: {loss_control}")
    
    return treatment_effect, loss_control, loss_treatment

treatment_effect, C["loss"], T["loss"] = metrics()


# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

def plot():
    # Plot the histogram + kernel (Posterior)
    plt.hist(C["post_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[0])
    plt.hist(T["post_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[1])
    sns.kdeplot(C["post_sample"], label='Control', fill = False, color = _colors[0])
    sns.kdeplot(T["post_sample"], label='Treatment', fill = False, color = _colors[1])
    plt.xlabel('Probability')
    plt.legend()
    plt.title("Samples from posterior distributions")
    plt.show()
    
plot()

# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")

