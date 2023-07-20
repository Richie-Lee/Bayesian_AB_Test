import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
import random
from datetime import datetime


# Track total runtime
_start_time = datetime.now()
 

# Define a function to run the code and store the results
def run_code_with_seed(seed):
    # Set the seed for reproducibility
    random.seed(seed)
        
    """
    Part 0: Settings & Hyperparameters
    """
     
    # H0: effect = 0, H1: effect = mde (note, not composite! though still practical for that purpose)
    hypotheses = {"null": 0.6, "alt": 0.65, "mde": 0.05}
    _relative_loss_theshold = 0.05 # Used for loss -> e.g. 0.05 = 5% of prior effect deviation is accepted 
     
    # Define Control & Treatment DGP (Bernoulli distributed)
    C = {"n": 1000, "true_prob": 0.6} 
    T = {"n": 1000, "true_prob": 0.65}
     
    # Define Prior (Beta distributed -> Conjugate)
    prior = {"n": 1000, "weight": 25, "prior_control": 0.6, "prior_treatment": 0.7}
     
    # Early Stopping parameters (criteria in % for intuitive use-cases)
    sequential_testing = True
    early_stopping = {"stopping_criteria_prob": 95, "interim_test_interval": 10}

 
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
    Part 2.5: Early Stopping
    """
 
    def early_stopping_sampling(treatment):
        # Stopping criteria (symmetric) - computed using hyperparameter confidence %
        k =  early_stopping["stopping_criteria_prob"] / (100 - early_stopping["stopping_criteria_prob"])
        
        # Initialise
        bayes_factor, n_test = 0, 0
        early_stop = False
        interim_tests = []
        
        while early_stop == False:
            # sample        
            n_test += 1
            n_observed = n_test * early_stopping["interim_test_interval"]
            
            # Full data set utilised
            if n_observed > T["n"]:
                break
            
            data_observed = treatment[:n_observed]
            bayes_factor = log_likelihood_ratio_test(data_observed)
            
            # Stopping criteria
            if (bayes_factor > k or bayes_factor < 1/k):
                early_stop = True
            
            interim_tests.append((n_observed, bayes_factor))
        
        # Format new collections of info on treatment/control (slice control based on sample size of early stopping)
        T_ES = {
            "sample": data_observed,
            "converted": sum(data_observed),
            "sample_conversion_rate": round(sum(data_observed) / n_observed, 3),
            "bayes_factor": bayes_factor,
            "interim_tests": interim_tests,
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
            
        return T_ES, C_ES, k
 
    if sequential_testing == True:
        T_ES, C_ES, early_stopping["k"] = early_stopping_sampling(T["sample"])
        T, C = T_ES, C_ES
 
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
    Part 5: Reporting (Metrics & Visualisations)
    """
    def metrics():    

        # Evaluate how often treatment outperformes control
        treatment_won = [t - c >= hypotheses['mde'] for c, t in zip(C["post_sample"], T["post_sample"])]
        prob_TE_better_mde = round(np.mean(treatment_won), 2)
        treatment_won = [t >= c for c, t in zip(C["post_sample"], T["post_sample"])]
        prob_TE_positive = round(np.mean(treatment_won), 2)        
        
        # Get treatment effect measurement
        treatment_effect = { 
                "true": round(T["true_prob"] - C["true_prob"], 4),
                "observed": round(T["sample_conversion_rate"] - C["sample_conversion_rate"], 4),
                "estimated": round(T["post_sample"].mean() - C["post_sample"].mean(), 4),
                "prior": round(prior["prior_treatment"] - prior["prior_control"], 4)
            }
                
        # Compute loss (Reward/Penalise not choosing probability closest to the truth, by difference |T-C|)
        loss_control = [max(j - i, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
        loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
        loss_control = round(np.mean(loss_control), 4)
        
        loss_treatment = [max(i - j, 0) for i,j in zip(C["post_sample"], T["post_sample"])]
        loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]
        loss_treatment = round(np.mean(loss_treatment), 4)
                
        return treatment_effect, loss_control, loss_treatment, prob_TE_better_mde, prob_TE_positive
 
    treatment_effect, C["loss"], T["loss"], treatment_effect["p[TE>mde]"], treatment_effect["p[T>C]"] = metrics()
    
    # =================================================================
    # Store the results in a dictionary
    result = {
        "seed": seed,
        "sample_size": T["n"],
        "bayes_factor": T["bayes_factor"],
        "prob_H1": round(T["bayes_factor"] / (T["bayes_factor"] + 1), 3),
        "treatment_effect": treatment_effect["estimated"],
        "P[T>C]" : treatment_effect["p[T>C]"],
        "P[TE>mde]" : treatment_effect["p[TE>mde]"],
    }
    
    interim_tests = T["interim_tests"]
    
    return result, early_stopping, interim_tests

# Define the number of runs and a list to store the results
num_runs = 50
results = []
results_interim_tests = []

# Run the code with different seeds and store the results
for _i in range(num_runs):
    _seed = _i  # Or use any other method to generate different seeds
    _result, early_stopping, interim_tests = run_code_with_seed(_seed)
    results.append(_result)
    results_interim_tests.append(interim_tests)


def early_stopping_dist():
    # plot stopping criteria
    plt.axhline(y = 1, color = "black", linestyle = "--", linewidth = "0.6")
    plt.axhline(y = early_stopping["k"], color = "black", linestyle = "--", linewidth = "0.6")
    plt.axhline(y = 1/early_stopping["k"], color = "black", linestyle = "--", linewidth = "0.6")
        
    for i in range(len(results_interim_tests)):
        x, y = zip(*results_interim_tests[i])
        
        if results[i]["bayes_factor"] > 1: # Reject H0 (Effect discovery)
            color_i = "blue"
        else: # Accept H0 (No effect)
            color_i = "red"
        plt.plot(x, y, linestyle = "-", alpha = 0.5, linewidth = 0.7, color = color_i)
    
    # Set the y-axis to log scale
    plt.yscale('log')
    
    plt.xlabel("Sample size")
    plt.ylabel("Bayes_factor")
    plt.title(f"Bayes Factors during interim testing ({num_runs})")
    plt.plot()

early_stopping_dist()


# def plot_posteriors():
    
#     for i in results:
        
#     # Plot the histogram + kernel (Posterior)
#     plt.hist(C["post_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[0])
#     plt.hist(T["post_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[1])
#     sns.kdeplot(C["post_sample"], label='Control', fill = False, color = _colors[0])
#     sns.kdeplot(T["post_sample"], label='Treatment', fill = False, color = _colors[1])
#     plt.xlabel('Probability')
#     plt.legend()
#     plt.title("Samples from posterior distributions")
#     plt.show()

# plot_posteriors()






# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")
