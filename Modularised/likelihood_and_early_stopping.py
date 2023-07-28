import numpy as np
    
"""
Part 2: Log Likelihoods 
"""
# Log is important, to prevent "float division by zero" as data dimensions increase and likelihoods converge to 0

def bernoulli_log_likelihood(hypothesis_mean, outcomes):
    log_likelihood = 0.0
    for y in outcomes:
        if y == 1:
            log_likelihood += np.log(hypothesis_mean)
        elif y == 0:
            log_likelihood += np.log(1 - hypothesis_mean)
        else:
            raise ValueError("Outcomes must contain non-negative integers")

    return log_likelihood

def log_likelihood_ratio_test(treatment_sample, hypotheses):
    # Get likelihoods
    null_log_likelihood = bernoulli_log_likelihood(hypotheses["null"], treatment_sample)
    alt_log_likelihood = bernoulli_log_likelihood(hypotheses["alt"], treatment_sample)

    # Compute BF: H1|H0
    log_bayes_factor = alt_log_likelihood - null_log_likelihood
    bayes_factor = round(np.exp(log_bayes_factor), 3)

    return bayes_factor




"""
Part 2.5: Early Stopping
"""

def early_stopping_sampling(T, C, early_stopping, hypotheses):
    # Skip interim testing if desired
    if early_stopping["enabled"] == False:
        return T, C, None # Bayes Factor parameter k is undefined
    
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
        
        data_observed = T["sample"][:n_observed]
        bayes_factor = log_likelihood_ratio_test(data_observed, hypotheses)
        print(f"n: {n_observed}/{T['n']}, BF: {bayes_factor}")
        
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
