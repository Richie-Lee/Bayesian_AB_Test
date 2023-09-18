import pandas as pd
import random

import part_1_dgp as p1_dgp
import part_2_prior as p2_prior
import part_3_bayes_factors as p3_bf
import part_4_inference as p4_metrics

from datetime import datetime

def multiple_iterations(T, C, prior_odds, prior_type, prior_parameters, early_stopping_settings, n_test, print_progress):    
    
    # Track runtime
    startTime = datetime.now()
    
    results = []
    results_columns = ["seed", "n", "P[H1|data]", "uplift", "P[T>C]", "loss"]
    all_interim_tests = []
    
    for i in range(n_test):
        # Set new random seed
        random.seed(i)
        
        if print_progress == True:
            print(f"{i}/{n_test}")
    
        # Part 1: Generate data
        C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
        T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean = T["true_prob"], n = T["n"])
        
        # Part 2: prior
        prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
        C_prior, T_prior = prior_calculator.get_values()
        
        # Part 3: bayes factor
        bf_calculator = p3_bf.get_bayes_factor(T, T_prior, C, C_prior, prior_type, early_stopping_settings)
        bf, interim_tests, k, sample_size = bf_calculator.get_values()
        
        # Part 4: inference
        metrics_calculator = p4_metrics.get_metrics(T, T_prior, C, C_prior, prior_type, bf, prior_odds)
        metrics = metrics_calculator.get_values()
        
        # Store results
        results.append([i, sample_size, metrics["P[H1|data]"], metrics["uplift"], metrics["P[T>C]"], metrics["loss"]])
        all_interim_tests.append(interim_tests)
        
    results = pd.DataFrame(results, columns = results_columns)
    
    # Print runtime
    print(f"\ntotal runtime: {datetime.now() - startTime}")
    
    return results, all_interim_tests