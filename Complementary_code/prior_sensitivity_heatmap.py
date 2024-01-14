import numpy as np
from scipy.stats import norm
import random
from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime

import part_1_dgp as p1_dgp

def get_normal_sample(mean, variance, n):
    sample = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return sample

def normal_bf(y, sigma_squared, H0_prior, H1_prior):
        # Get parameters
        mu_h0, sigma_h0 = H0_prior["mean"], np.sqrt(H0_prior["variance"])
        mu_h1, sigma_h1 = H1_prior["mean"], np.sqrt(H1_prior["variance"])

        # Updated mean / standard deviation
        mu_h0_prime = (y * sigma_h0**2 + mu_h0 * sigma_squared) / (sigma_h0**2 + sigma_squared)
        mu_h1_prime = (y * sigma_h1**2 + mu_h1 * sigma_squared) / (sigma_h1**2 + sigma_squared)
        sigma_h0_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h0**2))
        sigma_h1_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h1**2))

        # Log likelihood for H0 and H1
        log_likelihood_h0 = norm.logcdf(-mu_h0_prime / sigma_h0_prime) - norm.logcdf(-mu_h0 / sigma_h0)
        log_likelihood_h1 = np.log(1 - norm.cdf(-mu_h1_prime / sigma_h1_prime)) - np.log(1 - norm.cdf(-mu_h1 / sigma_h1))

        # Exponent term calculation
        exponent_term = 0.5 * (
            (mu_h1**2 / (sigma_squared + sigma_h1**2)) - (mu_h1_prime**2 / (sigma_squared + sigma_h1_prime**2)) +
            (mu_h0_prime**2 / (sigma_squared + sigma_h0_prime**2)) - (mu_h0**2 / (sigma_squared + sigma_h0**2))
        )

        # Calculate bayes factor
        log_bf = log_likelihood_h1 - log_likelihood_h0 + exponent_term
        bf = np.exp(log_bf)

        return bf

def get_es_rejections_acceptances(prior_mean, prior_variance, iteration):
    final_sample_sizes = []
    final_bfs = []
    for i in range(n_test):
        random.seed(i)

        # PART 1: Get data
        C["sample"] = get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
        T["sample"] = get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])

        # PART 2: Get prior
        if priors_to_be_tested == "H0":
          H0_prior = {"mean": prior_mean, "variance": prior_variance}
          H1_prior = {"mean": T["true_mean"] - C["true_mean"], "variance": T["true_variance"] + C["true_variance"]}
        elif priors_to_be_tested == "H1":
          H0_prior = {"mean": C["true_mean"], "variance": T["true_variance"] + C["true_variance"]}
          H1_prior = {"mean": prior_mean, "variance": prior_variance}

        # PART 3: Get BF (with early stopping)
        n_observed, min_sample = 0, 0
        while n_observed <= T["n"]:
            T_sample = T["sample"][:n_observed]
            C_sample = C["sample"][:n_observed]
            y_bar = np.mean(T_sample) - np.mean(C_sample)
            pooled_variance = np.var(T_sample)/len(T_sample) + np.var(C_sample)/len(C_sample)
            bf = normal_bf(y_bar, pooled_variance, H0_prior, H1_prior)

            if n_observed >= min_sample:
                if (bf > k or bf < 1/k) or n_observed == T["n"]:
                    break

            n_observed += early_stopping_settings["interim_test_interval"]

        final_sample_sizes.append(n_observed)
        final_bfs.append(bf)

    h0_rejections = sum(value > 1 for value in final_bfs)
    avg_sample = sum(final_sample_sizes) / len(final_sample_sizes)

    print(f"{iteration}: ({prior_mean}, {prior_variance}), {h0_rejections}, {avg_sample}")

    return h0_rejections, avg_sample

data_config = {
    "import_directory": "/Users/richie.lee/Downloads/uk_orders_21_10_2023.csv",
    "voi": "order_food_total",
    "time_variable": "order_datetime_local",
    "start_time_hour": 0, "start_time_minute": 0,
    "n": 50000,
    }

def get_es_rejections_acceptances_real(prior_mean, prior_variance, iteration, C, T):   
    final_sample_sizes = []
    final_bfs = []
    for i in range(n_test):
        random.seed(i)

        # PART 1: Get data
        data_C_df = C["df"].sample(n = data_config["n"], random_state = i)
        data_T_df = T["df"].sample(n = data_config["n"], random_state = i)
        
        # Sort by time for chronological data (At this point, it should already be in datetime format)
        data_C_df = data_C_df.sort_values(by = "time", ascending = True)
        data_T_df = data_T_df.sort_values(by = "time", ascending = True)
            
        # samples (df -> np array)
        C_sample = data_C_df[voi].to_numpy()
        T_sample = data_T_df[voi].to_numpy()
        
        C = {"n": len(C_sample), "sample": C_sample, "true_mean": C_sample.mean(), "true_variance": C_sample.var(), "df": C["df"]}
        T = {"n": len(T_sample), "sample": T_sample, "true_mean": T_sample.mean(), "true_variance": T_sample.var(), "df": T["df"]}

        
        # PART 2: Get prior
        if priors_to_be_tested == "H0":
          H0_prior = {"mean": prior_mean, "variance": prior_variance}
          H1_prior = {"mean": T["true_mean"] - C["true_mean"], "variance": T["true_variance"] + C["true_variance"]}
        elif priors_to_be_tested == "H1":
          H0_prior = {"mean": C["true_mean"], "variance": T["true_variance"] + C["true_variance"]}
          H1_prior = {"mean": prior_mean, "variance": prior_variance}


        # PART 3: Get BF (with early stopping)
        n_observed, min_sample = 0, 0
        while n_observed <= T["n"]:
            T_sample = T["sample"][:n_observed]
            C_sample = C["sample"][:n_observed]
            y_bar = np.mean(T_sample) - np.mean(C_sample)
            pooled_variance = np.var(T_sample)/len(T_sample) + np.var(C_sample)/len(C_sample)
            bf = normal_bf(y_bar, pooled_variance, H0_prior, H1_prior)

            if n_observed >= min_sample:
                if (bf > k or bf < 1/k) or n_observed == T["n"]:
                    break

            n_observed += early_stopping_settings["interim_test_interval"]

        final_sample_sizes.append(n_observed)
        final_bfs.append(bf)

    h0_rejections = sum(value > 1 for value in final_bfs)
    avg_sample = sum(final_sample_sizes) / len(final_sample_sizes)

    print(f"{iteration}: ({prior_mean}, {prior_variance}), {h0_rejections}, {avg_sample}")

    return h0_rejections, avg_sample


combinations = [
    [-0.1, "H0"],
    [0.1, "H0"],
    [-0.1, "H1"],
    [0.1, "H1"]
    ]
    



for combination in combinations:
    EFFECT = combination[0]
    PRIOR = combination[1]
    
    # # DGP
    # Choose 1 way to apply simulated treatment effect (other value should be None)
    simulated_treatment_effect = {
        "relative_treatment_effect": 1 + EFFECT, # format as multiplier, e.g. 5% lift should be "1.05" (H0 true if multiplier < 1)
        "absolute_treatment_effect": None, 
        }

    # Import data (from local file)
    real_data_collector = p1_dgp.get_real_data(data_config, simulated_treatment_effect, SEED = 0)
    C, T, real_data, voi = real_data_collector.get_values()
    print(C["true_mean"], T["true_mean"])

    # C = {"n": 200_000, "true_mean": 1, "true_variance": 1}
    # T = {"n": 200_000, "true_mean": 1 + EFFECT, "true_variance": 1}
    
    # Early stopping settings
    early_stopping_settings = {
        "prob_early_stopping" : 0.95,
        "interim_test_interval" : 200,
        "minimum_sample" : 1}
    n_test = 100
    k = 19
    
    # Specify which prior to test values on [pair ]
    priors_to_be_tested = PRIOR
    
    # Number of grid points per axis
    num_sample_points = 25
    
    # Generate range of x and y values
    mean_range = np.linspace(-6, 6, num_sample_points)
    variance_range = np.linspace(0.1, 8., num_sample_points)
    
    # Track time
    startTime = datetime.now()
    
    # Generate grid coordinates that will be used to evaluate the gaussian on
    mean_nodes, variance_nodes = np.meshgrid(mean_range, variance_range)
    nodes = np.column_stack((mean_nodes.ravel(), variance_nodes.ravel()))
    
    # Compute the function over the grid coordinates and store the values
    h0_rejections = np.zeros(len(nodes), dtype=np.float32)
    avg_samples = np.zeros(len(nodes), dtype=np.float32)
    for i, node in enumerate(nodes):
        r, avg_n = get_es_rejections_acceptances_real(prior_mean = node[0], prior_variance = node[1], iteration = f"{i+1}/{len(nodes)}", C = C, T = T)
        h0_rejections[i] = r
        avg_samples[i] = avg_n
    
    # Print total runtime
    print(f"\ntotal runtime: {datetime.now() - startTime}")
    
    # Convert arrays to DataFrames
    df1 = pd.DataFrame(mean_nodes)
    df2 = pd.DataFrame(variance_nodes)
    df3 = pd.DataFrame(h0_rejections)
    df4 = pd.DataFrame(avg_samples)
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(f'real_ps_ground_truth_{"H0" if EFFECT < 0 else "H1"}_prior_{PRIOR}.xlsx') as writer:
        df1.to_excel(writer, sheet_name='prior_means')
        df2.to_excel(writer, sheet_name='prior_variances')
        df3.to_excel(writer, sheet_name='rejections')
        df4.to_excel(writer, sheet_name='sample')


