import pandas as pd
import random
import part_1_dgp as p1_dgp
import part_2_prior as p2_prior
import part_3_bayes_factors as p3_bf
import part_4_inference as p4_metrics
from datetime import datetime

def multiple_iterations(T, C, prior_odds, prior_type, prior_parameters, early_stopping_settings, n_test, print_progress, data_type, data_config, simulated_treatment_effect, voi):    
    startTime = datetime.now()
    results = []
    results_columns = ["seed", "sample_size", "P[H1|data]", "uplift", "P[T>C]", "loss", "bayes_factor", "bayes_factor_fh"]
    all_interim_tests = []

    for i in range(n_test):
        random.seed(i)

        if print_progress:
            print(f"{i + 1}/{n_test}")

        # Part 1: Generate data
        if data_type == "beta":
            C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
            T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
        elif data_type == "normal":
            C["sample"] = p1_dgp.get_normal_sample(mean=C["true_mean"], variance=C["true_variance"], n=C["n"])
            T["sample"] = p1_dgp.get_normal_sample(mean=T["true_mean"], variance=T["true_variance"], n=T["n"])
        elif data_type == "real":
            # Real data already has been imported and preprocessed - avoid repeating this -> monte carlo sampling            
            # Random sample "n" observations from control/treatment to replicate monte carlo simulation on real data
            if data_config["n"] >= min(len(C["df"]), len(T["df"])):
                raise Exception(f"Unable to do monte carlo throughs simulation with requested sample size {data_config['n']} given data size (for control/treatment) {len(T['df'])}")
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
            

            
        # Part 2: Prior
        prior_calculator = p2_prior.get_prior(prior_type, prior_parameters[prior_type])
        if prior_type == "beta":
            C_prior, T_prior = prior_calculator.get_values()
        elif prior_type == "normal":
            H0_prior, H1_prior = prior_calculator.get_values()
        
        # Part 3: Bayes Factor
        if prior_type == "beta":
            bf_calculator = p3_bf.get_bayes_factor(T, C, prior_type, early_stopping_settings, T_prior=T_prior, C_prior=C_prior)
        elif prior_type == "normal":
            bf_calculator = p3_bf.get_bayes_factor(T, C, prior_type, early_stopping_settings, H0_prior=H0_prior, H1_prior=H1_prior)
        bf_fh, bf, interim_tests, k, sample_size = bf_calculator.get_values()
        
        # Part 4: Inference
        if prior_type == "beta":
            metrics_calculator = p4_metrics.get_metrics(T, C, prior_type, bf, prior_odds, T_prior=T_prior, C_prior=C_prior)
        elif prior_type == "normal":
            metrics_calculator = p4_metrics.get_metrics(T, C, prior_type, bf, prior_odds, H0_prior=H0_prior, H1_prior=H1_prior)
        metrics = metrics_calculator.get_values()
        
        results.append([i, sample_size, metrics["P[H1|data]"], metrics["uplift"], metrics.get("P[T>C]", None), metrics["loss"], bf, bf_fh])
        all_interim_tests.append(interim_tests)
    
    results = pd.DataFrame(results, columns=results_columns)
    print(f"\ntotal runtime: {datetime.now() - startTime}")
    
    return results, all_interim_tests
