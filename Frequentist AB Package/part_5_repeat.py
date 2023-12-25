import pandas as pd
import random
import part_1_dgp as p1_dgp
import part_3_p_values as p3_p
import part_4_inference as p4_metrics
from datetime import datetime

def multiple_iterations(T, C, data_type, test_type, early_stopping_settings, n_test, print_progress, data_config, voi):    
    startTime = datetime.now()
    results = []
    results_columns = ["seed", "sample_size", "uplift", "p_value", "p_value_fh", "alpha"]
    all_interim_tests = []

    for i in range(n_test):
        random.seed(i)

        if print_progress:
            print(f"{i + 1}/{n_test}")

        # Part 1: Generate data
        if data_type == "binary":
            # Bernoulli distributed Binary Data (Conversions)
            C["sample"], C["converted"], C["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=C["true_prob"], n=C["n"])
            T["sample"], T["converted"], T["sample_conversion_rate"] = p1_dgp.get_bernoulli_sample(mean=T["true_prob"], n=T["n"])
        elif data_type == "continuous":
            # Continuous data
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
        
        # Part 3: p-values
        p_value_calculator = p3_p.get_p_value(T, C, early_stopping_settings, test_type)
        p_value_fh, p_value_es, interim_tests, sample_size, alpha = p_value_calculator.get_values()

        # Part 4: Inference
        metrics_calculator = p4_metrics.get_metrics(T, C, data_type)
        metrics = metrics_calculator.get_values()
        
        results.append([i, sample_size, metrics["uplift"], p_value_es, p_value_fh, alpha])
        all_interim_tests.append(interim_tests)
    
    results = pd.DataFrame(results, columns=results_columns)
    print(f"\ntotal runtime: {datetime.now() - startTime}")
    
    return results, all_interim_tests 
    