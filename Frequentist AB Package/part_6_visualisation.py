import matplotlib.pyplot as plt
import numpy as np
import warnings
import part_3_p_values as p3_p

import pandas as pd

warnings.filterwarnings("ignore")
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_frequentist:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests, test_type, data_type, data_config=None):
        self.C = C
        self.T = T
        self.early_stopping_settings = early_stopping_settings
        self.results = results
        self.interim_tests = results_interim_tests
        self.test_type = test_type
        self.data_type = data_type
        self.data_config = data_config
        
    @staticmethod
    def get_obrien_fleming_alphas(max_n_test, early_stopping_settings):
        # Create pairs [i, n_i] for i'th test and the associated n (K = total nr of tests)
        K = max_n_test
        n_k = [early_stopping_settings["interim_test_interval"] * k for k in range(1, K + 1)]
        alpha_k = [p3_p.get_p_value.calculate_adjusted_alpha_one_sided(k, K, early_stopping_settings["alpha"]) for k in range(1, K + 1)]
        ob_alphas = list(zip(n_k, alpha_k))         
        return ob_alphas

    def plot_early_stopping_dist(self, results, interim_tests, early_stopping_settings, T, C, test_type):
        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])
        
        # Draw green lines for H1 (reject, identify effect) and red for H0 (accenpt, no effect / negative effect)
        h1_count, h0_count = 0, 0
        for i in range(len(interim_tests)):
            x, p_values = zip(*interim_tests[i])
            if results["p_value"][i] < self.results["alpha"][i]:
                color_i = "green"
                h1_count += 1
            else:
                color_i = "red"
                h0_count += 1
            plt.plot(x, p_values, linestyle = "-", alpha = 0.5, linewidth = 0.7, color = color_i)
        
        plt.scatter(0, 0, marker = ".", color = "green", label = f"H1: {h1_count}/{len(results)}")
        plt.scatter(0, 0, marker = ".", color = "red", label = f"H0: {h0_count}/{len(results)}")

        plt.axvline(x = early_stopping_settings["minimum_sample"], color = "grey", label = f"Minimum sample: {early_stopping_settings['minimum_sample']}")

        # Alpha / Rejection line 
        if test_type in ["naive t-test", "always valid inference", "always valid inference one-sided"]:
            plt.axhline(y=early_stopping_settings["alpha"], color = "black", label = f"alpha = {early_stopping_settings['alpha']}")
        elif test_type == "alpha spending":
            ob_alphas = self.get_obrien_fleming_alphas(max_n_test, early_stopping_settings)
            x, y = zip(*ob_alphas) # unpack zip to plot OB adjusted alpha
            plt.plot(x, y, color = "black", label = f"OB adjusted alpha (K = {len(x)})")

        plt.legend(loc=(1.02, 0))
        plt.xlabel("Sample size")
        plt.ylabel("P-value")
        plt.ylim(0, 1)  # p-value range 0 to 1
        plt.title(f"p-values over time with {test_type.upper()} ({len(results)} runs)")
        plt.show()

    def p_value_over_time(self, interim_tests, early_stopping_settings, test_type, T, C):
        # Initialise
        interim_tests_p_values = []
    
        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])
    
        # Collect p-values
        for i in range(len(interim_tests)):
            x, p_values_tuple = zip(*interim_tests[i])
            x = list(x)
            p_values = list(p_values_tuple)  # Convert tuple to list
    
            # Impute missing values (due to early stop) with the last observed p-value
            while len(p_values) < max_n_test:
                p_values.append(p_values[-1])
                x.append(x[-1] + self.early_stopping_settings["interim_test_interval"])
    
            # plt.plot(x, p_values, linestyle="-", alpha=0.3, linewidth=0.7, color=_colors[0]) # the lines that are used to create the confidence intervals
            interim_tests_p_values.append(p_values)

        # Convert to array for statistical processing
        p_values_array = np.array(interim_tests_p_values)

        # Calculate mean and median for each interim test
        column_means = np.mean(p_values_array, axis=0)
        column_medians = np.median(p_values_array, axis=0)

        # Calculate quantiles for the confidence interval
        quantile_lb, quantile_ub = 10, 90
        lb = np.percentile(p_values_array, quantile_lb, axis=0)
        ub = np.percentile(p_values_array, quantile_ub, axis=0)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(x, column_medians, label='Median (q10-q90 interval)', color=_colors[0])
        plt.errorbar(x, column_medians, yerr=[column_medians - lb, ub - column_medians], fmt='o', color=_colors[0], alpha=0.5)
        plt.plot(x, column_means, label='Mean', color=_colors[1])

        # Minimum sample line
        plt.axvline(x=early_stopping_settings["minimum_sample"], color="grey", label=f"Minimum sample: {early_stopping_settings['minimum_sample']}")

        # Alpha / Rejection line 
        if test_type in ["naive t-test", "always valid inference", "always valid inference one-sided"]:
            plt.axhline(y=early_stopping_settings["alpha"], color = "black", label = f"alpha = {early_stopping_settings['alpha']}")
        elif test_type == "alpha spending":
            ob_alphas = self.get_obrien_fleming_alphas(max_n_test, early_stopping_settings)
            k, adjusted_alpha = zip(*ob_alphas) # unpack zip to plot OB adjusted alpha
            plt.plot(k, adjusted_alpha, color = "black", label = f"OB adjusted alpha (K = {len(x)})")
        
        # Settings
        plt.xlabel('Sample size')
        plt.ylabel('P-value')
        plt.title('P-value over time')
        plt.ylim(0, 1)  # Adjust as needed
        plt.legend(loc=(1.02, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title(f"p-value distribution over time with {test_type.upper()} ({len(interim_tests)} runs)")
        plt.show()
    
    def power_curve(self, interim_tests, early_stopping_settings, test_type, T, C, data_type):  
        # Initialise
        interim_tests_p_values = []

        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])
        print(max_n_test)

        # Collect p-values
        for i in range(len(interim_tests)):
            x, p_values_tuple = zip(*interim_tests[i])
            x = list(x)
            p_values = list(p_values_tuple)  # Convert tuple to list

            # Impute missing values (due to early stop) with NaN value
            while len(p_values) < max_n_test:
                p_values.append(np.nan)
                x.append(x[-1] + self.early_stopping_settings["interim_test_interval"])

            # plt.plot(x, p_values, linestyle="-", alpha=0.3, linewidth=0.7, color=_colors[0]) # the lines that are used to create the confidence intervals
            interim_tests_p_values.append(p_values)

        # Convert to array for statistical processing
        p_values_array = np.array(interim_tests_p_values)
        
        # Initialize an empty array with the same shape as p_values_array
        binary_array = np.zeros_like(p_values_array, dtype=int)

        # Iterate over each row (each experiment run)
        for i in range(p_values_array.shape[0]):
            row = p_values_array[i]

            # Find the last non-NaN value in the row
            last_non_nan_index = np.where(~np.isnan(row))[0][-1]

            # Set the corresponding value to 1 if it's below alpha_last
            if row[last_non_nan_index] < 0.05: # ---------------------------------------:
                binary_array[i, last_non_nan_index] = 1

            # Set all subsequent values (NaNs) to 1
            binary_array[i, last_non_nan_index + 1:] = 1
        
        rejection_counts = list(np.sum(binary_array, axis=0))
        ratio_rejected = [x / len(interim_tests) for x in rejection_counts] # get ratio of experiments that terminated at each evaluation time
        
        # sample_size for each evaluation k
        sample_sizes_k = [early_stopping_settings["interim_test_interval"] * k for k in range(0, max_n_test)] # start range at 0 for plotting reasons 
        
        # Plot empirical type-I error if H0 = TRUE, power curve otherwise
        if data_type == "binary":
            h0 = True if C["true_prob"] >= T["true_prob"] else False
        elif data_type == "continuous":
            h0 = True if C["true_mean"] >= T["true_mean"] else False
        elif data_type == "real":
            h0 = True if C["true_mean"] >= T["true_mean"] else False
        
        # Plot critical values: (adjusted) alphas
        if h0 == True:
            plt.plot(sample_sizes_k, ratio_rejected, label = f"Type-I error ({ratio_rejected[-1]})", color = "red")
            if test_type in ["naive t-test", "always valid inference", "always valid inference one-sided"]:
                plt.axhline(y=early_stopping_settings["alpha"], color = "black", label = f"alpha = {early_stopping_settings['alpha']}")
            elif test_type == "alpha spending":
                ob_alphas = self.get_obrien_fleming_alphas(max_n_test, early_stopping_settings)
                k, adjusted_alpha = zip(*ob_alphas) # unpack zip to plot OB adjusted alpha
                plt.plot(k, adjusted_alpha, color = "black", label = f"OB adjusted alpha (K = {len(x)})")      
            plt.title(f'Type-I error over time - {test_type.upper()}')
        else:
            plt.plot(sample_sizes_k, ratio_rejected, label = f"Ratio rejected ({ratio_rejected[-1]})", color = "green")
            plt.title(f'Power over time - {test_type.upper()}')
        
        # Minimum sample line
        plt.axvline(x=early_stopping_settings["minimum_sample"], color="grey", label=f"Minimum sample: {early_stopping_settings['minimum_sample']}")

        
        plt.xlabel('Sample size')
        plt.ylabel('Ratio H0 rejected')
        plt.ylim(0, 1)  # Adjust as needed
        plt.legend(loc=(1.02, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
    
        return pd.DataFrame(list(zip(sample_sizes_k, ratio_rejected)), columns = ["sample", f"bayesian {test_type}"])
    
    def get_results(self):
        self.plot_early_stopping_dist(self.results, self.interim_tests, self.early_stopping_settings, self.T, self.C, self.test_type)
        self.p_value_over_time(self.interim_tests, self.early_stopping_settings, self.test_type, self.T, self.C)
        power_curve_results = self.power_curve(self.interim_tests, self.early_stopping_settings, self.test_type, self.T, self.C, self.data_type)
        
        return power_curve_results