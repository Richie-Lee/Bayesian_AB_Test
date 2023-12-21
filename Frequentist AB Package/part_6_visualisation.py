import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import part_3_p_values as p3_p

warnings.filterwarnings("ignore")
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_frequentist:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests, test_type):
        self.C = C
        self.T = T
        self.early_stopping_settings = early_stopping_settings
        self.results = results
        self.interim_tests = results_interim_tests
        self.test_type = test_type
        
    @staticmethod
    def get_obrien_fleming_alphas(T, C, early_stopping_settings):
        # Create pairs [i, n_i] for i'th test and the associated n (K = total nr of tests)
        K = math.floor(T["n"] / early_stopping_settings["interim_test_interval"])
        n_k = [early_stopping_settings["interim_test_interval"] * k for k in range(1, K + 1)]
        alpha_k = [p3_p.get_p_value.calculate_adjusted_alpha_one_sided(k, K, early_stopping_settings["alpha"]) for k in range(1, K + 1)]
        ob_alphas = list(zip(n_k, alpha_k))         
        return ob_alphas

    def plot_early_stopping_dist(self, results, interim_tests, early_stopping_settings, T, C, test_type):
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
        if test_type == "naive t-test":
            plt.axhline(y=early_stopping_settings["alpha"], color = "black", label = f"alpha = {early_stopping_settings['alpha']}")
        elif test_type == "alpha spending":
            ob_alphas = self.get_obrien_fleming_alphas(T, C, early_stopping_settings)
            x, y = zip(*ob_alphas) # unpack zip to plot OB adjusted alpha
            plt.plot(x, y, color = "black", label = f"OB adjusted alpha (K = {len(x)})")

        plt.legend(loc=(1.02, 0))
        plt.xlabel("Sample size")
        plt.ylabel("P-value")
        plt.ylim(0, 1)  # p-value range 0 to 1
        plt.title(f"p-values over time with {test_type} ({len(results)} runs)")
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
        if test_type == "naive t-test":
            plt.axhline(y=early_stopping_settings["alpha"], color = "black", label = f"alpha = {early_stopping_settings['alpha']}")
        elif test_type == "alpha spending":
            ob_alphas = self.get_obrien_fleming_alphas(T, C, early_stopping_settings)
            x, y = zip(*ob_alphas) # unpack zip to plot OB adjusted alpha
            plt.plot(x, y, color = "black", label = f"OB adjusted alpha (K = {len(x)})")
        
        # Settings
        plt.xlabel('Sample size')
        plt.ylabel('P-value')
        plt.title('P-value over time')
        plt.ylim(0, 1)  # Adjust as needed
        plt.legend(loc=(1.02, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title(f"p-value distribution over time with {test_type} ({len(interim_tests)} runs)")
        plt.show()

    def get_results(self):
        self.plot_early_stopping_dist(self.results, self.interim_tests, self.early_stopping_settings, self.T, self.C, self.test_type)
        self.p_value_over_time(self.interim_tests, self.early_stopping_settings, self.test_type, self.T, self.C)
