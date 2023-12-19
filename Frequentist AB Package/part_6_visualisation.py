import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_frequentist:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests):
        self.C = C
        self.T = T
        self.early_stopping_settings = early_stopping_settings
        self.results = results
        self.interim_tests = results_interim_tests

    def plot_early_stopping_dist(self, results, interim_tests, early_stopping_settings):
        plt.axhline(y = early_stopping_settings["alpha"], color = "black", linestyle = "--", linewidth = "0.6", label = f"Significance level: {early_stopping_settings['alpha']}")

        h1_count, h0_count = 0, 0
        for i in range(len(interim_tests)):
            x, p_values = zip(*interim_tests[i])
            if results["p_value"][i] < self.early_stopping_settings["alpha"]:
                color_i = "green"
                h1_count += 1
            else:
                color_i = "red"
                h0_count += 1
            plt.plot(x, p_values, linestyle = "-", alpha = 0.5, linewidth = 0.7, color = color_i)
        
        plt.scatter(0, 0, marker = ".", color = "green", label = f"H1: {h1_count}/{len(results)}")
        plt.scatter(0, 0, marker = ".", color = "red", label = f"H0: {h0_count}/{len(results)}")

        plt.axvline(x = early_stopping_settings["minimum_sample"], color = "grey", label = f"Minimum sample: {early_stopping_settings['minimum_sample']}")

        plt.legend()
        plt.xlabel("Sample size")
        plt.ylabel("P-value")
        plt.ylim(0, 1)  # p-value range 0 to 1
        plt.title(f"Distributions of early stopping (n = {len(results)}, alpha = {self.early_stopping_settings['alpha']})")
        plt.show()

    def p_value_over_time(self, interim_tests, early_stopping_settings):
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
    
            plt.plot(x, p_values, linestyle="-", alpha=0.3, linewidth=0.7, color=_colors[0])
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

        # Settings
        plt.xlabel('Sample size')
        plt.ylabel('P-value')
        plt.title('P-value over time')
        plt.ylim(0, 1)  # Adjust as needed
        plt.legend(loc=(1.04, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def get_results(self):
        self.plot_early_stopping_dist(self.results, self.interim_tests, self.early_stopping_settings)
        self.p_value_over_time(self.interim_tests, self.early_stopping_settings)
