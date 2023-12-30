import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import scipy.stats as stats
import pandas as pd
# from tabulate import tabulate # pip install tabulate

import warnings
warnings.filterwarnings("ignore")

# Set the style & colors for the plots
# sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_bayes:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests, prior_odds, prior_type, T_prior=None, C_prior=None, H0_prior=None, H1_prior=None):
        self.C = C
        self.T = T
        self.k = early_stopping_settings["k"]
        self.min_sample = early_stopping_settings["minimum_sample"]
        self.results = results
        self.interim_tests = results_interim_tests
        self.prior_odds = prior_odds
        self.C_prior = C_prior
        self.T_prior = T_prior
        self.H0_prior = H0_prior
        self.H1_prior = H1_prior
        self.prior_type = prior_type
        self.early_stopping_settings = early_stopping_settings
        # Get true effect from simulated DGP (label)
        self.true_effect = T["true_prob"] - C["true_prob"] if self.prior_type == "beta" else None
        self.get_results()
    
    def plot_prior(self):
        if self.prior_type == "beta":
            x = np.linspace(0, 1, 1000)
            C_dist = stats.beta.pdf(x, self.C_prior["alpha"], self.C_prior["beta"])
            T_dist = stats.beta.pdf(x, self.T_prior["alpha"], self.T_prior["beta"])
            C_label = f"Control: α = {self.C_prior['alpha']}, β = {self.C_prior['beta']}"
            T_label = f"Treatment: α = {self.T_prior['alpha']}, β = {self.T_prior['beta']}"
            # Plot distributions & means
            plt.plot(x, C_dist, label=C_label, color=_colors[0])
            plt.fill_between(x, C_dist, color=_colors[0], alpha=0.2)
            plt.plot(x, T_dist, label=T_label, color=_colors[1])
            plt.fill_between(x, T_dist, color=_colors[1], alpha=0.2)
        elif self.prior_type == "normal":
            x_min = min(self.H0_prior["mean"] - 2 * np.sqrt(self.H0_prior["variance"]), self.H1_prior["mean"] - 2 * np.sqrt(self.H1_prior["variance"]))
            x_max = max(self.H0_prior["mean"] + 2 * np.sqrt(self.H0_prior["variance"]), self.H1_prior["mean"] + 2 * np.sqrt(self.H1_prior["variance"]))
            x = np.linspace(x_min, x_max, 1000)
            H0_dist = stats.norm.pdf(x, self.H0_prior["mean"], self.H0_prior["variance"])
            H1_dist = stats.norm.pdf(x, self.H1_prior["mean"], self.H1_prior["variance"])
            H0_label = f"H0: N({self.H0_prior['mean']}, {self.H0_prior['variance']})"
            H1_label = f"H0: N({self.H1_prior['mean']}, {self.H1_prior['variance']})"
            # Plot distributions & means
            plt.plot(x, H0_dist, label=H0_label, color=_colors[0])
            plt.fill_between(x, H0_dist, color=_colors[0], alpha=0.2)
            plt.plot(x, H1_dist, label=H1_label, color=_colors[1])
            plt.fill_between(x, H1_dist, color=_colors[1], alpha=0.2)

        plt.title(f'Prior Distributions ({self.prior_type})')
        plt.xlabel('Conversion rate' if self.prior_type == "beta" else 'Value')
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()
            
    def plot_early_stopping_dist(self, results, k, interim_tests, min_sample):
        # plot stopping criteria
        plt.axhline(y = 1, color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = k, color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = 1/k, color = "black", linestyle = "--", linewidth = "0.6")
            
        # Plot interim testing Bayes-Factor development
        h1_count, h0_count = 0, 0
        for i in range(len(interim_tests)):
            x, y = zip(*interim_tests[i])
            if results["bayes_factor"][i] > 1: # Reject H0 (Effect discovery)
                color_i = "green"
                h1_count += 1
            else: # Accept H0 (No effect)
                color_i = "red"
                h0_count += 1
            plt.plot(x, y, linestyle = "-", alpha = 0.5, linewidth = 0.7, color = color_i)
        
        plt.scatter(0, 0, marker = ".", color = "green", label = f"H1: {h1_count}/{len(results)}")
        plt.scatter(0, 0, marker = ".", color = "red", label = f"H0: {h0_count}/{len(results)}")
        
        # minimum sample
        plt.axvline(x = min_sample, color = "grey", label = f"Minimum sample: {min_sample}")
        
        # Set the y-axis to log scale
        plt.legend()
        plt.yscale('log')
        plt.xlabel("Sample size")
        plt.ylabel("Bayes_factor")
        plt.ylim(1/(1.2*k), k*1.2) # symmetrical vertical range displayed
        plt.title(f"Distributions of early stopping (n = {len(results)}, k = {k})")
        plt.show()
    
    def post_prob_over_time(self, interim_tests, prior_odds, min_sample):
        # Initialise
        interim_tests_post_prob = []
        
        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])  
        
        # Transform interim tests BFs to posterior probabilities
        for i in range(len(interim_tests)):
            # gget BFs
            x, y_bf = zip(*interim_tests[i])
            x = list(x)
            y_post_odds = [prior_odds * bf for bf in y_bf] # add prior odds
            y_post_prob = [round(post_odds / (post_odds + 1), 2) for post_odds in y_post_odds]
            
            # Impute missing values (due to early stop) with ones for H1 and zeros for H0
            while len(y_post_prob) < max_n_test:
                y_post_prob.append(round(y_post_prob[-1])) # assume that it terminated at value closest to the final value (not guarenteed)
                x.append(x[-1] + x[1]) # add new x_value (sample size), by adding the iteration step (i = 1) to the last observed n (i = -1)
            
            # plt.plot(x, y_post_prob, linestyle = "-", alpha = 0.3, linewidth = 0.7, color = _colors[0])
            interim_tests_post_prob.append(y_post_prob)
        
        # Collect all interim test posterior probabilities & format for post processing
        post_probs = np.array(interim_tests_post_prob)
        
        # Calculate the average and median for each column (interim test sample size)
        column_means = np.mean(post_probs, axis=0)
        column_medians = np.median(post_probs, axis=0)
        
        # Calculate quantiles for the confidence interval (e.g., 25th and 75th percentiles)
        quantile_lb, quantile_ub = 10, 90
        lb = np.percentile(post_probs, quantile_lb, axis=0)
        ub = np.percentile(post_probs, quantile_ub, axis=0)

        # Plot the results        
        plt.figure(figsize=(10, 6))
        plt.plot(x, column_medians, label = f'Median (q{quantile_lb}-q{quantile_ub} interval)', color = _colors[0]) # Median
        plt.errorbar(x, column_medians, yerr=[column_medians - lb, ub - column_medians], fmt='o', color = _colors[0], alpha = 0.5) # Confidence interval
        plt.plot(x, column_means, label = 'Mean', color = _colors[1]) # Mean
        
        # minimum sample
        plt.axvline(x = min_sample, color = "grey", label = f"Minimum sample: {min_sample}")
        
        # Settings
        plt.xlabel('Sample size')
        plt.ylabel('Posterior Probability Sign. Effect')
        plt.title(f'Posterior probability over time ({post_probs.shape[0]} experiments)')
        plt.ylim(0, 1.05)
        plt.legend(loc=(1.04, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def power_curve(self, interim_tests, early_stopping_settings, T, C, prior_type):  
        # Initialise
        interim_tests_bf = []

        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])

        # Collect p-values
        for i in range(len(interim_tests)):
            x, bf_tuple = zip(*interim_tests[i])
            x = list(x)
            bfs = list(bf_tuple)  # Convert tuple to list

            # Impute missing values (due to early stop) with NaN value
            while len(bfs) < max_n_test:
                bfs.append(np.nan)
                x.append(x[-1] + self.early_stopping_settings["interim_test_interval"])

            # plt.plot(x, p_values, linestyle="-", alpha=0.3, linewidth=0.7, color=_colors[0]) # the lines that are used to create the confidence intervals
            interim_tests_bf.append(bfs)

        # Convert to array for statistical processing
        bf_array = np.array(interim_tests_bf)
        
        # Initialize an empty array with the same shape as p_values_array
        binary_array = np.zeros_like(bf_array, dtype=int)

        # Iterate over each row (each experiment run)
        for i in range(bf_array.shape[0]):
            row = bf_array[i]

            # Find the last non-NaN value in the row
            last_non_nan_index = np.where(~np.isnan(row))[0][-1]

            # Set the corresponding value to 1 (reject) if last bf is > 1
            if row[last_non_nan_index] > 1: # ---------------------------------------:
                binary_array[i, last_non_nan_index] = 1

                # Set all subsequent values (NaNs) to 1
                binary_array[i, last_non_nan_index + 1:] = 1
            # else accept, if last bf is < 1 (note Bayesian early stops both sides)   
            else:
                # Set all subsequent values (NaNs) to 1
                binary_array[i, last_non_nan_index + 1:] = 0
        
        rejection_counts = list(np.sum(binary_array, axis=0))
        ratio_rejected = [x / len(interim_tests) for x in rejection_counts] # get ratio of experiments that terminated at each evaluation time
        
        # sample_size for each evaluation k
        sample_sizes_k = [early_stopping_settings["interim_test_interval"] * k for k in range(0, max_n_test)] # start range at 0 for plotting reasons 
        
        # Plot empirical type-I error if H0 = TRUE, power curve otherwise
        if prior_type == "beta":
            h0 = True if C["true_prob"] == T["true_prob"] else False
        elif prior_type == "normal":
            h0 = True if C["true_mean"] > T["true_mean"] else False
        
        if h0 == True:
            plt.plot(sample_sizes_k, ratio_rejected, label = f"H0 rejected: {ratio_rejected[-1]} \n({ratio_rejected[-2]} excl inconclusive)", color = "red")      
            plt.title(f"Type-I error over time - {prior_type.upper()} prior (k = {early_stopping_settings['k']})")
        else:
            plt.plot(sample_sizes_k, ratio_rejected, label = f"H0 rejected: {ratio_rejected[-1]} \n({ratio_rejected[-2]} excl inconclusive)", color = "green")
            plt.title(f"Power over time - {prior_type.upper()} prior (k = {early_stopping_settings['k']})")
        
        # Minimum sample line
        plt.axvline(x=early_stopping_settings["minimum_sample"], color="grey", label=f"Minimum sample: {early_stopping_settings['minimum_sample']}")

        plt.xlabel('Sample size')
        plt.ylabel('Ratio correct classifications')
        plt.ylim(0, 1)  # Adjust as needed
        plt.legend(loc=(1.02, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        return pd.DataFrame(list(zip(sample_sizes_k, ratio_rejected)), columns = ["sample", f"bayesian {prior_type}"])
    
    def get_results(self):
        # self.plot_prior()
        self.plot_early_stopping_dist(self.results, self.k, self.interim_tests, self.min_sample)
        # self.plot_convergence_distribution(self.results) # Uncomment if needed
        self.post_prob_over_time(self.interim_tests, self.prior_odds, self.min_sample)
        power_curve_values = self.power_curve(self.interim_tests, self.early_stopping_settings, self.T, self.C, self.prior_type)
        
        return power_curve_values