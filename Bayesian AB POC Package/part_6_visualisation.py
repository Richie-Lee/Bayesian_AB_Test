import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tabulate import tabulate # pip install tabulate

import warnings
warnings.filterwarnings("ignore")

# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_bayes:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests, prior_odds):
        self.C = C
        self.T = T
        self.k = early_stopping_settings["k"]
        self.results = results
        self.interim_tests = results_interim_tests
        self.prior_odds = prior_odds
        
        # Get true effect from simulated DGP (label)
        self.true_effect = T["true_prob"] - C["true_prob"]
        
        # Always execute main class
        self.get_results() 
        
    def plot_early_stopping_dist(self, results, k, interim_tests):
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

        # Set the y-axis to log scale
        plt.legend()
        plt.yscale('log')
        plt.xlabel("Sample size")
        plt.ylabel("Bayes_factor")
        plt.title(f"Distributions of early stopping (n = {len(results)})")
        plt.show()
        
    def plot_convergence_distribution(self, results):
        # All observations
        sns.kdeplot(results["sample_size"], label = "All experiments", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()), bw_adjust=0.25)
        plt.xlabel('Sample size')
        plt.title(f"Distributions of experiment termination sample size (n = {len(results)})")
        plt.legend()
        plt.show()
        
        # Separate distributions for Bayes Factor stopping for H1/H0 respectively
        sns.kdeplot(results[results["bayes_factor"] >= 1]["sample_size"], label = f"H1 ({len(results[results['bayes_factor'] >= 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()))
        sns.kdeplot(results[results["bayes_factor"] < 1]["sample_size"], label = f"H0 ({len(results[results['bayes_factor'] < 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()))
        
        plt.xlabel('Sample size')
        plt.legend()
        plt.title(f"Distributions of experiment termination sample size (n = {len(results)})")
        plt.show()
    
    def corr_upift_sample_size(self, results, true_effect):
        plt.scatter(results["sample_size"], results["uplift"], label = "Observed effect")
        plt.axhline(y = true_effect, color = "red", linestyle = "--", linewidth = "0.6", label = "True effect")
        plt.axhline(y = 0, color = "black", linestyle = "--", linewidth = "0.6")
        plt.ylabel("uplift")
        plt.xlabel("Sample size")
        plt.title("correlation uplift & sample size")
        plt.legend()
        plt.show()
    
    def post_prob_over_time(self, interim_tests, prior_odds):
        
        interim_tests_post_prob = []
        
        # Get longest experiment:
        max_n_test = max([len(i) for i in interim_tests])  
        
        # Plot interim testing Bayes-Factor development
        for i in range(len(interim_tests)):
            # get interim test bayes factors and convert them to posterior probabilities
            x, y_bf = zip(*interim_tests[i])
            x = list(x)
            y_post_odds = [prior_odds * bf for bf in y_bf] # add prior odds
            y_post_prob = [round(post_odds / (post_odds + 1), 2) for post_odds in y_post_odds]
            
            # Impute missing values (due to early stop) with ones for H1 and zeros for H0
            while len(y_post_prob) < max_n_test:
                y_post_prob.append(round(y_post_prob[-1])) # assume that it terminated at value closest to the final value (not guarenteed)
                x.append(x[-1] + x[1]) # add new x_value, by adding the iteration step (i = 1) to the last observed n (i = -1)
            
            plt.plot(x, y_post_prob, linestyle = "-", alpha = 0.3, linewidth = 0.7, color = _colors[0])
            interim_tests_post_prob.append(y_post_prob)
        
        post_probs = np.array(interim_tests_post_prob)
        
        # Calculate the average and median for each column
        column_means = np.mean(post_probs, axis=0)
        column_medians = np.median(post_probs, axis=0)
        
        # Calculate quantiles for the confidence interval (e.g., 25th and 75th percentiles)
        q10 = np.percentile(post_probs, 25, axis=0)
        q90 = np.percentile(post_probs, 75, axis=0)
        
        # Plot the results        
        plt.figure(figsize=(10, 6))
        plt.plot(x, column_means, label='Average', color = _colors[0])
        plt.plot(x, column_medians, label='Median', color = _colors[1])
        plt.errorbar(x, column_means, yerr=[column_means - q10, q90 - column_means], fmt='o', label='Confidence Interval (10-90 quantile)', color = _colors[0])
        
        plt.xlabel('Sample size')
        plt.ylabel('P[H1|data]')
        plt.title('Average, Median, and Confidence Interval over time')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        
        
    
    def get_results(self):
        # self.plot_prior(self.T, self.C, self.prior)
        self.plot_early_stopping_dist(self.results, self.k, self.interim_tests)
        self.plot_convergence_distribution(self.results)
        self.corr_upift_sample_size(self.results, self.true_effect)
        self.post_prob_over_time(self.interim_tests, self.prior_odds)
        