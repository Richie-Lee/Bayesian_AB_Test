import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from tabulate import tabulate # pip install tabulate

import warnings
warnings.filterwarnings("ignore")

# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_bayes:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests, prior_odds, T_prior, C_prior, prior_type):
        self.C = C
        self.T = T
        self.k = early_stopping_settings["k"]
        self.results = results
        self.interim_tests = results_interim_tests
        self.prior_odds = prior_odds
        self.C_prior = C_prior
        self.T_prior = T_prior
        self.prior_type = prior_type
        
        # Get true effect from simulated DGP (label)
        self.true_effect = T["true_prob"] - C["true_prob"]
        
        # Always execute main class
        self.get_results() 
    
    def plot_prior(self, T_prior, C_prior, prior_type):
        x = np.linspace(0, 1, 1000)
        
        # Generate pdfs (based on distribution type & parameters)
        if prior_type == "beta":
            C_dist = stats.beta.pdf(x, C_prior["alpha"], C_prior["beta"])
            T_dist = stats.beta.pdf(x, T_prior["alpha"], T_prior["beta"])
            C_label = f"Control: α = {C_prior['alpha']}, β = {C_prior['beta']}, (λ = {C_prior['prior_prob']})"
            T_label = f"Treatment: α = {T_prior['alpha']}, β = {T_prior['beta']}, (λ = {T_prior['prior_prob']})"

        # Plot distributions & means
        plt.plot(x, C_dist, label = C_label, color = _colors[0])
        plt.fill_between(x, C_dist, color = _colors[0], alpha=0.2)
        plt.axvline(C_prior["prior_prob"], color = _colors[0], linestyle = "--", alpha = 0.5)
        
        plt.plot(x, T_dist, label = T_label, color = _colors[1])
        plt.fill_between(x, T_dist, color = _colors[1], alpha=0.2)
        plt.axvline(T_prior["prior_prob"], color = _colors[1], linestyle = "--", alpha = 0.5)
    
        plt.title(f'Prior Distributions ({prior_type})')
        plt.xlabel('Conversion rate')
        # plt.xlim(0, 1)
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()
    
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
    
    def post_prob_over_time(self, interim_tests, prior_odds):
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
            
            plt.plot(x, y_post_prob, linestyle = "-", alpha = 0.3, linewidth = 0.7, color = _colors[0])
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
        
        # Settings
        plt.xlabel('Sample size')
        plt.ylabel('Posterior Probability Sign. Effect')
        plt.title(f'Posterior probability over time ({post_probs.shape[0]} experiments)')
        plt.ylim(0, 1.05)
        plt.legend(loc=(1.04, 0))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def get_results(self):
        self.plot_prior(self.T_prior, self.C_prior, self.prior_type)
        self.plot_early_stopping_dist(self.results, self.k, self.interim_tests)
        self.plot_convergence_distribution(self.results)
        self.post_prob_over_time(self.interim_tests, self.prior_odds)
        