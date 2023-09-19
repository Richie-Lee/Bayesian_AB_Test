import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate # pip install tabulate

import warnings
warnings.filterwarnings("ignore")


# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_bayes:
    def __init__(self, T, C, early_stopping_settings, results, results_interim_tests):
        self.C = C
        self.T = T
        self.k = early_stopping_settings["k"]
        self.results = results
        self.interim_tests = results_interim_tests
        
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
        sns.kdeplot(results[results["bayes_factor"] < 1]["sample_size"], label = f"H0 ({len(results[results['bayes_factor'] < 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()))
        sns.kdeplot(results[results["bayes_factor"] >= 1]["sample_size"], label = f"H1 ({len(results[results['bayes_factor'] >= 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()))
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
    
    def early_stop_differences(self, results):
        # Calculate counts based on conditions
        es_reject_fh_accept = len(results[(results["bayes_factor"] > 1) & (results["bayes_factor_fh"] < 1)])
        es_accept_fh_reject = len(results[(results["bayes_factor"] < 1) & (results["bayes_factor_fh"] > 1)])
        both_reject = len(results[(results["bayes_factor"] > 1) & (results["bayes_factor_fh"] > 1)])
        both_accept = len(results[(results["bayes_factor"] < 1) & (results["bayes_factor_fh"] < 1)])
                
        print(f"- ES reject & FH Accept: {es_reject_fh_accept}/{len(results)}\n- ES reject & FH Accept: {es_accept_fh_reject}/{len(results)}\n- Both reject: {both_reject}/{len(results)}\n- Both accept: {both_accept}/{len(results)}")
    
    def get_results(self):
        # self.plot_prior(self.T, self.C, self.prior)
        self.plot_early_stopping_dist(self.results, self.k, self.interim_tests)
        self.plot_convergence_distribution(self.results)
        self.corr_upift_sample_size(self.results, self.true_effect)
        self.early_stop_differences(self.results)
        