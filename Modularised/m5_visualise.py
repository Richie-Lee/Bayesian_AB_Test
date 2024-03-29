import matplotlib.pyplot as plt
import seaborn as sns

# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation_bayes:
    def __init__(self, C, T, prior, early_stopping, results, results_interim_tests):
        self.C = C
        self.T = T
        self.prior = prior
        self.ES = early_stopping
        self.results = results
        self.IT = results_interim_tests
        
        # Get true effect from simulated DGP (label)
        self.true_effect = T["true_prob"] - C["true_prob"]
        
        # Always execute main class
        self.get_results() 
        
    def plot_prior(self, T, C, prior):
        # Plot the histogram + kernel (Posterior)
        plt.hist(C["prior_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[0])
        plt.hist(T["prior_sample"], bins = 30, alpha = 0.5, density=True, color = _colors[1])
        sns.kdeplot(C["prior_sample"], label='Control', fill = False, color = _colors[0])
        sns.kdeplot(T["prior_sample"], label='Treatment', fill = False, color = _colors[1])
        plt.axvline(x = T["true_prob"], color = "black", label = "True post-treatment")
        plt.xlabel('Probability')
        plt.legend()
        plt.title(f"Samples from prior distributions ({prior['distribution']})")
        plt.show()
        
    def plot_early_stopping_dist(self, results, ES, IT):
        # plot stopping criteria
        plt.axhline(y = 1, color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = ES["k"], color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = 1/ES["k"], color = "black", linestyle = "--", linewidth = "0.6")
            
        # Plot interim testing Bayes-Factor development
        h1_count, h0_count = 0, 0
        for i in range(len(IT)):
            x, y = zip(*IT[i])
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
        sns.kdeplot(results[results["bayes_factor"] >= 1]["sample_size"], label = f"H1 ({len(results[results['bayes_factor'] >= 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()), bw_adjust=0.25)
        sns.kdeplot(results[results["bayes_factor"] < 1]["sample_size"], label = f"H0 ({len(results[results['bayes_factor'] < 1])})", fill = True, alpha = 0.5, clip = (0, results["sample_size"].max()), bw_adjust=0.25)
        plt.xlabel('Sample size')
        plt.legend()
        plt.title(f"Distributions of experiment termination sample size (n = {len(results)})")
        plt.show()
        
    def plot_prob_distributions(self, results):    
        # Plot relevant probability distributions
        sns.kdeplot(results["P[TE>mde]"], label = "post P[TE > mde]", fill = True, alpha = 0.5, clip=(0, 1))
        sns.kdeplot(results["P[T>C]"], label = "post P[T > C]", fill = True, alpha = 0.5, clip=(0, 1))
        sns.kdeplot(results["prob_H1"], label = "P[H1|BF]", fill = True, alpha = 0.5, clip=(0, 1))
        plt.legend()
        plt.xlabel('Probability')
        plt.title(f"Distributions of predicted probabilities (n = {len(results)})")
        plt.show()
        
    def plot_treatment_effect(self, results, true_effect):  
        # Get distribution of all treatment effect estimation errors
        plt.hist(results["treatment_effect"], bins = 20, alpha = 0.5, density=True, color = _colors[0])
        sns.kdeplot(results["treatment_effect"], label = "Estimated", fill = True, alpha = 0.3, color = _colors[0])
        plt.axvline(x = true_effect, color = "black", label = "true TE")
        
        plt.legend()
        plt.xlabel('Treatment effect')
        plt.title(f"Distributions of treatment effect estimation (n = {len(results)})")
        plt.show()
        
        # Compare Fixed Horizon & Early stopping
        n = results['sample_size'].max() # Assume at least one non-early stop for coding convenience
        results_fh = results[results["sample_size"] == n]
        results_es = results[results["sample_size"] != n]
    
        sns.kdeplot(results_fh["treatment_effect"], label = f"Fixed Horizon (n = {len(results_fh.index)})", fill = True, alpha = 0.3, bw_adjust=0.4)
        sns.kdeplot(results_es["treatment_effect"], label = f"Early Stopping (n = {len(results_es.index)})", fill = True, alpha = 0.3, bw_adjust=0.4)
        plt.axvline(x = true_effect, color = "black", label = "true TE")
        plt.legend()
        plt.xlabel('Treatment effect')
        plt.title(f"Distributions of treatment effect estimation (n = {len(results)})")
        plt.show()
    
    def plot_corr_bias_sample_size(self, results, true_effect):
        # Plot scatter: x = sample size, y = TE estimation bias
        errors = [x - true_effect for x in results["treatment_effect"]]
        plt.scatter(results["sample_size"], errors, alpha = 0.6, label = "bias TE estimation")
        plt.axhline(y = 0, linewidth = 0.6, linestyle = "--", color = "black")
        plt.legend()
        plt.ylabel("Error")
        plt.xlabel("Sample size")
        
        # symmetric range for better illustration correlation
        largest_error = max(-min(errors), max(errors))
        plt.ylim(-largest_error*1.2, largest_error*1.2)
        plt.title(f"Distributions of treatment effect estimation (n = {len(results)})")
        plt.show()
            
    def get_results(self):
        self.plot_prior(self.T, self.C, self.prior)
        self.plot_early_stopping_dist(self.results, self.ES, self.IT)
        self.plot_convergence_distribution(self.results)
        self.plot_prob_distributions(self.results)
        self.plot_treatment_effect(self.results, self.true_effect)
        self.plot_corr_bias_sample_size(self.results, self.true_effect)
        