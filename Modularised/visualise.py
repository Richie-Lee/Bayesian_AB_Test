import matplotlib.pyplot as plt
import seaborn as sns

# Set the style & colors for the plots
sns.set_style('darkgrid')
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class visualisation:
    def __init__(self, C, T, early_stopping, results, results_interim_tests):
        self.C = C
        self.T = T
        self.ES = early_stopping
        self.results = results
        self.IT = results_interim_tests
        
        # Always execute main class
        self.get_results() 
        
    def plot_early_stopping_dist(self, results, ES, IT):
        # plot stopping criteria
        plt.axhline(y = 1, color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = ES["k"], color = "black", linestyle = "--", linewidth = "0.6")
        plt.axhline(y = 1/ES["k"], color = "black", linestyle = "--", linewidth = "0.6")
            
        # Plot interim testing Bayes-Factor development
        for i in range(len(IT)):
            x, y = zip(*IT[i])
            if results["bayes_factor"][i] > 1: # Reject H0 (Effect discovery)
                color_i = "green"
            else: # Accept H0 (No effect)
                color_i = "red"
            plt.plot(x, y, linestyle = "-", alpha = 0.5, linewidth = 0.7, color = color_i)
        
        # Set the y-axis to log scale
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
        
    def plot_treatment_effect(self, results, T, C):  
        # Get true effect from simulated DGP (label)
        true_effect = T["true_prob"] - C["true_prob"]
        
        # Get distribution of all treatment effect estimation errors
        plt.hist(results["treatment_effect"], bins = 20, alpha = 0.5, density=True, color = "green")
        sns.kdeplot(results["treatment_effect"], label = "Estimated", fill = True, alpha = 0.3, color = "green")
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
                
            
    def get_results(self):
        self.plot_early_stopping_dist(self.results, self.ES, self.IT)
        self.plot_convergence_distribution(self.results)
        self.plot_prob_distributions(self.results)
        self.plot_treatment_effect(self.results, self.T, self.C)
        