import matplotlib.pyplot as plt
import seaborn as sns


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
        
            
    def get_results(self):
        self.plot_early_stopping_dist(self.results, self.ES, self.IT)
        self.plot_convergence_distribution(self.results)
        