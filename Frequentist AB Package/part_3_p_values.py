from scipy import stats
import math
import numpy as np

class get_p_value():
    def __init__(self, T, C, early_stopping_settings, test_type):
        # Initialize with treatment and control samples, early stopping settings, and test type
        self.T = T  # Treatment group sample
        self.C = C  # Control group sample
        self.early_stopping_settings = early_stopping_settings  # Settings for early stopping
        self.test_type = test_type  # Type of test: "naive t-test" or "alpha spending"

    def t_test_one_tailed(self, T_sample, C_sample):
        """
        Perform a one-tailed t-test on the samples.
        """
        # Perform t-test
        t_stat, p_two_tailed = stats.ttest_ind(T_sample, C_sample, equal_var=False)
       
        # Determine one-tailed p-value based on the direction of the test statistic
        if t_stat > 0:  # Test statistic aligns with H1: B > A
            p_one_tailed = p_two_tailed / 2
        else:  # Test statistic is opposite to H1: B > A
            p_one_tailed = 1 - (p_two_tailed / 2)
    
        return p_one_tailed
   
    @staticmethod
    def calculate_adjusted_alpha_one_sided(k, K, alpha):
        """
        Calculate the O'Brien-Fleming adjusted alpha threshold for the k-th interim analysis.
        """
        # Compute the standard normal quantile for the overall alpha level
        z_alpha = stats.norm.ppf(1 - alpha)
        
        # Adjust the quantile for the interim analysis
        adjusted_z = z_alpha * (k / K)**0.5
        
        # Convert the adjusted quantile back to an alpha level
        adjusted_alpha = 1 - stats.norm.cdf(adjusted_z)
        return adjusted_alpha

    def always_valid_p_value(self, T_sample, C_sample, early_stopping_settings):    
        # Calculate Z as the difference of T and C sample respectively
        Z = [t - c for t, c in zip(T_sample, C_sample)]
        
        # Get relevant stats
        Z_n = np.mean(Z)
        known_variance = np.var(Z)
        sample_size = len(Z)
        
        # Normal prior of differences
        theta_0, tau_squared = early_stopping_settings["avi_normal_prior_mean"], early_stopping_settings["avi_normal_prior_var"]
        
        # Calculate the mSPRT statistic Lambda_n_hat analytically
        lambda_n_hat = np.sqrt(2 * known_variance / (2 * known_variance + sample_size * tau_squared)) * \
                       np.exp((tau_squared * sample_size**2 * (Z_n - theta_0)**2) / 
                              (4 * known_variance * (2 * known_variance + sample_size * tau_squared)))
        
        # Compute the sequential p-value
        p_value = min(1, 1 / lambda_n_hat)
        
        return p_value
    
    def get_values(self):
        """
        Perform sequential testing and determine the p-value for early stopping.
        """
        n_observed, interim_tests = 0, []
        min_sample = self.early_stopping_settings["minimum_sample"]
        
        # Calculate the total number of planned interim tests
        if self.test_type == "alpha spending":
            k = math.floor(min_sample / self.early_stopping_settings["interim_test_interval"]) # start at min-sample's k
            K = math.floor(self.T["n"] / self.early_stopping_settings["interim_test_interval"])

        # Fixed Horizon
        if self.test_type in ["naive t-test", "alpha spending"]:
            p_value_fixed_horizon = self.t_test_one_tailed(self.T["sample"], self.C["sample"])
        elif self.test_type == "always valid inference":
            p_value_fixed_horizon = self.always_valid_p_value(self.T["sample"], self.C["sample"], self.early_stopping_settings)

        # Early Stopping
        while n_observed <= len(self.T["sample"]):
            alpha = self.early_stopping_settings["alpha"]
            T_sample = self.T["sample"][:n_observed]
            C_sample = self.C["sample"][:n_observed]
            if self.test_type in ["naive t-test", "alpha spending"]:
                p_value = self.t_test_one_tailed(T_sample, C_sample)
            elif self.test_type == "always valid inference":
                p_value = self.always_valid_p_value(T_sample, C_sample, self.early_stopping_settings)
            interim_tests.append((n_observed, p_value))

            # Check early stopping conditions based on the test type
            if n_observed >= min_sample:
                if n_observed == len(self.T["sample"]):
                    break  # Full sample reached
                elif self.test_type in ["naive t-test", "always valid inference"] and p_value < alpha:
                    break  # Early stopping for naive t-test / AVI with regular alpha
                elif self.test_type == "alpha spending": 
                    if p_value < self.calculate_adjusted_alpha_one_sided(k, K, alpha):
                        alpha = self.calculate_adjusted_alpha_one_sided(k, K, alpha) # update to early stopping alpha
                        break # Early stopping for alpha spending
                    else:
                        k += 1  # Increment interim test counter
                
            n_observed += self.early_stopping_settings["interim_test_interval"]

        return p_value_fixed_horizon, p_value, interim_tests, n_observed, alpha
