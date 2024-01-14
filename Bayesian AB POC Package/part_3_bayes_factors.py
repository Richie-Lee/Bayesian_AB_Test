import numpy as np
from scipy.special import betaln
from scipy.stats import norm

class get_bayes_factor():
    def __init__(self, T, C, prior_type, early_stopping_settings, T_prior=None, C_prior=None, H0_prior=None, H1_prior=None):
        self.T, self.C = T, C
        self.prior_type = prior_type
        self.early_stopping_settings = early_stopping_settings
        self.T_prior, self.C_prior = T_prior, C_prior
        self.H0_prior, self.H1_prior = H0_prior, H1_prior
        # Execute main method
        self.get_values()

    def beta_bf(self, c_t, n_t, c_c, n_c, T_prior, C_prior):    
        # Unpack prior values
        alpha_0, beta_0 = C_prior["alpha"], C_prior["beta"] # prior under H0 (control in this design)
        alpha_t, beta_t = T_prior["alpha"], T_prior["beta"] # prior under Treatment
        alpha_c, beta_c = C_prior["alpha"], C_prior["beta"] # prior under Control
        
        # Log probability of data under H0
        log_prob_data_H0 = betaln(alpha_0 + c_t + c_c, beta_0 + n_t + n_c - c_t - c_c) - betaln(alpha_0, beta_0)
    
        # Log probability of data under H1 for treatment & control group
        log_prob_data_H1_treatment = betaln(alpha_t + c_t, beta_t + n_t - c_t) - betaln(alpha_t, beta_t)
        log_prob_data_H1_control = betaln(alpha_c + c_c, beta_c + n_c - c_c) - betaln(alpha_c, beta_c)
        # Log joint probability of data under H1
        log_prob_data_H1 = log_prob_data_H1_treatment + log_prob_data_H1_control
    
        # Compute Log Bayes Factor and convert to regular Bayes Factor
        log_bf = log_prob_data_H1 - log_prob_data_H0
        bf = np.exp(log_bf)
    
        return bf
    
    def normal_bf_new(self, y, sigma_squared, H0_prior, H1_prior):
        # Get parameters
        mu_h0, sigma_h0 = H0_prior["mean"], np.sqrt(H0_prior["variance"])
        mu_h1, sigma_h1 = H1_prior["mean"], np.sqrt(H1_prior["variance"])
        
        # Updated mean / standard deviation
        mu_h0_prime = (y * sigma_h0**2 + mu_h0 * sigma_squared) / (sigma_h0**2 + sigma_squared)
        mu_h1_prime = (y * sigma_h1**2 + mu_h1 * sigma_squared) / (sigma_h1**2 + sigma_squared)
        sigma_h0_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h0**2))
        sigma_h1_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h1**2))
        
        # Log likelihood for H0 and H1
        log_likelihood_h0 = 0.5 * np.log(sigma_h0**2 + sigma_squared) + norm.logcdf(-mu_h0 / sigma_h0) - norm.logcdf(-mu_h0_prime / sigma_h0_prime) + 0.5 * (mu_h0 - y)**2 / (sigma_squared + sigma_h0**2)
        log_likelihood_h1 = - 0.5 * np.log(sigma_h1**2 + sigma_squared) + np.log(1 - norm.cdf(-mu_h1_prime / sigma_h1_prime)) - np.log(1 - norm.cdf(mu_h1 / sigma_h1)) - 0.5 * (mu_h1 - y)**2 / (sigma_squared + sigma_h1**2)
        
        # Calculate bayes factor
        log_bf = log_likelihood_h1 + log_likelihood_h0
        bf = np.exp(log_bf)
        
        # print("H1:", np.exp(log_likelihood_h1), "       H0: ", np.exp(log_likelihood_h0))
        
        return bf
    
    def normal_bf(self, y, sigma_squared, H0_prior, H1_prior):
        # Get parameters
        mu_h0, sigma_h0 = H0_prior["mean"], np.sqrt(H0_prior["variance"])
        mu_h1, sigma_h1 = H1_prior["mean"], np.sqrt(H1_prior["variance"])
        
        # Updated mean / standard deviation
        mu_h0_prime = (y * sigma_h0**2 + mu_h0 * sigma_squared) / (sigma_h0**2 + sigma_squared)
        mu_h1_prime = (y * sigma_h1**2 + mu_h1 * sigma_squared) / (sigma_h1**2 + sigma_squared)
        sigma_h0_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h0**2))
        sigma_h1_prime = np.sqrt(1 / (1/sigma_squared + 1/sigma_h1**2))
        
        # Log likelihood for H0 and H1
        log_likelihood_h0 = norm.logcdf(-mu_h0_prime / sigma_h0_prime) - norm.logcdf(-mu_h0 / sigma_h0)
        log_likelihood_h1 = np.log(1 - norm.cdf(-mu_h1_prime / sigma_h1_prime)) - np.log(1 - norm.cdf(-mu_h1 / sigma_h1))
        
        # Exponent term calculation
        exponent_term = 0.5 * (
            (mu_h1**2 / (sigma_squared + sigma_h1**2)) - (mu_h1_prime**2 / (sigma_squared + sigma_h1_prime**2)) +
            (mu_h0_prime**2 / (sigma_squared + sigma_h0_prime**2)) - (mu_h0**2 / (sigma_squared + sigma_h0**2))
        )
        
       
        # Calculate bayes factor
        log_bf = log_likelihood_h1 - log_likelihood_h0 + exponent_term
        bf = np.exp(log_bf)
        
        # print(np.exp(log_likelihood_h0), np.exp(log_likelihood_h1), exponent_term, "-> ", bf)
        
        return bf
    
    def get_values(self):
        # Common setup for both beta and normal priors
        n_observed, interim_tests = 0, []
        min_sample = self.early_stopping_settings["minimum_sample"]
        k = (self.early_stopping_settings["prob_early_stopping"] * 100) / (100 - self.early_stopping_settings["prob_early_stopping"] * 100)

        if self.prior_type == "beta":
            # Fixed Horizon for Beta
            c_t, n_t = self.T["converted"], self.T["n"]
            c_c, n_c = self.C["converted"], self.C["n"]
            bf_fixed_horizon = self.beta_bf(c_t, n_t, c_c, n_c, self.T_prior, self.C_prior)

            # Early Stopping for Beta
            while n_observed <= self.T["n"]:
                # Mask observations to resemble partial sampling
                c_t, n_t = sum(self.T["sample"][:n_observed]), n_observed
                c_c, n_c = sum(self.C["sample"][:n_observed]), n_observed
                bf = self.beta_bf(c_t, n_t, c_c, n_c, self.T_prior, self.C_prior)
                interim_tests.append((n_observed, bf))
                
                if n_observed >= min_sample:
                    if (bf > k or bf < 1/k) or n_observed == self.T["n"]:
                        break

                n_observed += self.early_stopping_settings["interim_test_interval"]

            return bf_fixed_horizon, bf, interim_tests, k, n_observed

        elif self.prior_type == "normal":
            # Fixed Horizon for Normal
            y_bar = np.mean(self.T["sample"]) - np.mean(self.C["sample"])
            pooled_variance = np.var(self.T["sample"])/self.T["n"] + np.var(self.C["sample"])/self.C["n"]
            bf_fixed_horizon = self.normal_bf_new(y_bar, pooled_variance, self.H0_prior, self.H1_prior) # --------------

            # Early Stopping for Normal
            while n_observed <= self.T["n"]:
                T_sample = self.T["sample"][:n_observed]
                C_sample = self.C["sample"][:n_observed]
                y_bar = np.mean(T_sample) - np.mean(C_sample)
                pooled_variance = np.var(T_sample)/len(T_sample) + np.var(C_sample)/len(C_sample)
                bf = self.normal_bf_new(y_bar, pooled_variance, self.H0_prior, self.H1_prior) # --------------
                interim_tests.append((n_observed, bf))
                
                if n_observed >= min_sample:
                    if (bf > k or bf < 1/k) or n_observed == self.T["n"]:
                        break

                n_observed += self.early_stopping_settings["interim_test_interval"]

            return bf_fixed_horizon, bf, interim_tests, k, n_observed

        else:
            raise ValueError(f"Unsupported prior_type: {self.prior_type}")

       