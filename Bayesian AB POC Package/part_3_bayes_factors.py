import numpy as np
from scipy.special import betaln

class get_bayes_factor():
    def __init__(self, T, T_prior, C, C_prior, prior_type, early_stopping_settings):
        self.T, self.C = T, C
        self.prior_type = prior_type
        self.T_prior, self.C_prior = T_prior, C_prior
        self.es_settings = early_stopping_settings
        
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
    
    
    
    def get_values(self):
        """
        Fixed Horizon
        """
        c_t, n_t = self.T["converted"], self.T["n"]  # Successes and total observations for treatment group
        c_c, n_c = self.C["converted"], self.C["n"]  # Successes and total observations for control group
        
        if self.prior_type == "beta":
            bf_fixed_horizon = self.beta_bf(c_t, n_t, c_c, n_c, self.T_prior, self.C_prior)
        
        
        """
        Early Stopping
        """
        n_observed, interim_tests = 0, []
        
        # Convert input probability to odds parameter k (Bayes factors stopping rule)
        k = (self.es_settings["prob_early_stopping"] * 100) / (100 - (self.es_settings["prob_early_stopping"] * 100))
        
        while n_observed <= self.T["n"]:
            # Mask observations to resemble partial sampling
            c_t, n_t = sum(self.T["sample"][:n_observed]), n_observed
            c_c, n_c = sum(self.C["sample"][:n_observed]), n_observed

            if self.prior_type == "beta":
                bf = self.beta_bf(c_t, n_t, c_c, n_c, self.T_prior, self.C_prior)
            
            # Store results
            interim_tests.append((n_observed, bf))
          
            # Stopping criteria
            if (bf > k or bf < 1/k) or n_observed == self.T["n"]:
                break

            # Extend sample & get conversions
            n_observed += self.es_settings["interim_test_interval"]
        
        return bf_fixed_horizon, bf, interim_tests, k, n_observed
        
        