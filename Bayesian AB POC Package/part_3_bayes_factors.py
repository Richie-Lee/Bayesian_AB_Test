import numpy as np
from scipy.special import betaln


class get_bayes_factor():
    
    def __init__(self, T, T_prior, C, C_prior, prior_type):
        self.prior_type = prior_type
        self.T, self.C = T, C
        self.T_prior, self.C_prior = T_prior, C_prior
        
        # Execute main method
        self.get_values()

    def beta_bf(self, T, C, T_prior, C_prior):    
        # Unpack prior values
        alpha_0, beta_0 = C_prior["alpha"], C_prior["beta"] # prior under H0 (control in this design)
        alpha_t, beta_t = T_prior["alpha"], T_prior["beta"] # prior under Treatment
        alpha_c, beta_c = C_prior["alpha"], C_prior["beta"] # prior under Control
        
        # Unpack outcomes
        c_t, n_t = T["converted"], T["n"]  # Successes and total observations for treatment group
        c_c, n_c = C["converted"], C["n"]  # Successes and total observations for control group

        # Log probability of data under H0
        log_prob_data_H0 = betaln(alpha_0 + c_t + c_c, beta_0 + n_t + n_c - c_t - c_c) - betaln(alpha_0, beta_0)
    
        # Log probability of data under H1 for treatment & control group
        log_prob_data_H1_treatment = betaln(alpha_t + c_t, beta_t + n_t - c_t) - betaln(alpha_t, beta_t)
        log_prob_data_H1_control = betaln(alpha_c + c_c, beta_c + n_c - c_c) - betaln(alpha_c, beta_c)
        # Log joint probability of data under H1
        log_prob_data_H1 = log_prob_data_H1_treatment + log_prob_data_H1_control
    
        # Compute Log Bayes Factor and convert to regular Bayes Factor
        log_bf_10 = log_prob_data_H1 - log_prob_data_H0
        bf_10 = np.exp(log_bf_10)
    
        return bf_10
    
    def get_values(self):
        if self.prior_type == "beta":
            bf = self.beta_bf(self.T, self.C, self.T_prior, self.C_prior)
            
        return bf