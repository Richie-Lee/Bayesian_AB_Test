from scipy.stats import beta
import numpy as np

class get_metrics():
    def __init__(self, T, T_prior, C, C_prior, prior_type, bf, prior_odds):
        self.T, self.C = T, C
        self.prior_type = prior_type
        self.T_prior, self.C_prior = T_prior, C_prior
        self.bf = bf
        self.prior_odds = prior_odds
        
        # Execute main method
        self.get_values()
    
    def sample_posterior_distribution(self, T, T_prior, C, C_prior, prior_type):
        # Unpack outcomes
        c_t, n_t = T["converted"], T["n"]  # Successes and total observations for treatment group
        c_c, n_c = C["converted"], C["n"]  # Successes and total observations for control group
        
        if prior_type == "beta":
            # Compute posterior parameters
            post_alpha_t, post_beta_t = T_prior["alpha"] + c_t, T_prior["beta"] + (n_t - c_t)
            post_alpha_c, post_beta_c = C_prior["alpha"] + c_c, C_prior["beta"] + (n_c - c_c)
        
            # Sample from the posterior distributions
            t_posterior_samples = beta.rvs(post_alpha_t, post_beta_t, size = 100_000)
            c_posterior_samples = beta.rvs(post_alpha_c, post_beta_c, size = 100_000)
        
        return t_posterior_samples, c_posterior_samples
    
    
    
    def posterior_odds(self, prior_odds, bf):
        # Calculate posterior prob & convert to percentage
        post_odds = prior_odds * bf
        post_prob_effect = round(post_odds / (post_odds + 1) * 100, 2)
        return post_prob_effect
    
    
    
    def uplift(self, T, C):
        # Uplift is observed difference in conversion rates
        return round(T["converted"]/T["n"] - C["converted"]/C["n"], 4)
    
    
    
    def prob_treatment_beats_control(self):
        t_posterior_samples, c_posterior_samples =  self.sample_posterior_distribution(self.T, self.T_prior, self.C, self.C_prior, self.prior_type)
        proportion_greater = round(np.mean(t_posterior_samples > c_posterior_samples) * 100, 2)
        return proportion_greater
        
    
    
    def loss(self):
        t_posterior_samples, c_posterior_samples =  self.sample_posterior_distribution(self.T, self.T_prior, self.C, self.C_prior, self.prior_type)
        # Compute the differences for samples where control > treatment
        differences = t_posterior_samples - c_posterior_samples
        negative_differences = differences[t_posterior_samples > c_posterior_samples]
    
        # Return the (rounded) average loss in %
        if len(negative_differences) > 0:
          expected_loss = np.mean(negative_differences)
          return round(expected_loss, 4)
        # If no samples where treatment < control, return 0 loss (note that this is rounded, as it should always be non-zero following stats concepts)
        else:
          return 0
        
    def get_values(self):
        metrics = {
        "P[H1|data]" : self.posterior_odds(self.prior_odds, self.bf),
        "uplift" : self.uplift(self.T, self.C),
        "P[T>C]" : self.prob_treatment_beats_control(),
        "loss" : self.loss()
            }
        
        return metrics