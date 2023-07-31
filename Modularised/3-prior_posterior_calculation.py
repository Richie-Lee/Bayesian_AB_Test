import scipy.stats as stats

class prior_posterior:
    def __init__(self, prior, C, T, model_selection):
        self.prior = prior
        self.C = C
        self.T = T
        self.model_selection = model_selection
        
        # Always execute main class
        self.get_results() 
    
    def beta_distribution(self, prior_belief, weight, converted, n):
        """
        Prior (Conjugate)
        """
        prior_a = round(prior_belief, 1) * weight + 1
        prior_b = (1 - round(prior_belief, 1)) * weight + 1
        
        beta_prior = stats.beta(prior_a, prior_b)
        sample_prior = beta_prior.rvs(size=n)
        
        """
        Posterior
        """
        beta_posterior = stats.beta(prior_a + converted, prior_b + (n - converted))
        sample_posterior = beta_posterior.rvs(size=n)
        
        return beta_prior, sample_prior, beta_posterior, sample_posterior

    def get_results(self):
        # Select desired distribution with corresponding posterior calculation
        if self.model_selection == "beta":
            C_prior, C_prior_sample, C_post, C_post_sample = self.beta_distribution(
                self.prior["prior_control"], self.prior["weight"], self.C["converted"], self.C["n"]
            )
            
            T_prior, T_prior_sample, T_post, T_post_sample = self.beta_distribution(
                self.prior["prior_treatment"], self.prior["weight"], self.T["converted"], self.T["n"]
            )
        
        # Raise ValueError for invalid distributions
        else:
            raise ValueError(f"the distribution '{self.model_selection}' is not supported by this implementation of Bayesian A/B testing")
                        
        return C_prior, C_prior_sample, C_post, C_post_sample, T_prior, T_prior_sample, T_post, T_post_sample 


# C["prior_dist"], C["prior_sample"], C["post_dist"], C["post_dist"], T["prior_dist"], T["prior_sample"], T["post_dist"], T["post_dist"] = pp.prior_posterior(prior, C, T, model_selection = "beta")
