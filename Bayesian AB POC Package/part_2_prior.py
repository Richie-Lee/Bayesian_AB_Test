class get_prior():
    def __init__(self, prior_type, prior_parameters):
        self.prior_type = prior_type
        self.parameters = prior_parameters
        
        # Execute main method
        self.get_values()
        
    def beta_prior(self, parameters):
        # Sample from Beta(weight(prior belief) + 1, weight(1 - prior belief) + 1)       
        T_alpha = round(parameters["T_prior_prob"], 2) * parameters["T_weight"] + 1
        T_beta = (1 - round(parameters["T_prior_prob"], 2)) * parameters["T_weight"] + 1
        C_alpha = round(parameters["C_prior_prob"], 2) * parameters["C_weight"] + 1
        C_beta = (1 - round(parameters["C_prior_prob"], 2)) * parameters["C_weight"] + 1
        
        # Collect and store parameters
        T_prior = {"alpha": T_alpha, "beta": T_beta, "prior_prob": parameters["T_prior_prob"]}
        C_prior = {"alpha": C_alpha, "beta": C_beta, "prior_prob": parameters["C_prior_prob"]}
        return C_prior, T_prior
    
    def get_values(self):
        if self.prior_type == "beta":
            C_prior, T_prior = self.beta_prior(self.parameters)
            
        # Raise ValueError for unsupported/invalid prior types
        else:
            raise ValueError(f"the '{self.prior_type}' prior is not supported by this implementation of Bayesian A/B testing")
        
        return C_prior, T_prior
