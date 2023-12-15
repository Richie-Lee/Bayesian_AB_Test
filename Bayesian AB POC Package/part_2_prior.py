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
    
    def normal_prior(self, parameters):
        # Store the parameters for normal distribution
        H0_prior = {"mean": parameters["mean_H0"], "variance": parameters["variance_H0"]}
        H1_prior = {"mean": parameters["mean_H1"], "variance": parameters["variance_H1"]}
        return H0_prior, H1_prior

    def get_values(self):
        if self.prior_type == "beta":
            return self.beta_prior(self.parameters)
        elif self.prior_type == "normal":
            return self.normal_prior(self.parameters)
        else:
            raise ValueError(f"Prior type '{self.prior_type}' is not supported.")
    
