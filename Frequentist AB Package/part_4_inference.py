import numpy as np

class get_metrics():
    def __init__(self, T, C, data_type):
        self.T = T 
        self.C = C
        self.data_type = data_type
        
        # Execute main method
        self.get_values()
    
    def uplift(self, T, C, data_type):
        # Uplift is observed differences
        if data_type == "binary":
            uplift = round(T["converted"]/T["n"] - C["converted"]/C["n"], 4)
        elif data_type == "continuous":
            uplift = round(np.mean(T["sample"]) - np.mean(C["sample"]), 4)
        elif data_type == "real":
            uplift = round(np.mean(T["sample"]) - np.mean(C["sample"]), 4)
        return uplift     
        
    def get_values(self):
        metrics = {
        "uplift" : self.uplift(self.T, self.C, self.data_type)
        # optional confidence intervals
            }
        
        return metrics
    

    
    