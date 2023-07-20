import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
import random

from datetime import datetime

# Track total runtime
_start_time = datetime.now()


"""
Part 0: Settings & Hyperparameters
"""
C = {"n": 1000, "true_prob": 0.5} # control
T = {"n": 1000, "true_prob": 0.55} # treatment




"""
Part 1: Generate Data
"""

def get_bernoulli_sample(mean, n):
    # Sample bernoulli distribution with relevant metrics
    samples = [1 if random.random() < mean else 0 for _ in range(n)]
    converted = sum(samples)
    mean = converted/n

    # Create a DataFrame
    data = {
        "userId": range(1, n + 1),
        "converted": samples
    }
    data = pd.DataFrame(data)
    
    return data, converted, mean 

C["sample"], C["converted"], C["sample_mean"] = get_bernoulli_sample(mean = C["true_prob"], n = C["n"])
T["sample"], T["converted"], T["sample_mean"] = get_bernoulli_sample(mean = T["true_prob"], n = T["n"])







# Print execution time
print(f"===============================\nTotal runtime:  {datetime.now() - _start_time}")

