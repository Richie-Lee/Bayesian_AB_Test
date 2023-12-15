import random
import numpy as np

def get_bernoulli_sample(mean, n):
    # Sample bernoulli distribution with relevant metrics
    samples = [1 if random.random() < mean else 0 for _ in range(n)]
    converted = sum(samples)
    mean = converted/n 

    return samples, converted, mean

def get_normal_sample(mean, variance, n):
    return np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
