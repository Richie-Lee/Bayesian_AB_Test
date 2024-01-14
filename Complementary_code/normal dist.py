import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_multiple_normal_distributions(distributions, sample_size=10000):
    """
    Plot multiple normal distributions on the same figure using matplotlib.

    Parameters:
    distributions (list of dict): List of dictionaries, each containing 'mean', 'variance', 'title', and optional 'linestyle'.
    sample_size (int): The number of samples to generate for each distribution.
    """
    plt.figure(figsize=(10, 6))

    for dist in distributions:
        mean = dist['mean']
        variance = dist['variance']
        title = dist['title']
        linestyle = dist.get('linestyle', '-')
        std_dev = np.sqrt(variance)

        # Generate sample
        sample = np.random.normal(loc=mean, scale=std_dev, size=sample_size)

        # KDE using scipy
        kde = gaussian_kde(sample, bw_method=std_dev/2.0)
        x_range = np.linspace(np.min(sample), np.max(sample), 1000)
        kde_values = kde.evaluate(x_range)

        # Determine color based on mean
        color = 'green' if mean > 0 else 'red'

        # Plotting the KDE
        plt.plot(x_range, kde_values, label=f'{title} (mean={mean}, var={variance})',
                 linestyle=linestyle, color=color)
    
    plt.axvline(x = 0, color = "black", linewidth = 0.5)
    
    plt.title('Multiple Normal Distributions')
    plt.xlabel('Value')
    plt.xlim(-6, 6)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage
distributions = [
    {'mean': 0.5, 'variance': 2, 'title': 'H1 true effect', 'linestyle': '--'},
    {'mean': -0.5, 'variance': 2, 'title': 'H0 true effect', 'linestyle': '--'}
    # {'mean': 6, 'variance': 1, 'title': 'prior', 'linestyle': ':'}
]

plot_multiple_normal_distributions(distributions)
