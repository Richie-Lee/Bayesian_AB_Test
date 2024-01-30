import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from typing import Tuple


class BayesianAB:

    def generate_data(self, n, mu, sigma):
        """
        Generate data from a normal distribution with mean `mu` and standard deviation `sigma`.
        The data should be generated using the provided `seed`.
        The function should return a numpy array of length `n`.

        Parameters
        ----------
        n
        mu
        sigma
        seed

        Returns
        -------
        data
        """
        return np.random.normal(mu, sigma, n)


    @staticmethod
    def likelihood_normal(
            mu_prior: float,
            sigma_prior: float,
            mu_observed: float,
            sigma_observed: float,
            hypothesis: Tuple[float, float] = (-np.inf, 0)
    ):
        """
        Calculate the likelihood of the observed data given the hypothesized parameters.
        The function should return the likelihood. The likelihood should be calculated using the normal distribution.
        This function is mainly used for testing purposes, since calculating the likelihood using an integral is
        computationally expensive, and in some cases numerically unstable.

        Parameters
        ----------
        mu_hypothesized (float): hypothesized mean
        sigma_hypothesized (float): hypothesized standard deviation
        mu_observed (float): observed mean
        sigma_observed (float): observed standard deviation
        hypothesis (tuple): tuple containing the lower and upper bound of the hypothesis

        Returns
        -------
        likelihood (float): likelihood of the observed data given the hypothesized parameters
        """

        lower_bound, upper_bound = hypothesis

        # Calculate the likelihood using the normal distribution
        scalar = 1 / (
                (
                        norm.cdf((upper_bound - mu_prior) / sigma_prior)
                        - norm.cdf((lower_bound - mu_prior) / sigma_prior)
                )
                * (2 * np.pi * sigma_observed * sigma_prior)
        )

        def integrand(mu):
            # Auxiliary function to calculate the likelihood. This is the integrand of the integral.
            return np.exp(
                -0.5
                * (
                        (mu_observed - mu) ** 2 / sigma_observed**2
                        + (mu - mu_prior) ** 2 / sigma_prior**2
                )
            )

        # Calculate the integral using the scipy quad function and provided bounds
        integrand_result = quad(integrand, lower_bound, upper_bound)[0]

        return scalar * integrand_result

    @staticmethod
    def log_likelihood_h0(
            mu_prior: float, sigma_prior: float, mu_observed: float, sigma_observed: float
    ) -> float:
        """
        Calculate the logarithmic likelihood given H0 and the observed data.
        H0 is the hypothesis that mu <= 0 (i.e. the treatment has no effect).
        The posterior should be calculated using the prior parameters `mu_h0` and `sigma_h0`.
        The observed data should be provided as `mu_observed` and `sigma_observed`.
        The function should return the logarithm of the likelihood.

        Parameters
        ----------
        mu_prior (float): prior mean
        sigma_prior (float): prior standard deviation
        mu_observed (float): observed mean
        sigma_observed (float): observed standard deviation

        Returns
        -------
        log_likelihood (float): logarithm of the likelihood given H0
        """
        # Updated mean / standard deviation
        mu_h0_prime = (mu_observed * sigma_prior**2 + mu_prior * sigma_observed**2) / (
                sigma_prior**2 + sigma_observed**2
        )
        sigma_h0_prime = np.sqrt(
            (sigma_prior**2 * sigma_observed**2) / (sigma_prior**2 + sigma_observed**2)
        )

        # Log marginal likelihood H0. Since we are using a normal distribution, we can use the log cdf function to
        # derive the log marginal likelihood. This is thus an exact solution.
        log_likelihood_h0 = (
                -0.5 * np.log(2 * np.pi * (sigma_prior**2 + sigma_observed**2))
                - norm.logcdf(-mu_prior / sigma_prior)
                + norm.logcdf(-mu_h0_prime / sigma_h0_prime)
                - 0.5 * (mu_prior - mu_observed) ** 2 / (sigma_observed**2 + sigma_prior**2)
        )

        # Revert log
        return log_likelihood_h0

    @staticmethod
    def log_likelihood_h1(
            mu_prior: float, sigma_prior: float, mu_observed: float, sigma_observed: float
    ) -> float:
        """
        Calculate the logarithmic likelihood given H1 and the observed data.
        H1 is the hypothesis that mu > 0 (i.e. the treatment has a positive effect).
        The posterior should be calculated using the prior parameters `mu_h1` and `sigma_h1`.
        The observed data should be provided as `mu_observed` and `sigma_observed`.
        The function should return the logarithm of the likelihood.

        Parameters
        ----------
        mu_prior (float): prior mean
        sigma_prior (float): prior standard deviation
        mu_observed (float): observed mean
        sigma_observed (float): observed standard deviation

        Returns
        -------
        log_likelihood (float): logarithm of the likelihood given H1
        """
        # Updated mean / standard deviation
        mu_h1_prime = (mu_observed * sigma_prior**2 + mu_prior * sigma_observed**2) / (
                sigma_prior**2 + sigma_observed**2
        )
        sigma_h1_prime = np.sqrt(
            (sigma_prior**2 * sigma_observed**2) / (sigma_prior**2 + sigma_observed**2)
        )

        # Log marginal likelihood H1. Since we are using a normal distribution, we can use the log cdf function to
        # derive the log marginal likelihood. This is thus an exact solution.
        log_likelihood_h1 = (
                -0.5 * np.log(2 * np.pi * (sigma_prior**2 + sigma_observed**2))
                + np.log(1 - norm.cdf(-mu_h1_prime / sigma_h1_prime))
                - np.log(1 - norm.cdf(-mu_prior / sigma_prior))
                - 0.5 * (mu_prior - mu_observed) ** 2 / (sigma_observed**2 + sigma_prior**2)
        )

        return log_likelihood_h1



    def get_bayes_factor1(self, data_treatment, data_control, prior_h0, prior_h1):
        # Mean difference / Pooled variance
        observed_mean = np.mean(data_treatment) - np.mean(data_control)
        observed_sigma = np.sqrt(np.var(data_treatment) / len(data_treatment) + np.var(
            data_control
        ) / len(data_control))

        h0 = self.__class__.likelihood_normal(prior_h0["mean"], np.sqrt(prior_h0["variance"]), observed_mean, observed_sigma, hypothesis=(-np.inf, 0))
        h1 = self.__class__.likelihood_normal(prior_h1["mean"], np.sqrt(prior_h1["variance"]), observed_mean, observed_sigma, hypothesis=(0, np.inf))

        return h1/h0, [h1, h0]


    def get_bayes_factor2(self, data_treatment, data_control, prior_h0, prior_h1):
        # Mean difference / Pooled variance
        observed_mean = np.mean(data_treatment) - np.mean(data_control)
        observed_sigma = np.sqrt(np.var(data_treatment) / len(data_treatment) + np.var(
            data_control
        ) / len(data_control))

        h0 = self.__class__.log_likelihood_h0(prior_h0["mean"], np.sqrt(prior_h0["variance"]), observed_mean, observed_sigma)
        h1 = self.__class__.log_likelihood_h1(prior_h1["mean"], np.sqrt(prior_h1["variance"]), observed_mean, observed_sigma)

        return np.exp(h1 - h0), [np.exp(h1), np.exp(h0)]


    def get_bf(self, data_treatment, data_control, H0_prior, H1_prior):
        """
        Function that computes Bayes factor for H0: mu <= 0 & H1: mu > 0
        """
        # Mean difference / Pooled variance
        y = np.mean(data_treatment) - np.mean(data_control)
        sigma_squared = np.var(data_treatment) / len(data_treatment) + np.var(
            data_control
        ) / len(data_control)

        # Get parameters
        mu_h0, sigma_h0 = H0_prior["mean"], np.sqrt(H0_prior["variance"])
        mu_h1, sigma_h1 = H1_prior["mean"], np.sqrt(H1_prior["variance"])

        # Updated mean / standard deviation
        mu_h0_prime = (y * sigma_h0**2 + mu_h0 * sigma_squared) / (
                sigma_h0**2 + sigma_squared
        )
        mu_h1_prime = (y * sigma_h1**2 + mu_h1 * sigma_squared) / (
                sigma_h1**2 + sigma_squared
        )
        sigma_h0_prime = np.sqrt(1 / (1 / sigma_squared + 1 / sigma_h0**2))
        sigma_h1_prime = np.sqrt(1 / (1 / sigma_squared + 1 / sigma_h1**2))

        # Log likelihood for H0 and H1
        log_likelihood_h0 = (
                - 0.5 * np.log(2 * np.pi * (sigma_h0**2 + sigma_squared))
                - norm.logcdf(-mu_h0 / sigma_h0)
                + norm.logcdf(-mu_h0_prime / sigma_h0_prime)
                - 0.5 * (mu_h0 - y) ** 2 / (sigma_squared + sigma_h0**2)
        )
        log_likelihood_h1 = (
                -0.5 * np.log(2 * np.pi * (sigma_h1**2 + sigma_squared))
                + np.log(1 - norm.cdf(-mu_h1_prime / sigma_h1_prime))
                - np.log(1 - norm.cdf(-mu_h1 / sigma_h1))
                - 0.5 * (mu_h1 - y) ** 2 / (sigma_squared + sigma_h1**2)
        )

        # Calculate bayes factor
        log_bf = log_likelihood_h1 - log_likelihood_h0
        bf = np.exp(log_bf)

        return bf, [np.exp(log_likelihood_h1), np.exp(log_likelihood_h0)]



bf = BayesianAB()


data_control = bf.generate_data(n=1000, mu=0, sigma=1)
data_treatment = bf.generate_data(n=1000, mu=0.1, sigma=1)

bf1 = bf.get_bayes_factor1(
    data_treatment=data_treatment,
    data_control=data_control,
    prior_h0={"mean": 0, "variance": 2},
    prior_h1={"mean": 0.1, "variance": 2},
)
bf2 = bf.get_bayes_factor2(
    data_treatment=data_treatment,
    data_control=data_control,
    prior_h0={"mean": 0, "variance": 2},
    prior_h1={"mean": 0.1, "variance": 2},
)

bf3 = bf.get_bf(
    data_treatment=data_treatment,
    data_control=data_control,
    H0_prior={"mean": 0, "variance": 2},
    H1_prior={"mean": 0.1, "variance": 2},
)

print('BF Integral', bf1)
print('BF analytical (corrected)', bf2)
print('BF original', bf3)
