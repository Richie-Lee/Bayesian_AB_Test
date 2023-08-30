The Beta distribution is the conjugate prior for the Binomial likelihood. This means that when you multiply a Binomial likelihood with a Beta prior, the resulting posterior is also a Beta distribution. This nice property simplifies the computation a lot.

Let's find the marginal likelihood for binomial data using a Beta prior.

**Setup:**

- Suppose you've conducted \( n \) trials and observed \( k \) successes.
- Your Binomial likelihood is \( P(x=k|p) = \binom{n}{k} p^k (1-p)^{n-k} \) where \( p \) is the probability of success.
- You have a Beta prior on \( p \) given by \( P(p|\alpha, \beta) = \frac{p^{\alpha-1} (1-p)^{\beta-1}}{B(\alpha, \beta)} \) where \( B(\alpha, \beta) \) is the Beta function.

**Marginal Likelihood:**

The marginal likelihood (also known as the model evidence) can be found by integrating out \( p \) from the product of the likelihood and the prior:

\[ P(x=k) = \int_0^1 P(x=k|p)P(p|\alpha, \beta) dp \]

Plugging in our likelihood and prior:

\[ P(x=k) = \int_0^1 \binom{n}{k} p^k (1-p)^{n-k} \frac{p^{\alpha-1} (1-p)^{\beta-1}}{B(\alpha, \beta)} dp \]

After simplifying:

\[ P(x=k) = \binom{n}{k} \frac{B(k+\alpha, n-k+\beta)}{B(\alpha, \beta)} \]

Where \( B(a, b) \) is the Beta function. The term \( B(k+\alpha, n-k+\beta) \) can be interpreted as the normalization constant that makes the posterior distribution (after observing the data) a valid probability distribution.

**Intuition:**

The above integration is feasible due to the conjugacy of the Beta and Binomial distributions. The result gives us the probability of observing \( k \) successes out of \( n \) trials, averaged over all possible values of \( p \), weighted by our prior beliefs about \( p \) (encoded in the Beta distribution parameters \( \alpha \) and \( \beta \)).

This marginal likelihood can be used to compare different models (with different priors or different likelihood forms) to see which one better predicts the observed data.
