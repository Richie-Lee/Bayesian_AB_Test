Modularised version of the bayesian A/B test to test and improve new features in a more convenient and efficient way

Modules:

1. **Main**: executes all the code. Specify experiment design decisions (distributions, parameters, ...) here.
2. **Data generation**: Generates the data.
3. **Likelihood and Early Stopping**: Computes the data likelihood and performs early stopping if enabled.
4. **Prior posterior calculation**: Creates prior and corresponding posterior distributions (using likelihood data as well).
5. **Reporting**: Extracs the relevant metrics, summary statistics and probabilities from the calculated posteriors.



From a high-level, the architecture looks as follows:
![High level overview modules](https://github.com/Richie-Lee/Msc_Thesis/blob/main/img/Architecture%20code.jpg)

