# How MCMC and PyMC Work

## **Markov Chain Monte Carlo (MCMC)**

MCMC is a computational method for approximating complex probability distributions, often used in Bayesian inference to sample from a posterior distribution when direct computation is infeasible. The method constructs a Markov chain that converges to the desired distribution, allowing statistical properties to be estimated from the generated samples.

**Key Concepts in MCMC**:

1. **Markov Chain**: A sequence of random variables where the next state depends only on the current state.
2. **Monte Carlo**: Uses repeated random sampling to approximate results.
3. **Target Distribution**: The probability distribution you want to sample from, often a posterior distribution in Bayesian analysis.
4. **Convergence**: The process by which the Markov chain stabilizes to approximate the target distribution.

**Common MCMC Algorithms**:

- **Metropolis-Hastings**: Proposes new samples and decides their acceptance based on the ratio of the target densities.
- **Gibbs Sampling**: Breaks sampling into smaller parts by iteratively sampling from conditional distributions.
- **Hamiltonian Monte Carlo (HMC)**: Utilizes gradient information to propose efficient updates (used in PyMC).

## **PyMC**

[PyMC](https://www.pymc.io/) is a Python library for probabilistic programming that simplifies defining and fitting complex statistical models using MCMC and other Bayesian methods. It provides a high-level framework for model definition, sampling, and inference.

**How PyMC Works**:

1. **Model Definition**:

   - Users define variables as random variables with specified priors and likelihoods.
   - Models can represent complex statistical relationships, such as Bayesian regression or hierarchical models.

2. **Sampling**:
   - PyMC employs state-of-the-art MCMC algorithms, like **NUTS** (No-U-Turn Sampler), a variant of HMC that automatically tunes parameters.
   - Sampling generates draws from the posterior distribution for inference.

3. **Inference and Diagnostics**:
   - PyMC provides tools for analyzing posterior distributions, including trace plots, autocorrelation, and summary statistics.
   - Diagnostics assess convergence and the quality of sampling.

4. **Integration**:
   - PyMC integrates well with visualization tools (e.g., ArviZ for diagnostics) and computational backends like TensorFlow and Theano.

## **Why Use PyMC?**

- **Ease of Use**: Intuitive APIs for defining and fitting probabilistic models.
- **Advanced Sampling**: State-of-the-art methods like NUTS ensure efficient sampling.
- **Extensibility**: Supports custom priors, likelihoods, and hierarchical structures.
- **Community Support**: Active ecosystem and resources for learning Bayesian statistics.

PyMC is a powerful tool for Bayesian modeling, abstracting the complexities of MCMC while providing flexibility for advanced users.
  
## arvis plots

### Output of `arviz.plot_trace`

The `arviz.plot_trace` function generates **trace plots** for the posterior distributions of the parameters in a Bayesian model. A trace plot is a combination of two key visualizations:

1. **Trace Plot**:
   - A time series of sampled values for each parameter across iterations of the MCMC process.
   - It helps assess how well the Markov chain has explored the parameter space.
   - A well-mixed and stationary trace (no trends, consistent variability) indicates good sampling and convergence.

2. **Posterior Distribution Plot**:
   - A kernel density estimate (KDE) or histogram of the sampled values for the parameter.
   - Provides a summary of the posterior distribution for each parameter.

#### **Key Features of `arviz.plot_trace`**

1. **Parameter Rows**:
   - Each row corresponds to a parameter or variable in the model.
   - Includes both the trace plot and the posterior distribution for that parameter.

2. **Multiple Chains**:
   - If multiple MCMC chains are used, each chain's trace is plotted in a different color.
   - This helps assess chain convergence and mixing.

3. **Stationarity and Mixing**:
   - A well-converged trace should appear stationary, without trends or drifts.
   - Chains should overlap and mix well, indicating efficient exploration of the posterior.

4. **Density Representation**:
   - The posterior distribution on the right shows the central tendency and spread of the parameter estimates.
   - Multiple peaks in the density could indicate convergence issues or multimodal distributions.

#### **How to Interpret the Trace Plot**

- **Trace Plots**:
  - Look for **stationarity**: The chains should oscillate around a consistent mean without trends or drifts.
  - Assess **mixing**: Overlapping chains indicate good mixing.
  - Identify anomalies, such as abrupt jumps, trends, or long flat regions, which may suggest poor sampling or convergence issues.

- **Posterior Distribution**:
  - Examine the shape of the distribution: A unimodal, smooth distribution is desirable.
  - Use the posterior to identify the central tendency (mean, median) and spread (credible intervals) for each parameter.

#### **Example Trace Plot**

For a parameter like `alpha`:

- **Trace Plot (Left)**:
  - A time series showing sampled values for `alpha` from each MCMC chain.
  - Should exhibit stationary behavior with no trends.

- **Posterior Distribution (Right)**:
  - A KDE plot summarizing the posterior distribution of `alpha`, showing the mean and variability.

### **Common Use Cases**

- **Diagnose Convergence**:
  - Identify whether chains have reached stationarity and mixed well.
- **Assess Sampling Quality**:
  - Poor mixing or trends in the trace suggest the need for more samples or better tuning.
- **Summarize Posterior**:
  - The posterior distribution plot helps infer the parameter's central tendency and credible intervals.

By combining these visualizations, `arviz.plot_trace` provides a powerful diagnostic tool for evaluating the results of Bayesian MCMC sampling.

### Rank Bars Plot (`arviz.plot_trace()`)

In the **rank bars** plot within `arviz.plot_trace()`, the **y-axis** represents the **frequency or proportion of ranks** within the chain for the given parameter.

#### **Y-Axis Interpretation**

- The rank bars divide the sampled values into bins (rank groups) based on their relative order (rank) across all samples.
- The **y-axis values** indicate how often samples from a specific Markov Chain Monte Carlo (MCMC) chain fall into a given rank bin.
- If the ranks are uniformly distributed across bins (and chains), the heights of the bars on the y-axis will be approximately equal.

#### **Key Points**

1. **Uniform Distribution**:

   - If the model has converged and chains are mixing well, the bars will have roughly the same height, meaning each rank bin is equally populated.

2. **Non-Uniform Distribution**:
   - Skewed or uneven heights suggest convergence or mixing issues. For example:
     - A chain stuck in one part of the parameter space may overpopulate certain rank bins.
     - Poor mixing could result in some chains dominating certain ranks while avoiding others.

#### **Example**

- **Y-axis scale (relative frequencies)**:
  - Suppose you have 100 samples per chain and the ranks are divided into 10 bins. If mixing is perfect, each rank bin will have about 10% (0.1) of the samples for each chain, and the bars on the y-axis will have a height of 0.1.

- **Issues with mixing or convergence**:
  - If one chain gets "stuck," it might have more than 10% in one bin and less than 10% in others.

This y-axis behavior is crucial for visually assessing the quality of the sampling process.

### `arviz.plot_bf`

The `arviz.plot_bf` function in ArviZ estimates and visualizes the **Bayes Factor (BF)** for comparing two nested models in Bayesian analysis. It assesses the evidence provided by the data in favor of one model (alternative hypothesis, \( H_1 \)) over another (null hypothesis, \( H_0 \)), where \( H_0 \) is a special case of \( H_1 \). This function is particularly useful for hypothesis testing involving point-null hypotheses.

#### **Key Features of `arviz.plot_bf`**

1. **Bayes Factor Estimation**:
   - Estimates the Bayes Factor using the **Savage-Dickey density ratio** method, which is applicable when comparing nested models with a point-null hypothesis. [oai_citation_attribution:1‡ArviZ](https://python.arviz.org/en/stable/api/generated/arviz.plot_bf.html?utm_source=chatgpt.com)

2. **Visualization**:
   - Plots the prior and posterior distributions of the parameter of interest, highlighting the density at the reference value (e.g., zero) to illustrate the calculation of the Bayes Factor.

3. **Customization**:
   - Allows specification of the parameter to test, the reference value for the null hypothesis, custom prior distributions, and plot aesthetics such as colors and figure size.

### **Savage-Dickey Density Ratio**

The **Savage-Dickey Density Ratio** is a method for computing the **Bayes Factor (BF)** to compare a point-null hypothesis ($H_0$) against an alternative hypothesis ($H_1$) in Bayesian inference. It leverages the relationship between prior and posterior distributions, providing an efficient way to compute the Bayes Factor for nested models where $H_0$ is a special case of $H_1$.

### **Definition**

The Savage-Dickey Density Ratio is defined as:

$$
BF_{10} = \frac{P(\theta = \theta_0 \mid \text{data})}{P(\theta = \theta_0)}
$$

Where:

- $P(\theta = \theta_0 \mid \text{data})$: The posterior density at the null value $\theta_0$.
- $P(\theta = \theta_0)$: The prior density at the null value $\theta_0$.

### **Key Idea**

The Bayes Factor is computed by comparing the relative densities of the prior and posterior distributions at the specific point representing the null hypothesis. Since the posterior is proportional to the product of the prior and the likelihood, this approach avoids integrating over the full parameter space, simplifying the calculation.

## **When to Use It**

The Savage-Dickey Density Ratio is appropriate when:

1. **Nested Models**: The null hypothesis ($H_0$) is a special case of the alternative hypothesis ($H_1$).
   - Example: Testing whether a parameter $\theta = 0$ ($H_0$) versus $\theta \neq 0$ ($H_1$).
2. **Point-Null Hypothesis**: The null hypothesis is a specific point value (e.g., $\theta = 0$).
3. **Well-Defined Prior**: The prior distribution is continuous and well-defined at the null value.

### **Advantages**

1. **Efficiency**: Avoids integrating over the full parameter space, making it computationally efficient.
2. **Simplicity**: Requires only the prior and posterior densities at the null value.
3. **Flexibility**: Can be applied with any prior and posterior distribution as long as densities at the null are defined.

### **Interpretation**

- A large $BF_{10}$ indicates strong evidence in favor of the alternative hypothesis ($H_1$).
- A small $BF_{10}$ (or a large $BF_{01}$) indicates strong evidence in favor of the null hypothesis ($H_0$).

For example:

- If $BF_{10} = 3$, the data are three times more likely under $H_1$ than under $H_0$.
- If $BF_{10} = 0.5$ (or equivalently $BF_{01} = 2$), the data are twice as likely under $H_0$ than under $H_1$.

### **Limitations**

1. **Point Nulls Only**: This method applies only to point-null hypotheses and cannot be used for interval-null or other complex hypotheses.
2. **Prior Sensitivity**: The result is sensitive to the choice of the prior, especially the prior density at the null value.

### *BF *Example**

Suppose we are testing whether a regression coefficient $\beta = 0$ (null hypothesis) versus $\beta \neq 0$ (alternative hypothesis). Using the Savage-Dickey Density Ratio:

1. Compute the prior density $P(\beta = 0)$ at $\beta = 0$.
2. Compute the posterior density $P(\beta = 0 \mid \text{data})$ at $\beta = 0$.
3. Calculate the Bayes Factor as:

$$
BF_{10} = \frac{P(\beta = 0 \mid \text{data})}{P(\beta = 0)}
$$

The **Savage-Dickey Density Ratio** is a powerful tool for Bayesian hypothesis testing, offering an efficient way to compute Bayes Factors for point-null hypotheses in nested models.

#### `arviz.plot_bf` **Parameters**

- **`idata`**: InferenceData object containing the posterior samples and optionally prior samples.

- **`var_name`**: String specifying the name of the variable to test.

- **`prior`**: Optional array specifying a custom prior distribution for sensitivity analysis. If not provided, the prior from `idata` is used.

- **`ref_val`**: Numeric value representing the point-null hypothesis (default is 0).

- **`colors`**: Tuple specifying colors for the prior and posterior distributions (default is `('C0', 'C1')`).

- **`figsize`**: Tuple specifying the figure size. If `None`, it is defined automatically.

- **`textsize`**: Float for scaling text size of labels and titles. If `None`, it is auto-scaled based on `figsize`.

- **`plot_kwargs`**: Dictionary of additional keyword arguments passed to the plotting function.

- **`hist_kwargs`**: Dictionary of additional keyword arguments passed to the histogram function (for discrete variables).

- **`ax`**: Matplotlib axes object to plot on. If `None`, a new figure and axes are created.

- **`backend`**: String specifying the plotting backend (`'matplotlib'` or `'bokeh'`; default is `'matplotlib'`).

- **`backend_kwargs`**: Dictionary of backend-specific keyword arguments.

- **`show`**: Boolean to display the plot immediately (default is `None`).

#### **Returns**

- **`dict`**: A dictionary containing:
  - **`BF10`**: Bayes Factor in favor of \( H_1 \) over \( H_0 \).
  - **`BF01`**: Bayes Factor in favor of \( H_0 \) over \( H_1 \).

- **`axes`**: Matplotlib Axes or Bokeh Figure object containing the plot.

## Loss Functions

### Loss Functions in Bayesian Analysis

In Bayesian analysis, **loss functions** quantify the cost or penalty associated with making incorrect decisions or predictions based on uncertain data. They are a cornerstone of **decision theory**, translating probabilistic results into actionable decisions by minimizing expected losses.

#### **Key Concepts**

1. **Posterior Distribution**:
   - The posterior distribution represents the probability of model parameters or outcomes given the observed data.
   - Loss functions are applied to the posterior to derive optimal decisions, estimates, or predictions.

2. **Decision Rule**:
   - A decision rule maps the posterior distribution to a specific action or estimate by minimizing the expected loss.
   - The optimal decision depends on the choice of the loss function.

3. **Expected Loss**:
   - The expected loss is the average penalty across all possible outcomes, weighted by their posterior probabilities.
   - Formally:
     $$
     \begin{align}
     \text{Expected Loss} = \int L(\theta, a) \, p(\theta \mid \text{data}) \, d\theta
     \end{align}
     $$
     Where:
     - \(L(\theta, a)\): The loss function measuring the penalty for choosing action \(a\) when the true parameter is \(\theta\).
     - \(p(\theta \mid \text{data})\): The posterior distribution of \(\theta\).

4. **Action**:
   - The "action" can be a prediction, an estimate of a parameter, or any decision based on the analysis.

#### **Common Loss Functions**

1. **Squared Error Loss**:
   - Penalizes the squared difference between the true value (\(\theta\)) and the estimate (\(a\)).
   - Commonly used in regression and estimation problems.
   - **Optimal decision**: The posterior **mean**.
   - Formula:
     $$
     \begin{align}
     L(\theta, a) = (\theta - a)^2
     \end{align}
     $$

2. **Absolute Error Loss**:
   - Penalizes the absolute difference between the true value and the estimate.
   - **Optimal decision**: The posterior **median**.
   - Formula:
     $$
     \begin{align}
     L(\theta, a) = |\theta - a|
     \end{align}
     $$

3. **Zero-One Loss**:
   - Assigns a fixed penalty for making an incorrect decision, regardless of the degree of error.
   - Common in classification problems.
   - **Optimal decision**: The posterior **mode**.
   - Formula:
     $$
     \begin{align}
     L(\theta, a) =
     \begin{cases}
     0 & \text{if } \theta = a \\
     1 & \text{otherwise}
     \end{cases}
     \end{align}
     $$

4. **Custom Loss Functions**:
   - Tailored to specific decision-making contexts where the cost of different types of errors varies.
   - For example, in medical diagnostics, the loss for a false negative may be much higher than for a false positive.

#### **Applications of Loss Functions in Bayesian Analysis**

1. **Parameter Estimation**:
   - Use the posterior distribution to select an estimate (e.g., mean, median, or mode) based on the chosen loss function.

2. **Prediction**:
   - Generate optimal predictions that balance uncertainty and the costs of potential errors.

3. **Model Selection**:
   - Compare models by incorporating both their fit to the data and the cost of decisions implied by their predictions.

4. **Decision-Making Under Uncertainty**:
   - Make choices (e.g., treatment plans, financial investments) that minimize expected loss, accounting for the risks and benefits of each action.

#### **Key Takeaways**

- Loss functions bridge the gap between probabilistic Bayesian inference and practical decision-making.
- The choice of a loss function depends on the context and the costs associated with different types of errors.
- By minimizing expected loss, Bayesian analysis provides a principled framework for making optimal decisions under uncertainty.
