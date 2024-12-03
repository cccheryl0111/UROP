# Abstract
Model selection is a crucial tool for avoiding the overfitting problem and simplifying the statistical model for the dataset with various potential covariates. We have explored a specific model selection method for the logistic regression model that uses data augmentation via the Pólya-Gamma distribution. We also compared it with a general and simple Markov chain Monte Carlo method, Metropolis-Hastings algorithm.

Given the possibility of a high rejection probability and autocorrelation for the Metropolis-Hastings algorithm, as well as some constrains on applying the new data augmentation method, we evaluated the performance of both methods with multiple chains under different evaluation criteria for a set of binary data. Our findings suggest that while samples from both methods converge after 10000 draws, the result of the new data augmentation is more stable, with a higher effective sample size and lower autocorrelation. This comparison introduces a new model selection method for the logistic regression model.

# Algorithm for Pólya-Gamma Variable Selection
1. sample $\omega_i^{(s+1)}$ from $PG(1,\theta_i)$;
2. sample $\beta^{(s+1)}$ from $N(m_{\omega^{(s+1)}},V_{\omega^{(s+1)}})$;
3. Update __z__ by Metropolis-Hastings algorithm:
   
   a. set proposed $\mathbf{z} = \mathbf{z}^{(s)}$;
   
   b. for $j \in \{1,... ,n\}$ in random order, set $z_j^{*} = 1 - z_j$;
   
   c. determine whether to replace $z_j$ by $z_j^{*}$ with the probability in the acceptance ratio under $\beta^{(s+1)}$;
   
   d. set $\mathbf{z}^{(s+1)} = \mathbf{z}$.
5. let $\phi^{(s+1)} = \{\boldsymbol{\beta}^{(s+1)}, \mathbf{z}^{(s+1)}\}$.
