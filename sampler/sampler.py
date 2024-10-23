import numpy as np
from polyagamma import random_polyagamma
from scipy.stats import multivariate_normal


class Sampler:
    def __init__(self, y, X, z, beta, mean_beta,
                 cov_beta, S=10000):
        # Set values of the random variables and initial values of parameters
        self.y = y
        self.X = X
        self.S = S
        self.mean_beta = mean_beta
        self.cov_beta = cov_beta

        self.z = z
        self.beta = beta

        # Store samples
        self.z_samples = np.zeros((self.S, self.X.shape[1]))
        self.beta_samples = np.zeros((self.S, self.X.shape[1]))

    def run_sampler(self):
        raise NotImplementedError

    # Likelihood function
    def likelihood(self, y, beta, z, x):
        res = 0
        for r in range(len(y)):
            theta = np.dot(z, beta * x[r])
            res += y[r] * np.log(np.exp(theta) / (1 + np.exp(theta)))
        return res


class MetropolisHastingsSampler(Sampler):
    """
    Perform Metropolis Hastings for model selection for logistic regression.

    Parameters
    ----------
    y: numpy array with the result in the values of 1s and 0s.
    X: numpy array with the potential variables affecting the values of y
       and a intercept column, it should be centred and scaled.
    likelihood: The likelihood of y used in computing conditional probability
    mean_beta, cov_beta: the parameters in the conditional distribution of beta

    Returns
    -------
    get the computed results by self.beta_samples and self.z_samples

    Notes
    -----
    Here we assume proposal distribution of beta is multivariate normal,
    the covariance mat of the proposal dist of beta can be approximated by
    sigma^2(X^{T} X)^{-1}, which is 'var_prop' in the run_sampler part.
    We also assume that beta and z are independent, with z_j ~ binary(1/2),
    and beta_i ~ normal(mean_beta[i], cor_beta[i])
    """

    def prop_logpdf(self, beta, prop, delta):
        pdf = 0
        for r in range(len(beta)):
            pdf += np.log((beta[r] - (prop[r] - delta)) / (2*delta))
        return pdf

    def run_sampler(self):
        for i in range(self.S):
            # Sample beta one by one
            delta = 1
            for j in range(self.X.shape[1]):
                beta_prop_arr = self.beta.copy()
                beta_prop = np.random.uniform(self.beta[j]-delta,
                                              self.beta[j]+delta)
                beta_prop_arr[j] = beta_prop

                p_beta = multivariate_normal.logpdf(self.beta, self.mean_beta,
                                                    self.cov_beta)
                J_prop_beta = self.prop_logpdf(beta_prop_arr, self.beta, delta)
                p_prop_beta = multivariate_normal.logpdf(beta_prop,
                                                         self.mean_beta,
                                                         self.cov_beta)
                J_beta = self.prop_logpdf(self.beta, beta_prop_arr, delta)

                L_prop_beta = self.likelihood(self.y, beta_prop_arr,
                                              self.z, self.X)

                L_beta = self.likelihood(self.y, self.beta, self.z, self.X)

                r_log_beta = (L_prop_beta + p_prop_beta
                              - L_beta - p_beta) + J_beta - J_prop_beta
                # print(list([L_prop_beta, p_prop_beta,
                # L_beta, p_beta, J_beta, J_prop_beta]), r_log_beta)

                if np.log(np.random.uniform()) < r_log_beta:
                    self.beta = beta_prop_arr
                    # print("1")

            self.beta_samples[i] = self.beta

            # Sample z
            for r in np.random.permutation(range(1, self.X.shape[1])):
                z_prop = self.z.copy()
                z_prop[r] = 1 - z_prop[r]

                L_prop_z = self.likelihood(self.y, self.beta, z_prop, self.X)

                L_z = self.likelihood(self.y, self.beta, self.z, self.X)

                r_log_z = L_prop_z - L_z

                if np.log(np.random.uniform()) < r_log_z:
                    self.z = z_prop
                    # print("2")

            self.z_samples[i] = self.z

        return self.beta_samples, self.z_samples


class PolyaGamma(Sampler):
    """
    Apply Pólya-Gamma for model selection for logistic regression.

    Parameters
    ----------
    y: numpy array with the result in the values of 1s and 0s.
    X: panda DataFrame with the potential variables affecting the values of y.
    mean_beta, cor_beta: the parameters in the conditional distribution of beta

    Returns
    -------
    get the computed results by self.beta_samples and self.z_samples
    """
    def __init__(self, y, X, z, beta,
                 mean_beta, cov_beta, S=10000):
        super().__init__(y, X, z, beta, mean_beta, cov_beta, S)
        self.omega_samples = np.zeros((self.S, len(self.y)))

    def run_sampler(self):
        # Gibbs sampling by Pólya-Gamma
        for i in range(self.S):
            # Sample omega
            omega = np.zeros(len(self.y))
            for r in range(len(self.y)):
                x = self.X[r]
                omega[r] = random_polyagamma(1,
                                             np.dot(x, self.beta), size=1)[0]

            self.omega_samples[i] = omega

            # Sample beta
            V_omega = np.linalg.inv(self.X.T @ np.diag(omega) @ self.X
                                    + np.linalg.inv(self.cov_beta))
            k = self.y - np.ones(len(self.y))/2
            m_omega = V_omega @ (self.X.T @ k + np.linalg.inv(self.cov_beta)
                                 @ self.mean_beta)
            self.beta = np.random.multivariate_normal(m_omega, V_omega)
            self.beta_samples[i] = self.beta

            # Sample z
            for r in np.random.permutation(range(1, self.X.shape[1])):
                z_prop = self.z.copy()
                z_prop[r] = 1 - z_prop[r]

                L_prop_z = self.likelihood(self.y, self.beta, z_prop, self.X)

                L_z = self.likelihood(self.y, self.beta, self.z, self.X)

                r_log_z = L_prop_z - L_z

                if np.log(np.random.uniform()) < r_log_z:
                    self.z = z_prop
                    # print(".")

            self.z_samples[i] = self.z

        return self.beta_samples, self.z_samples
