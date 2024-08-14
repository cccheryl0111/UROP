import numpy as np
from scipy.stats import gamma, norm
from scipy.special import gammaln
import statsmodels.api as sm
from polyagamma import random_polyagamma


class Sampler:
    def __init__(self, y, X, z, beta, mean_beta, cov_beta, S=10000):
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


class GibbsSampler(Sampler):
    """
    Perform Gibbs Sampler for model selection for logistic regression.

    Parameters
    ----------
    y : numpy array with the result in the values of 1s and 0s.
    X : numpy array with the potential variables affecting the values of y
        and a intercept column, it should be centred and scaled.
    g : the parameter in g-prior
    nu0, sigma01 : the parameters in the distribution of sigma^2

    Returns
    -------
    get the computed results by self.beta_samples and self.z_samples

    Notes
    -----
    Assume 1/sigma^2 ~ gamma(nu0 / 2, nu0 * sigma^2 / 2),
    we get {sigma^2|y, X} ~ inverse-gamma([nu0+n]/2, [nu0 * sigma^2 + SSRg]/2),
    and assume {beta|X, sigma^2} ~ MVN(0, g * sigma^2[X^T @ X]^{-1}),
    which are used in iteation step.
    We also get from the assumption above the log likelihood of y used
    to compare whether to accept the proposed z.
    """
    def __init__(self, y, X, z, beta, mean_beta, cov_beta,
                 g, nu0, sigma02, S=10000):
        super().__init__(y, X, z, beta, mean_beta, cov_beta, S)
        self.g = g
        self.nu0 = nu0
        self.sigma02 = sigma02

    def lp_y_X(self):
        n = self.X.shape[0]
        p = self.X.shape[1]

        if p == 0:
            Hg = 0
            s20 = np.mean(self.y ** 2)
        else:
            Hg = self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
            s20 = sm.OLS(self.y, self.X).fit().mse_resid

        SSRg = self.y.T @ (np.identity(n)
                           - (self.g/(self.g+1)) * Hg) @ self.y

        log_prob = -0.5 * (n * np.log(np.pi) + p * np.log(1 + self.g)
                           + (self.nu0 + n) * np.log(self.nu0 * s20 + SSRg)
                           - self.nu0 * np.log(self.nu0*s20))
        + gammaln((self.nu0 + n) / 2) - gammaln(self.nu0 / 2)
        return log_prob

    def run_sampler(self):
        used = GibbsSampler(self.y, self.X[:, self.z == 1],
                            self.g, self.nu0, self.sigma02)
        lp_y_c = used.lp_y_X()
        for s in range(self.S):
            # update z
            for j in np.random.permutation(range(1, self.X.shape[1])):
                zp = self.z.copy()
                zp[j] = 1 - zp[j]
                used_p = GibbsSampler(self.y, self.X[:, zp == 1],
                                      self.g, self.nu0, self.sigma02)
                lp_y_p = used_p.lp_y_X()

                r = (lp_y_p - lp_y_c) * ((-1) ** (zp[j] == 0))
                self.z[j] = np.random.binomial(1, 1 / (1 + np.exp(-r)))

                if self.z[j] == zp[j]:
                    lp_y_c = lp_y_p

            self.z_samples[s, :] = self.z

            # Update sigma(???)
            X_z = self.X[:, self.z == 1]
            y_z = self.y
            Hg_z = X_z @ np.linalg.inv(X_z.T @ X_z) @ X_z.T
            SSRg_z = y_z.T @ (np.identity(len(X_z))
                              - (self.g/(self.g+1)) * Hg_z) @ y_z
            shape = (self.nu0 + len(X_z)) / 2
            scale = (self.nu0 * self.sigma02 + SSRg_z) / 2
            gamma2 = 1 / gamma.rvs(a=shape, scale=scale, size=1)

            # Update beta
            cor = self.g * gamma2 * np.linalg.inv(X_z.T @ X_z)
            self.beta[self.z == 1] = np.random.multivariate_normal(
                np.zeros(X_z.shape[1]), cor)
            self.beta_samples[s] = self.beta

        return self.beta_samples, self.z_samples


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
    def __init__(self, y, X, z, beta,
                 mean_beta, cov_beta, likelihood, S=10000):

        super().__init__(y, X, z, beta, mean_beta, cov_beta, S)
        self.likelihood = likelihood

    def run_sampler(self):
        for i in range(self.S):
            # Sample beta one by one
            delta = 40
            for j in range(self.X.shape[1]):
                beta_prop_arr = self.beta.copy()
                beta_prop = np.random.uniform(self.beta[j]-delta,
                                              self.beta[j]+delta)
                beta_prop_arr[j] = beta_prop

                p_beta = norm.logpdf(self.beta[j], self.mean_beta[j],
                                     self.cov_beta[j])
                J_prop_beta = np.log((beta_prop - (self.beta[j]
                                                   - delta)) / (2*delta))
                p_prop_beta = norm.logpdf(beta_prop, self.mean_beta[j],
                                          self.cov_beta[j])
                J_beta = np.log((self.beta[j] - (beta_prop
                                                 - delta)) / (2*delta))

                L_prop_beta = np.log(self.likelihood(
                    sum(beta_prop_arr * self.z * self.X[j]))) * self.y[j]
                + np.log(1 - self.likelihood(
                    sum(beta_prop_arr * self.z * self.X[j]))) * (1 - self.y[j])

                print(beta_prop_arr * self.z * self.X[j])

                L_beta = np.log(self.likelihood(
                    sum(self.beta * self.z * self.X[j]))) * self.y[j]
                + np.log(1 - self.likelihood(
                    sum(self.beta * self.z * self.X[j]))) * (1 - self.y[j])

                r_log_beta = L_prop_beta + p_prop_beta - L_beta
                - p_beta + J_beta - J_prop_beta

                if np.log(np.random.uniform()) < min(0, r_log_beta):
                    self.beta = beta_prop_arr

            self.beta_samples[i] = self.beta

            # Sample z
            for r in np.random.permutation(range(1, self.X.shape[1])):
                z_prop = self.z.copy()
                z_prop[r] = 1 - z_prop[r]

                L_prop_z = sum(np.log(self.likelihood(
                    np.sum(self.beta * z_prop * self.X, axis=1)) ** self.y
                    * (np.ones(len(self.y)) - self.likelihood(
                        np.sum(self.beta * z_prop
                               * self.X, axis=1))) ** (1 - self.y)))

                L_z = sum(np.log(self.likelihood(
                    np.sum(self.beta * self.z * self.X, axis=1)) ** self.y
                    * (np.ones(len(self.y)) - self.likelihood(
                        np.sum(self.beta * self.z
                               * self.X, axis=1))) ** (1 - self.y)))

                r_log_z = L_prop_z - L_z

                if np.log(np.random.uniform()) < min(0, r_log_z):
                    self.z = z_prop

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
                 mean_beta, cov_beta, likelihood, S=10000):
        super().__init__(y, X, z, beta, mean_beta, cov_beta, S)
        self.likelihood = likelihood
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

                L_prop_z = sum(np.log(self.likelihood(
                    np.sum(self.beta * z_prop * self.X, axis=1)) ** self.y
                    * (np.ones(len(self.y)) - self.likelihood(
                        np.sum(self.beta * z_prop
                               * self.X, axis=1))) ** (1 - self.y)))

                L_z = sum(np.log(self.likelihood(
                    np.sum(self.beta * self.z * self.X, axis=1)) ** self.y
                    * (np.ones(len(self.y)) - self.likelihood(
                        np.sum(self.beta * self.z
                               * self.X, axis=1))) ** (1 - self.y)))

                r_log_z = L_prop_z - L_z

                if np.log(np.random.uniform()) < min(0, r_log_z):
                    self.z = z_prop

            self.z_samples[i] = self.z

        return self.beta_samples, self.z_samples
