import numpy as np
from scipy.special import expit

from models.samplers.logistic_regression import LogisticRegressionSampler, LogisticPriorPathSampler, \
    GeometricTemperedLogisticSampler
from path_estimators import PathEstimator, GeometricTemperedEstimator
from utils import gaussian_kernel


class LogisticRegressionEstimator(PathEstimator, LogisticRegressionSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance):
        """ Path sampling estimator for logistic regression. Moves from the prior to model of interest via a
        tempered geometric mixture. Markov kernels are done via nuts in STAN

        attributes
        ----------
        X: np.array
            (N, D) array of covariates, should include the intercept column of ones
        Y: np.array
            (N, ) vector of class labels in {0,1}^N
        prior_mean: np.array
            (D, ) vector of prior means
        prior_covariance: np.array
            (D, D) positive definite prior covariance matrix
        """

        LogisticRegressionSampler.__init__(self, X, Y, prior_mean, prior_covariance)
        PathEstimator.__init__(self, 2)

        # GeometricTemperedLogisticSampler.__init__(self, X, Y, prior_mean, prior_covariance, **sampler_kwargs)
        # GeometricTemperedEstimator.__init__(self, **estimator_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            likelihood inverse temperature in [0,1] and prior mixture parameter in [0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, inverse_temperature = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0 - 10 ** -6
        pi[pi == 0.0] = 10 ** -6

        log_likelihood = (self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)).sum(1)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)

        # compute potentials
        potential_beta = inverse_temperature*log_likelihood
        potential_temperature = beta*log_likelihood + log_prior

        return np.vstack([potential_beta, potential_temperature]).T


class PriorPathLogisticEstimator(PathEstimator, LogisticPriorPathSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance):
        """ Path sampling estimator for logistic regression. Moves from the prior to model of interest via a
        tempered geometric mixture. Markov kernels are done via nuts in STAN

        attributes
        ----------
        X: np.array
            (N, D) array of covariates, should include the intercept column of ones
        Y: np.array
            (N, ) vector of class labels in {0,1}^N
        prior_mean: np.array
            (D, ) vector of prior means
        prior_covariance: np.array
            (D, D) positive definite prior covariance matrix

        """

        LogisticPriorPathSampler.__init__(self, X, Y, prior_mean, prior_covariance)
        PathEstimator.__init__(self, 2)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, alpha = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0 - 10 ** -6
        pi[pi == 0.0] = 10 ** -6

        log_likelihood = (self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)).sum(1)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)
        log_posterior = -0.5 * gaussian_kernel(samples, self.posterior_mean, self.posterior_precision)

        # compute potentials
        potential_beta = log_likelihood
        potential_alpha = log_prior + log_posterior

        return np.vstack([potential_beta, potential_alpha]).T


class GeometricTemperedLogisticEstimator(GeometricTemperedEstimator, GeometricTemperedLogisticSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance, estimator_kwargs={}, sampler_kwargs={}):
        """ Path sampling estimator for logistic regression. Moves from the prior to model of interest via a
        tempered geometric mixture. Markov kernels are done via nuts in STAN

        attributes
        ----------
        X: np.array
            (N, D) array of covariates, should include the intercept column of ones
        Y: np.array
            (N, ) vector of class labels in {0,1}^N
        prior_mean: np.array
            (D, ) vector of prior means
        prior_covariance: np.array
            (D, D) positive definite prior covariance matrix

        parameters
        ----------
        sampler_kwargs: dict
            additional arguments to TemperedRandomEffectsSampler. options include 'seed' or 'load'
        estimator_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        GeometricTemperedLogisticSampler.__init__(self, X, Y, prior_mean, prior_covariance, **sampler_kwargs)
        GeometricTemperedEstimator.__init__(self, **estimator_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, 2) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta, temperature = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0 - 10 ** -6
        pi[pi == 0.0] = 10 ** -6

        log_likelihood = (self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)).sum(1)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)
        log_intial = -0.5 * gaussian_kernel(samples, self.initial_mean, self.initial_precision)

        # compute potentials
        potential_beta = temperature*(log_likelihood + log_prior - log_intial)
        potential_temperature = beta*(log_likelihood + log_prior) + (1-beta)*log_intial

        return np.vstack([potential_beta, potential_temperature]).T

