from path_samplers import PathEstimator, GeometricPathEstimator, GeometricTemperedEstimator
import numpy as np
import scipy.stats as stats
from samplers import NormalPathSampler, MeanFieldIsingSampler, LogisticRegressionSampler, LogisticPriorPathSampler, \
    RandomEffectsSampler, GeometricTemperedLogisticSampler
from utils import gaussian_kernel
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns


class NormalPathEstimator(GeometricTemperedEstimator, NormalPathSampler):

    def __init__(self, mean1, mean2, covariance1, covariance2, path_sampler_kwargs={}):
        """Path sampling module for the geometric-tempered D-dimensional normal. Inherits SMC sampling routines from
        smc.samplers.NormalPathSampler

        attributes
        ----------
        mean1: np.array
            (D, ) mean vector for the initial normal distribution
        mean2: np.array
            (D, ) mean vector for the target normal distribution
        covariance1:  np.array
            (D,D) positive definite covariance matrix for the initial distribution
        covariance2:  np.array
            (D,D) positive definite covariance matrix for the initial distribution
        path_sampler_kwargs: dict
            additional arguments to be passed to GeometricTemperedEstimator. options inclue
            min_temp, beta_spacing and
        """

        NormalPathSampler.__init__(self, mean1, mean2, covariance1, covariance2)
        GeometricTemperedEstimator.__init__(self, **path_sampler_kwargs)

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

        beta, inverse_temperature = params

        # pre-compute the kernels
        kernel1 = gaussian_kernel(samples, self.mean1, self.precision1)
        kernel2 = gaussian_kernel(samples, self.mean2, self.precision2)

        # compute potentials
        potential_beta = -0.5*inverse_temperature*(kernel2-kernel1)
        potential_temperature = -0.5*((1.0-beta)*kernel1 + beta*kernel2)

        return np.vstack([potential_beta, potential_temperature]).T

    def _true_energy(self, path, N=10**5):
        """ uses independent gaussian sampling to estimate the true energy values at the specified parameters

        parameters
        ----------
        path: list of tuples
            parameters used at each sampling step. first value is a mixing parameter in [0,1]
            second value is inverse temperature parameter in (0,1]
        N: int
            number of samples to use in estimation

        return
        ------
        np.array
            (N, 4) energy estimates (beta*beta, beta*inverse_temperature, inverse_temperature*inverse_temperature)
        """

        # statistics
        sufficient_statistics = [self._get_normal_parameters(params) for params in path]
        samples = np.array([stats.multivariate_normal(mean=mean, cov=cov).rvs(N)
                            for mean, precision, cov in sufficient_statistics])

        return self._estimate_energy(samples, path)

    def plot_true_energy_map(self, N=10**3, plot_kwargs=[{}]*3):
        """ plots the estimated energy map for the sampler on each of the three dimensions.
        Uses a large number of independent gaussian samples to estimate the true energy

        parameters
        ----------
        N: int
            number of samples to use in estimation
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta, n_temperature = 101, 101

        # create grids and estimate energy
        betas, temperatures = self._get_grids([n_beta, n_temperature])
        path_array = np.array(map(lambda x: x.flatten(), np.meshgrid(betas, temperatures))).T
        true_energy = self._true_energy(path_array, N).reshape(n_temperature, n_beta, 3)

        # setup some stuff for the axis and titles
        beta_locs = np.linspace(0, n_beta, 11).astype(int)[:-1]
        beta_labels = ['{0:.1f}'.format(beta) for beta in betas[beta_locs]]

        temp_locs = np.linspace(0, n_temperature, 11).astype(int)[:-1]
        temp_labels = ['{0:.1f}'.format(temp) for temp in temperatures[temp_locs]]

        titles = [
            r'True $\beta -\beta$ path variance',
            r'True $\beta -t$ path variance',
            r'True $t -t$ path variance'
        ]

        # do the plotting
        plt.rcParams['figure.figsize'] = 15, 3
        cmap = sns.cubehelix_palette(as_cmap=True)
        for q in range(3):
            plt.subplot(1, 3, q + 1)
            sns.heatmap(true_energy[:, :, q][::-1], cmap=cmap, vmin=0, **plot_kwargs[q])
            plt.xticks(beta_locs, beta_labels)
            plt.xlabel(r'$\beta$')

            plt.yticks(temp_locs, temp_labels)
            plt.ylabel(r'$t$')
            plt.title(titles[q])

    def true_lambda(self):
        """ returns the true log ratio of normalizing constants

        returns
        -------
        float:
            the true log ratio of normalizing constants
        """
        return 0.5*(np.linalg.slogdet(2*np.pi*self.covariance2)[1] - np.linalg.slogdet(2*np.pi*self.covariance1)[1])


class IsingPathEstimator(GeometricPathEstimator, MeanFieldIsingSampler):

    def __init__(self, dimension, alpha, path_sampler_kwargs={}):
        """ Path sampling estimator for the ising model. Uses a tempered path from the uniform distribution
        to the distribution of interest.

        attributes
        ----------
        dimension: int
            the number of sites/nodes in the graph
        alpha: float > 0
            the temperature of the model, determining the behaviour

        parameters
        ----------
        path_sampler_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        MeanFieldIsingSampler.__init__(self, dimension, alpha)
        GeometricPathEstimator.__init__(self, **path_sampler_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples
        params: tuple
            first value is the inverse temperature parameter in [0,1]

        return
        ------
        np.array
            (N, 1) vector of potentials w.r.t the mixing parameter and inverse temperature parameter
        """

        beta = params[0]

        # pre-compute the kernels
        potential = self.alpha/self.dimension*(samples.sum(1)**2).reshape(-1, 1)

        return potential

    def _true_energy(self, path):
        """ computes the exact energy at each step along a path

        parameters
        ----------
        path: list of tuples
            inverse temperature parameters used at each sampling step

        return
        ------
        np.array
            (N, Q) energy estimates
        """

        energies = np.zeros(len(path))

        # for each step along the path
        for i, params in enumerate(path):

            probabilities, magnetism = self._true_probabilities(params)
            energies[i] = np.cov(self.alpha/self.dimension*magnetism**2, aweights=probabilities)

        return energies

    def plot_true_energy_map(self, plot_kwargs={}):
        """ plots the estimated energy map for the sampler on each of the three dimensions

        parameters
        ----------
        plot_kwargs: list of dict
            length three dictionary, each containing arguments to be passed to a subplot
        """

        # I might add this as an argument but it seems clumsy
        n_beta = 1001

        # create grids and estimate energy
        betas = self._get_grids([n_beta])[0]
        true_energy = self._true_energy([(beta,) for beta in betas])

        plt.plot(betas, true_energy, color='red',  label='true', **plot_kwargs)
        plt.xlabel(r'$\beta$')
        plt.ylabel('Energy')

    def true_lambda(self):
        """ returns the true log ratio of normalizing constants

        returns
        -------
        float:
            the true log ratio of normalizing constants
        """

        target_z = self._true_normalizing_constant((1.0,))
        initial_z = self._true_normalizing_constant((0.0,))

        return np.log(target_z) - np.log(initial_z)


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


class LogisticPriorPathEstimator(PathEstimator, LogisticPriorPathSampler):

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


class GeometricRandomEffectsEstimator(GeometricPathEstimator, RandomEffectsSampler):

    def __init__(self, n_observations, n_groups, alpha, sigma, tau, estimator_kwargs={}, sampler_kwargs={}):
        """ Path sampling estimator for the ising model. Uses a tempered path from the uniform distribution
        to the distribution of interest.

        attributes
        ----------
        n_observations: int
            number of observations to generate
        n_groups: int
            number of groups to use
        alpha: float
            intercept term
        sigma: float > 0
            observation deviation
        tau: float > 0
            random effects standard deviation

        parameters
        ----------
        sampler_kwargs: dict
            additional arguments to TemperedRandomEffectsSampler. options include 'seed' or 'load'
        estimator_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        RandomEffectsSampler.__init__(self, n_observations, n_groups, alpha, sigma, tau,
                                      path_type='geometric', **sampler_kwargs)
        GeometricPathEstimator.__init__(self, **estimator_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            contains geometric mixture parameter in [0,1]

        return
        ------
        np.array
            (N, ) vector of potentials
        """

        beta = self._params_to_params(params)[0]
        alphas, random_effects, taus, sigmas = self._unpackage_samples(samples)

        # for the initial distribution
        deltas = (self.observations[None, :] - alphas[:, None])/sigmas[:, None]
        potential_initial = 0.5*(deltas**2).sum(1)

        # for the target distribution
        deltas = (self.observations[None, :] - alphas[:, None] - random_effects[:, self.groups])/sigmas[:, None]
        potential_target = -0.5*(deltas**2).sum(1)

        return (potential_initial+potential_target).reshape(-1, 1)


class GeometricTemperedRandomEffectsEstimator(GeometricTemperedEstimator, RandomEffectsSampler):

    def __init__(self, n_observations, n_groups, alpha, sigma, tau, estimator_kwargs={}, sampler_kwargs={}):
        """ Path sampling estimator for the ising model. Uses a tempered path from the uniform distribution
        to the distribution of interest.

        attributes
        ----------
        n_observations: int
            number of observations to generate
        n_groups: int
            number of groups to use
        alpha: float
            intercept term
        sigma: float > 0
            observation deviation
        tau: float > 0
            random effects standard deviation

        parameters
        ----------
        sampler_kwargs: dict
            additional arguments to TemperedRandomEffectsSampler. options include 'seed' or 'load'
        estimator_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        RandomEffectsSampler.__init__(self, n_observations, n_groups, alpha, sigma, tau,
                                      path_type='geometric-tempered', **sampler_kwargs)
        min_temp = 6.0 / (self.n_observations + 2)
        GeometricTemperedEstimator.__init__(self, min_temp=min_temp, **estimator_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            contains geometric mixture parameter in [0,1]

        return
        ------
        np.array
            (N, ) vector of potentials
        """

        beta, temperature = self._params_to_params(params)[:2]
        alphas, random_effects, taus, sigmas = self._unpackage_samples(samples)

        # for the initial distribution
        deltas = (self.observations[None, :] - alphas[:, None])/sigmas[:, None]
        potential_initial = -0.5*(deltas**2).sum(1)

        # for the target distribution
        deltas = (self.observations[None, :] - alphas[:, None] - random_effects[:, self.groups])/sigmas[:, None]
        potential_target = -0.5*(deltas**2).sum(1)

        # potential
        potential_beta = (-potential_initial+potential_target)*temperature
        potential_temperature = (1-beta)*potential_initial + beta*potential_target

        return np.vstack([potential_beta, potential_temperature]).T


class RelaxedRandomEffectsEstimator(GeometricTemperedEstimator, RandomEffectsSampler):

    def __init__(self, n_observations, n_groups, alpha, sigma, tau, estimator_kwargs={}, sampler_kwargs={}):
        """ Path sampling estimator for the random effects model. Uses a geometric path with prior tightening.

        attributes
        ----------
        n_observations: int
            number of observations to generate
        n_groups: int
            number of groups to use
        alpha: float
            intercept term
        sigma: float > 0
            observation deviation
        tau: float > 0
            random effects standard deviation

        parameters
        ----------
        sampler_kwargs: dict
            additional arguments to TemperedRandomEffectsSampler. options include 'seed' or 'load'
        estimator_kwargs: dict
            arguments to be passed to GeometricPathEstimator, generally just 'grid_type'
        """

        RandomEffectsSampler.__init__(self, n_observations, n_groups, alpha, sigma, tau,
                                      path_type='prior-relaxing', **sampler_kwargs)
        GeometricTemperedEstimator.__init__(self, **estimator_kwargs)

    def _potential(self, samples, params):
        """ computes the potential w.r.t path parameters

        parameters
        ----------
        samples: np.array
            (N, D) array of samples
        params: tuple
            contains geometric mixture parameter in [0,1]

        return
        ------
        np.array
            (N, ) vector of potentials
        """

        beta, temperature, relax = self._params_to_params(params)
        alphas, random_effects, taus, sigmas = self._unpackage_samples(samples)

        # for the initial distribution
        deltas = (self.observations[None, :] - alphas[:, None]) / sigmas[:, None]
        potential_initial = -0.5 * (deltas ** 2).sum(1)

        # for the target distribution
        deltas = (self.observations[None, :] - alphas[:, None] - random_effects[:, self.groups]) / sigmas[:, None]
        potential_target = -0.5 * (deltas ** 2).sum(1)

        # for the tightening distribution
        log_pdf_re_prior = -0.5*(deltas**2).sum(1)

        # potential
        potential_beta = (-potential_initial + potential_target) * temperature
        potential_relax = log_pdf_re_prior

        return np.vstack([potential_beta, potential_relax]).T


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
        log_intial = -0.5 * gaussian_kernel(samples, self.initial_mean, self.initial_precisional)

        # compute potentials
        potential_beta = temperature*(log_likelihood + log_prior - log_intial)
        potential_temperature = beta*(log_likelihood + log_prior) + (1-beta)*log_intial

        return np.vstack([potential_beta, potential_temperature]).T
