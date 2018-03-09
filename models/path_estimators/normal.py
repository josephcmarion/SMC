import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

from models.samplers.normal import NormalPathSampler, RegressionSelectionSampler
from smc_path_estimators import GeometricTemperedEstimator
from utils import gaussian_kernel


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
            sns.heatmap(true_energy[:, :, q][::-1], cmap=cmap, **plot_kwargs[q])
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


class RegressionSelectionEstimator(GeometricTemperedEstimator, RegressionSelectionSampler):

    def __init__(self, n_observations, n_dimensions, prior_sd, design_correlation=0.3, coefficient_sd=1.0,
                 seed=1337, path_sampler_kwargs={}):
        """Path sampling module for comparing a linear model with one set of features
        to a linear model with another set of features. Uses a tempered geometric mixture.
        Markov kernels are done via independent sampling. Inherits SMC sampling routines from
        smc.samplers.NormalPathSampler

        The likelihood is:

        Y ~ X*coefficients_true + epsilon

        where epsilon is N(0, 1) noise. We use the following prior

        coefficients ~ N(0, 1)

        The X's are drawn from a highly correlated normal, so that the many models are equally reasonable.

        attributes
        ----------
        n_observations: int
            number of observations to generate
        n_dimensions: even int
            number of covariate dimensions to use
        prior_sd: float > 0
            prior standard deviation on the covariates
        design_correlation: float in (-1.0, 0.5)
            roughly speaking, this is the correlation between adjacent columns in the design matrix
        coefficient_sd: float > 0
            standard deviation to use when sampling the coefficients
        seed: float
            used to control the RNG. If set to None does not use a seed
        path_sampler_kwargs: dict
            additional arguments to be passed to GeometricTemperedEstimator. options inclue
            min_temp, beta_spacing and
        """

        RegressionSelectionSampler.__init__(self, n_observations, n_dimensions, prior_sd, design_correlation,
                                            coefficient_sd, seed)
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

        beta, temperature = params

        # compute the log pdf for each distribution
        residuals0 = self.Y[None, :] - np.dot(samples, np.dot(self.X, self.gamma0).T)
        log_density0 = -0.5 * (residuals0 ** 2).sum(1)

        residuals1 = self.Y[None, :] - np.dot(samples, np.dot(self.X, self.gamma1).T)
        log_density1 = -0.5 * (residuals1 ** 2).sum(1)

        log_prior = (samples ** 2).sum(1) / self.prior_sd ** 2

        # compute potentials
        potential_beta = temperature*(-log_density0 + log_density1)
        # todo: change this back
        potential_temperature = (1.0 - beta) * log_density0 + beta * log_density1  # + log_prior

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
            sns.heatmap(true_energy[:, :, q][::-1], cmap=cmap, **plot_kwargs[q])
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
        mean0, precision0, covariance0 = self._get_normal_parameters((0.0, 1.0))
        mean1, precision1, covariance1 = self._get_normal_parameters((1.0, 1.0))

        part1 = 0.5*(np.linalg.slogdet(2*np.pi*covariance1)[1] - np.linalg.slogdet(2*np.pi*covariance0)[1])
        part2 = 0.5*(np.dot(np.dot(mean1.T, precision1), mean1.T) - np.dot(np.dot(mean0.T, precision0), mean0.T))

        return part1 + part2
