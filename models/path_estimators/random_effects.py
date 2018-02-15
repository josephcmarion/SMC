import numpy as np
from models.samplers.random_effects import RandomEffectsSampler
from smc_path_estimators import GeometricPathEstimator, GeometricTemperedEstimator


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