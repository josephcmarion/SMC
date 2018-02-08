from path_estimators import GeometricPathEstimator
import numpy as np
from samplers import MeanFieldIsingSampler
import matplotlib.pyplot as plt


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