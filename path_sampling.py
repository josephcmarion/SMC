import numpy as np
import scipy.stats as stats
import GPy
from samplers import NormalPathSampler, MeanFieldIsingSampler
from utils import gaussian_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.misc import comb


class PathEstimator:

    def __init__(self, Q):
        """I'm not totally sure what this class will do but I think it makes sense as a class

        parameters
        ----------
        Q: int
            dimension of the potential
        """

        self.Q = Q

        # place holder stuff for the GP regression
        self.pca = None
        self.scaler = None
        self.gps = []
        self.dependent = False

    def _estimate_energy(self, samples, path):
        """ estimates the energy at each point in the path using output from the model.
        Samples are drawn using self.sampling with save_all_samples=True

        parameters
        ----------
        samples: np.array
            array of samples from self.sampling
        path: list of tuples
            parameters used at each sampling step.

        return
        ------
        np.array
            (N, 4) energy estimates (beta*beta, beta*inverse_temperature, inverse_temperature*inverse_temperature)
        """

        # initialize storage and indices
        energy = np.zeros((samples.shape[0], np.int(comb(self.Q + 1, 2))))
        indices = np.triu_indices(self.Q)

        for i, (x, params) in enumerate(zip(samples, path)):
            potential = self._potential(x, params)
            energy[i] = np.cov(potential.T).reshape(self.Q,self.Q)[indices]

        return energy

    def fit_energy(self, samples_list, path_list, dependent=False):
        """ fits gaussian process to estimate component energy at different parameter configurations.
        Uses samples and paths from various runs of self.sampling

        parameters
        ----------
        samples_list: list of np.array
            a list of sample outputs from multiple sequential monte carlo runs
        path_list: list of list of tuples
            a list of paths corresponding to the sample estimates
        dependent: boolean
            if true, uses PCA to fit dependent GPs.
        """

        # first, convert the samples and path into energy estimates and spatial coordinates
        self.dependent = dependent
        self.gps = []

        xs = np.vstack(np.array(path_list))
        energy = np.vstack(map(lambda x, y: self._estimate_energy(x, y), samples_list, path_list))

        # if necessary do scaling and PCA
        if self.dependent:

            # scale data
            self.scaler = StandardScaler()
            ys_scale = self.scaler.fit_transform(energy)

            # rotate data
            self.pca = PCA(n_components=energy.shape[1], whiten=True)
            ys = self.pca.fit_transform(ys_scale)

        else:
            ys = energy

        # now fit the gps
        for q in range(ys.shape[1]):
            print '\nFitting GP #{}'.format(q)
            kernel = GPy.kern.RBF(input_dim=self.Q)
            model = GPy.models.GPRegression(xs, ys[:, q].reshape(-1, 1), kernel)
            model.optimize_restarts(num_restarts=10)
            self.gps.append(model.copy())

    def predict_energy(self, param_array):
        """ predicts component-wise energy at various parameter locations using trained gps

        parameters
        ----------
        param_array: np.array
            (M, Q) array of path parameters. Each row should correspond to a path parameter for the model

        """

        # make sure the gps have been fit
        if len(self.gps) == 0:
            raise RuntimeError('''no trained gps present, run self.fit_energy''')

        # predict
        predictions = np.array([gp.predict(param_array)[0].flatten() for gp in self.gps]).T
        if self.dependent:
            predictions = self.scaler.inverse_transform(
                self.pca.inverse_transform(predictions)
            )

        return predictions


class NormalPathEstimator(PathEstimator, NormalPathSampler):

    def __init__(self, mean1, mean2, covariance1, covariance2):
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
        """

        NormalPathSampler.__init__(self, mean1, mean2, covariance1, covariance2)
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


class IsingPathEstimator(PathEstimator, MeanFieldIsingSampler):

    def __init__(self, dimension, alpha):
        """ Path sampling estimator for the ising model. Uses a tempered path from the uniform distribution
        to the distribution of interest.

        attributes
        ----------
        dimension: int
            the number of sites/nodes in the graph
        alpha: float > 0
            the temperature of the model, determining the behaviour
        """

        MeanFieldIsingSampler.__init__(self, dimension, alpha)
        PathEstimator.__init__(self, 1)

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

    def _true_probabilities(self, params):
        """ returns the true probabilities of the total magnetism for a specified parameter setting

        parameters
        ----------
        params: tuple
            first value is the inverse temperature parameter in [0,1]

        returns
        -------
        np.array
            (dimension+1,) array of magnetism from most negative to most positive
        np.array
            (dimension+1,) probabilities of each magnetism
        """

        beta = params[0]
        probabilities = np.zeros(self.dimension+1)
        magnetism = np.zeros(self.dimension+1)

        # compute probabilities
        for d in range(self.dimension+1):
            magnetism[d] = -self.dimension + 2*d
            probabilities[d] = comb(self.dimension, d)*np.exp(beta*self.alpha / self.dimension * magnetism[d]**2)

        # normalize
        probabilities /= probabilities.sum()

        return probabilities, magnetism

    def _total_variation(self, samples, params):
        """ returns the total variation distance between the empirical distribution and the true distribution
        for a specified inverse temperature

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples from a single step of sequential monte carlo
        params: tuple
            first value is the inverse temperature parameter in [0,1]

        returns
        -------
        float
            total variation distance in (0,1). Smaller is better
        """

        true_probabilities, magnetism = self._true_probabilities(params)
        empirical_probabilities = np.array([(samples.sum(1) == m).mean() for m in magnetism])

        return np.abs(true_probabilities-empirical_probabilities).sum()/2.0
