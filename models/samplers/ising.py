# general imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import expit, comb
from smc_samplers import SMCSampler


class MeanFieldIsingSampler(SMCSampler):

    def __init__(self, dimension, alpha):
        """ sequential monte carlo sampler for the mean field Ising model. Uses tempering to move from the uniform
        distribution to the temperature of interest (alpha). Markov transitions accomplished via Gibb's sampling.

        This implementation scales really well in N and the number of steps. Increasing the dimension can be problematic
        as the markov kernels have an nested for loop. Moving this to cython could improve the performance noticeably

        attributes
        ----------
        dimension: int
            the number of sites/nodes in the graph
        alpha: float > 0
            the temperature of the model, determining the behaviour
        """

        self.dimension = dimension
        self.alpha = np.float(alpha)

        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _log_pdf(self, samples, params):
        """ log_pdf of the tempered mean field ising model

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples taking values in {-1, 1}
        params: tuple
            tuple containing the inverse temperature in (0,1)

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta = params[0]

        # compute the pdf
        log_density = beta*self.alpha/self.dimension*(samples.sum(1)**2)

        return log_density

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a flattened normal distribution

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            tuple containing the inverse temperature in (0,1)


        returns
        -------
        np.array
            (N, dimension) matrix of normally distributed samples
        """

        beta = params[0]

        # init some things
        samples = -1 + 2 * np.random.choice(2, (N, self.dimension), replace=True)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples taking values in {-1, 1}
        params: tuple
            tuple containing the inverse temperature in (0,1)
        kernel_steps: int
            number of full gibbs scans to complete

        return
        ------
        np.array
            (N, dimension) array of updated samples
        """

        # init some things
        beta = params[0]
        N = samples.shape[0]
        magnetism = samples.sum(1)

        for k in range(kernel_steps):
            for d in range(self.dimension):
                # compute probability of state 1
                energy = (self.alpha * beta / self.dimension) * (
                    (magnetism - samples[:, d] + 1.0) ** 2 - (magnetism - samples[:, d] - 1.0) ** 2)
                probabilities = expit(energy)  # probability of transitioning to +1

                # propose state +1
                accept = stats.uniform().rvs(N) < probabilities
                new_nodes = -1 + 2 * accept

                # update the model
                # this method seems a bit complicated but I'm trying to avoid summing at each step
                magnetism = magnetism - samples[:, d] + new_nodes
                samples[:, d] = new_nodes.copy()  # I'm being a bit anal here

        return samples

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
        probabilities = np.zeros(self.dimension + 1)
        magnetism = np.zeros(self.dimension + 1)

        # compute probabilities
        for d in range(self.dimension + 1):
            magnetism[d] = -self.dimension + 2 * d
            probabilities[d] = comb(self.dimension, d) * np.exp(beta * self.alpha / self.dimension * magnetism[d] ** 2)

        # normalize
        probabilities /= probabilities.sum()

        return probabilities, magnetism

    def _true_normalizing_constant(self, params):
        """ returns the true normalizing constant for a specified parameter

        parameters
        ----------
        params: tuple
            first value is the inverse temperature parameter in [0,1]

        returns
        -------
        float
            the true normalizing constant
        """
        beta = params[0]

        # target normalizing constant
        probabilities = np.zeros(self.dimension + 1)

        for d in range(self.dimension + 1):
            magnetism = -self.dimension + 2 * d
            probabilities[d] = comb(self.dimension, d) * np.exp(beta*self.alpha / self.dimension * magnetism ** 2)

        return probabilities.sum()

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

        return np.abs(true_probabilities - empirical_probabilities).sum() / 2.0

    def plot_diagnostics(self, output, path, max_steps=50):
        """ plots diagnostics for the mean field ising sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        path: list of tuples
            the sequence of interpolating distributions. Initial distribution is at index 0,
            the target distribution is the final entry
        max_steps:
            maximum number of density plots to create. If larger than the number of steps, uses the number of steps
        """

        samples, log_weights, ess = output[0], output[1], output[2]
        plt.rcParams['figure.figsize'] = 10, 6

        # plot probability of the first mode
        plt.subplot(221)
        positive_probability = (samples.mean(2) > 0).mean(1)
        plt.plot(positive_probability)
        plt.axhline(0.5, color='red')
        plt.ylim(-0.05, 1.05)
        plt.title('Positive mode probability')
        plt.xlabel('Iteration')

        # plot ess over iterations
        plt.subplot(222)
        plt.plot(ess, color='blue', label='path')
        plt.plot(np.array(path).flatten()[1:], color='red', label='path')
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')

        plt.xlabel('Iteration')

        # plot histogram of the final samples
        plt.subplot(223)
        tvs = []
        for sample, params in zip(samples, path):
            tvs.append(self._total_variation(sample, params))
        plt.plot(tvs)
        plt.title('Total variation distance')
        plt.xlabel('iteration')
        plt.ylim(-0.05, 1.05)

        # density plot at iterations
        max_steps = np.min((max_steps, samples.shape[0]))
        indices = np.floor(np.linspace(0, samples.shape[0] - 1, max_steps)).astype('int')
        color_map = plt.cm.plasma(np.linspace(0, 1, max_steps))

        # compute density
        xs = np.arange(-self.dimension, self.dimension + 1, 2)
        density = np.array([[(samples[ind].sum(1) == x).mean() for x in xs] for ind in indices])

        # plot the density
        plt.subplot(224)
        for i in range(max_steps):
            plt.plot(xs, density[i], color=color_map[i])
        plt.title('Density plot')
        plt.xlabel('Magnetism')
        plt.xlim(-self.dimension - 1, self.dimension + 1)

        plt.tight_layout()
        plt.show()
