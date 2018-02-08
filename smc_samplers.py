# general imports
import numpy as np
from utils import effective_sample_size, log_weights_to_weights, residual_resampling, ProgressBar


# Todo: Consider adding a subclass specifically for stan models
class SMCSampler:

    def __init__(self, log_pdf, initial_distribution, markov_kernel):
        """generic sampler for sequential monte carlo

        attributes
        ----------
        log_pdf: function
            see self.log_pdf
        initial_distribution: function
            see self.initial_distribution
        markov_kernel:
            see self.markov_kernel
        """

        self.log_pdf = log_pdf
        self.initial_distribution = initial_distribution
        self.markov_kernel = markov_kernel

    # These methods are provided as examples
    # They need to be filled out and input to __init__
    @staticmethod
    def log_pdf(samples, params):
        """computes the log pdf of each sample under a certain set of parameters

        parameters
        ----------
        samples: np.array
            array of samples with shape (N, ...)
        params: tuple
            parameters of the sequential monte carlo path (i.e. temperature)

        returns
        -------
        np.array
            log_pdf evaluated at path with shape (N, )
        """

    @staticmethod
    def initial_distribution(N, params):
        """draws S samples from the initial distribution

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            parameters of the sequential monte carlo path (i.e. temperature)

        returns
        -------
        np.array
            initial samples, has dimension (N,...)
        """

    @staticmethod
    def markov_kernel(samples, params, kernel_steps):
        """applies a markov transition kernel to each sample, targeting the distribution corresponding to path

        parameters
        ----------
        samples: np.array
            array of samples, has dimension (N,...)
        params: tuple
            parameters of the sequential monte carlo path (i.e. temperature)
        kernel_steps: int
            number of markov kernel transitions to apply

        returns
        -------
        np.array
            new matrix of samples targeting distribution corresponding to path
        """

    def _smc_step(self, samples, log_weights, current_params, next_params,
                  resampling_method='residual', kernel_steps=10):
        """ applies a single step of sequential monte carlo to the samples, moving from current_path to next_path

        parameters
        ----------
        samples: np.array
            2d array of samples with shape (N, ...)
        log_weights: np.array
            current particle weights on the log scale scale, has shape (N, )
        current_params: tuple
            parameters of the current distribution
        next_params: tuple
            parameters of the next distribution
        resampling_method: str
            what kind of resampling to use. Should be 'multinomial', 'residual' or 'ais'. 'residual' is
            believed to produce lower variance than 'multinomial'. 'ais' is no resampling and corresponds
            to annealed importance sampling (see Neal 1998). Currently doesn't support adaptive resampling
        kernel_steps: int
            number of times to apply the markov kernel

        returns
        -------
        np.array
            (N,...) matrix of samples targeting distribution corresponding to next_path
        np.array
            (N,) vector of updated log_weights
        float
            effective sample size
        """

        # init some things
        N = samples.shape[0]

        # update the log weights, compute ess and weights
        updated_log_weights = log_weights + self.log_pdf(samples, next_params) - self.log_pdf(samples, current_params)
        weights = log_weights_to_weights(updated_log_weights)
        ess = effective_sample_size(weights)

        # resample according the the specified method
        next_log_weights = np.zeros_like(weights)
        if resampling_method == 'multinomial':
            indices = np.random.multinomial(N, weights)
            samples = samples[indices]

        elif resampling_method == 'residual':
            indices = residual_resampling(weights)
            samples = samples[indices]

        elif resampling_method == 'ais':
            next_log_weights = updated_log_weights.copy()

        else:
            raise ValueError("""resampling_method must be 'multinomial', 'residual', or 'ais' """)

        samples = self.markov_kernel(samples, next_params, kernel_steps)

        return samples, next_log_weights, ess

    def sampling(self, path, N, smc_kwargs={}, save_all_samples=True, verbose=True):
        """uses sequential monte carlo to sample from a target distribution

        parameters
        ----------
        path: list of tuples
            the sequence of interpolating distributions. Initial distribution is at index 0,
            the target distribution is the final entry
        N: int
            number of particles to use
        smc_kwargs: dict
            additional arguments to smc_step. For example, resampling_method or kernel_steps
        save_all_samples: bool
            if true, saves the samples at each trajectory. if false only returns the final particles
        verbose: bool
            if true prints a progress bar tracking sampling progress

        returns
        -------
        np.array
            (N,...) matrix of samples if save_all_samples==True, the the first dimension is the
            sample index and the the last index is the step index (the final samples are samples[...,-1])
        np.array
            (N,) vector of final weights (only meaningful for 'ais' resampling)
        np.array
            (len(path)-1, ) vector of effective sample size following each resampling
        """

        # init some things
        S = len(path)
        log_weights = np.zeros(N)
        ess_vector = np.zeros(S-1)
        samples = self.initial_distribution(N, path[0])
        all_samples = None
        if verbose:
            progress_bar = ProgressBar(S - 1)

        if save_all_samples:
            all_samples = np.zeros((S,)+samples.shape)
            all_samples[0] = samples.copy()

        # do the sequential monte carlo steps
        for s in range(1, S):
            samples, log_weights, ess = self._smc_step(samples, log_weights, path[s-1], path[s], **smc_kwargs)
            ess_vector[s-1] = ess

            if save_all_samples:
                all_samples[s] = samples.copy()

            if verbose:
                progress_bar.increment()
        if verbose:
            progress_bar.finish()

        # package the output
        output = (samples, log_weights, ess_vector)
        if save_all_samples:
            output = (all_samples, log_weights, ess_vector)

        return output
