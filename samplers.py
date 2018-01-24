# general imports
import numpy as np
from utils import effective_sample_size, log_weights_to_weights, residual_resampling, load_stan_model, ProgressBar, \
    stan_model_wrapper, plot_density, gaussian_kernel, unzip

# model specific imports
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import expit


class Sampler:

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

    def sampling(self, path, N, smc_kwargs={}, save_all_samples=False):
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
        progress_bar = ProgressBar(S-1)
        log_weights = np.zeros(N)
        ess_vector = np.zeros(S-1)
        samples = self.initial_distribution(N, path[0])
        all_samples = None

        if save_all_samples:
            all_samples = np.zeros((S,)+samples.shape)
            all_samples[0] = samples.copy()

        # do the sequential monte carlo steps
        for s in range(1, S):
            samples, log_weights, ess = self._smc_step(samples, log_weights, path[s-1], path[s], **smc_kwargs)
            ess_vector[s-1] = ess

            if save_all_samples:
                all_samples[s] = samples.copy()
            progress_bar.increment()
        progress_bar.finish()

        # package the output
        output = (samples, log_weights, ess_vector)
        if save_all_samples:
            output = (all_samples, log_weights, ess_vector)

        return output


class NormalSampler(Sampler):

    def __init__(self, mean, covariance, directory='stan', stan_file='normal_model.stan',
                 model_code_file='normal_model.txt', load=True):
        """Sequential monte carlo sampler for a normal distribution with specified mean and variance
        Uses a tempered sequence of distributions for sampling. Markov kernels are done via nuts in STAN

        attributes
        ----------
        mean: np.array
            (D, ) vector
        covariance: np.array
            (D, D) positive definite matrix

        parameters
        ----------
        directory: str
            directory where the stan_file and model_text_file are found. Usually the location of the package
        stan_file: str
            File name of the stan file. If the stan file does not exist, will pickle the model to directory/stan_file
        model_code_file: str
            File name of the model text file.  Can be none if the stan file exists
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.mean = mean
        self.covariance = covariance
        self.precision = np.linalg.inv(self.covariance)

        # load the stan model
        self.stan_model = load_stan_model(directory, stan_file, model_code_file, self._stan_text_model(), load)

        Sampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _log_pdf(self, samples, params):
        """ log_pdf of the normal distribution

        parameters
        ----------
        samples: np.array
            (N, D) array of samples with shape
        params: tuple
            contains the inverse temperature parameter

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        # init some things
        beta = params[0]

        # compute the pdf
        log_density = -beta/2 * gaussian_kernel(samples, self.mean, self.precision)

        return log_density

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a flattened version of the target

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            contains the inverse temperature parameter

        returns
        -------
        np.array
            (N, D) matrix of normally distributed samples
        """

        # init some things
        beta = params[0]
        rng = stats.multivariate_normal(mean=self.mean, cov=self.covariance/beta)

        return rng.rvs(N)

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of samples with shape
        params: tuple
            contains the inverse temperature parameter
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """

        # init some things
        beta = params[0]
        data = {
            'D': self.mean.shape[0],
            'beta': beta,
            'mu': self.mean,  # this name sucks but stan doesn't like mean
            'covariance': self.covariance
        }
        sample_list = [{'x': s} for s in samples]  # data needs to be a list of dictionaries for multi chains

        # do the sampling
        new_samples, fit = stan_model_wrapper(sample_list, data, self.stan_model, kernel_steps)

        return new_samples

    @staticmethod
    def _stan_text_model():
        """ returns the text for a stan model, just in case you've lost it

         returns
         -------
         str
            a stan model file
         """

        model_code = '''
        data {
            int<lower = 1> D;
            real<lower=0> beta;
            vector[D] mu;
            matrix[D,D] covariance;
        }

        parameters {
            vector[D] x;
        }

        model {
            target += beta*multi_normal_lpdf(x | mu, covariance);
        }
        '''

        return model_code


class MultimodalNormalSampler(Sampler):

    def __init__(self, means, scales, probability=0.5, dimension=2,
                 directory='stan', stan_file='multimodal_normal_model.stan',
                 model_code_file='multimodal_normal_model.txt', load=True):
        """Sequential monte carlo sampler for a mixture of two spherical normal distribution with specified mean and scales.
        A tempered sequence of distributions is used for sampling, beginning with a spherical normal.
        Markov kernels are done via nuts in STAN

        attributes
        ----------
        means: array-like
            first mode has mean np.ones(dimension)*means[0], similar for the second mode
        scales: array-like
            first mode has covariance np.eye(dimensions)*scales[0]**2, similar for the second mode
        probability: float in (0,1)
            probability of the first mode
        dimension: int>1
            number of dimensions

        parameters
        ----------
        directory: str
            directory where the stan_file and model_text_file are found. Usually the location of the package
        stan_file: str
            File name of the stan file. If the stan file does not exist, will pickle the model to directory/stan_file
        model_code_file: str
            File name of the model text file.  Can be none if the stan file exists
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.mean1 = means[0]
        self.mean2 = means[1]
        self.scale1 = scales[0]
        self.scale2 = scales[1]
        self.dimension = dimension
        self.probability = probability

        # initial distribution, there is probably a better way to do this
        self.mean0 = (self.mean1 + self.mean2)/2.0
        self.scale0 = np.abs(self.mean1-self.mean2)/4.0 + np.max(scales)
        self.noise = stats.norm()

        # load the stan model
        self.stan_model = load_stan_model(directory, stan_file, model_code_file, self._stan_text_model(), load)

        Sampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _log_pdf_initial(self, samples):
        """log pdf of samples w.r.t. the initial distribution

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples with shape

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        normed_samples = (samples - self.mean0)/self.scale0
        log_density = self.noise.logpdf(normed_samples).sum(1)  # spherical so you can sum accross dimensions

        return log_density

    def _log_pdf_target(self, samples):
        """log pdf of samples w.r.t. the target distribution

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples with shape

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        # first mode
        normed_samples1 = (samples - self.mean1)/self.scale1
        density1 = np.exp(self.noise.logpdf(normed_samples1).sum(1))

        # second mode
        normed_samples2 = (samples - self.mean2)/self.scale2
        density2 = np.exp(self.noise.logpdf(normed_samples2).sum(1))

        # combine and log
        log_density = np.log(self.probability*density1 + (1-self.probability)*density2)

        return log_density

    def _log_pdf(self, samples, params):
        """ log_pdf of the tempered mixture distribution

        parameters
        ----------
        samples: np.array
            (N, D) array of samples with shape
        params: tuple
            contains the mixture parameter, a float in (0,1)

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta = params[0]
        log_density = (1-beta)*self._log_pdf_initial(samples) + beta*self._log_pdf_target(samples)

        return log_density

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a flattened normal distribution

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            contains the mixture parameter, a float in (0,1)

        returns
        -------
        np.array
            (N, dimension) matrix of normally distributed samples
        """

        # init some things
        beta = params[0]  # isn't used for anything here, but this function takes params as an argument
        samples = self.noise.rvs((N, self.dimension))*self.scale0 + self.mean0

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, dimension) array of samples with shape
        params: tuple
            contains the mixture parameter, a float in (0,1)
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, dimension) array of updated samples
        """

        if kernel_steps < 2:
            raise ValueError('Number of iterations in sampler must be at least 2, change kernel_kwargs')

        # init some things
        beta = params[0]
        data = {
            'D': self.dimension,
            'beta': beta,
            'mean0': self.mean0 * np.ones(self.dimension),
            'mean1': self.mean1 * np.ones(self.dimension),
            'mean2': self.mean2 * np.ones(self.dimension),
            'covariance0': self.scale0**2 * np.eye(self.dimension),
            'covariance1': self.scale1**2 * np.eye(self.dimension),
            'covariance2': self.scale2**2 * np.eye(self.dimension),
            'probability': self.probability
        }
        stan_kwargs = {'pars': 'x'}
        sample_list = [{'x': s} for s in samples]  # data needs to be a list of dictionaries for multi chains

        # do the sampling
        new_samples, fit = stan_model_wrapper(sample_list, data, self.stan_model, kernel_steps, stan_kwargs)

        return new_samples

    @staticmethod
    def _stan_text_model():
        """ returns the text for a stan model, just in case you've lost it

         returns
         -------
         str
            a stan model file
         """

        model_code = """
        data {
            int<lower=1> D;
            real<lower=0, upper=1> beta;
            real<lower=0, upper=1> probability;

            vector[D] mean0;
            vector[D] mean1;
            vector[D] mean2;

            matrix[D,D] covariance0;
            matrix[D,D] covariance1;
            matrix[D,D] covariance2;
        }

        parameters {
            vector[D] x;
        }

        transformed parameters {
            real log_pdf_target;
            real log_pdf_initial;

            log_pdf_initial = multi_normal_lpdf(x | mean0, covariance0);
            log_pdf_target = log(probability*exp(multi_normal_lpdf(x | mean1, covariance1)) +
                (1-probability)*exp(multi_normal_lpdf(x | mean2, covariance2)));
        }

        model {
            target += (1-beta)*log_pdf_initial + beta*log_pdf_target;
        }
        """

        return model_code

    def plot_diagnostics(self, output, dim=0, max_steps=50):
        """ plots diagnostics for the multimodal normal sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        dim: int
            dimension to plot, must be less than self.dimension
        max_steps:
            maximum number of density plots to create. If larger than the number of steps, uses the number of steps
        """

        if dim >= self.dimension:
            raise ValueError("""dim must be less than MultimodalNormalSampler.dimension""")

        samples, log_weights, ess = output
        plt.rcParams['figure.figsize'] = 10, 6

        # plot probability of the first mode
        plt.subplot(221)
        distance1 = ((samples - self.mean1) ** 2).sum(2)
        distance2 = ((samples - self.mean2) ** 2).sum(2)
        mode_probabilities = (distance1 < distance2).mean(1)
        plt.plot(mode_probabilities)
        plt.axhline(self.probability, color='red')
        plt.ylim(-0.05, 1.05)
        plt.title('First mode probability')
        plt.xlabel('Iteration')

        # plot ess over iterations
        plt.subplot(222)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot histogram of the final samples
        plt.subplot(223)
        plt.hist(samples[-1, :, dim], bins=np.sqrt(samples.shape[1]).astype(int))
        plt.title('Target distribution histogram')
        plt.xlabel('Dimension {}'.format(dim))

        # density plot at iterations
        max_steps = np.min((max_steps, samples.shape[0]))
        indices = np.floor(np.linspace(0, samples.shape[0] - 1, max_steps)).astype('int')
        color_map = plt.cm.plasma(np.linspace(0, 1, max_steps))

        plt.subplot(224)
        for ind, color in zip(indices, color_map):
            x = samples[ind, :, dim]
            plot_density(x, {'color': color})
        plt.title('Density plot')
        plt.xlabel('Dimension {}'.format(dim))

        plt.tight_layout()
        plt.show()


class NormalPathSampler(Sampler):

    def __init__(self, mean1, mean2, covariance1, covariance2,
                 directory='stan', stan_file='normal_path_model.stan',
                 model_code_file='normal_path_model.txt', load=True):
        """ Sequential monte carlo sampler moving from one normal distribution to another via a tempered geometric mixture.
        Primarily used to test path sampling methods. Markov kernels are done via nuts in STAN

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

        parameters
        ----------
        directory: str
            directory where the stan_file and model_text_file are found. Usually the location of the package
        stan_file: str
            File name of the stan file. If the stan file does not exist, will pickle the model to directory/stan_file
        model_code_file: str
            File name of the model text file.  Can be none if the stan file exists
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.mean1 = mean1
        self.mean2 = mean2
        self.covariance1 = covariance1
        self.covariance2 = covariance2
        self.precision1 = np.linalg.inv(covariance1)
        self.precision2 = np.linalg.inv(covariance2)

        # load the stan model
        self.stan_model = load_stan_model(directory, stan_file, model_code_file, self._stan_text_model(), load)

        Sampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _get_normal_parameters(self, params):
        """ returns the mean, precision and covariance for the specified params

         parameters
         ----------
         params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        returns
        -------
        np.array
            (D,) mean vector
        np.array
            (D, D) precision matrix
        np.array
            (D, D) covariance matrix
        """

        beta, inverse_temperature = params
        precision = inverse_temperature*((1-beta)*self.precision1 + beta*self.precision2)
        covariance = np.linalg.inv(precision)
        almost_mean = inverse_temperature*(np.dot(self.precision1,self.mean1)*(1-beta) +
                                           np.dot(self.precision2, self.mean2)*beta)
        mean = np.dot(covariance,almost_mean)

        return mean, precision, covariance

    def _log_pdf(self, samples, params):
        """ log_pdf of the tempered geometric mixture distribution

        parameters
        ----------
        samples: np.array
            (N, D) array of samples with shape
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        mean, precision, covariance = self._get_normal_parameters(params)

        # compute the pdf
        log_density = -0.5 * gaussian_kernel(samples, mean, precision)

        return log_density

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a flattened normal distribution

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]


        returns
        -------
        np.array
            (N, D) matrix of normally distributed samples
        """

        # init some things
        samples = stats.multivariate_normal(mean=self.mean1, cov=self.covariance1).rvs(N)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of samples with shape
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """

        # init some things
        mean, precision, covariance = self._get_normal_parameters(params)
        data = {
            'D': self.mean1.shape[0],
            'mu': mean,  # this name sucks but stan doesn't like mean
            'covariance': covariance
        }
        sample_list = [{'x': s} for s in samples]  # data needs to be a list of dictionaries for multi chains

        # do the sampling
        new_samples, fit = stan_model_wrapper(sample_list, data, self.stan_model, kernel_steps)

        return new_samples

    @staticmethod
    def _stan_text_model():
        """ returns the text for a stan model, just in case you've lost it

         returns
         -------
         str
            a stan model file
         """

        model_code = '''
        data {
            int<lower = 1> D;
            vector[D] mu;
            matrix[D,D] covariance;
        }

        parameters {
            vector[D] x;
        }

        model {
            target += multi_normal_lpdf(x | mu, covariance);
        }
        '''

        return model_code

    def _compute_kl_divergence(self, samples, path):
        """ computes the KL divergence at each step between the sample mean/covariance and the true parameters of the
        interpolating distribution. KL divergence is a measure of discrepancy between the intended distribution
        and the observed distribution.

        parameters
        ----------
        samples: np.array
            (S, N, D) array from self.sampling with save_all_samples=True
        path: list of tuples
            parameters used at each sampling step. first value is a mixing parameter in [0,1]
            second value is inverse temperature parameter in (0,1]

        returns
        -------
        np.array
            (S,) vector of KL divergences.
        """

        # compute sample moments
        sample_covariances = map(np.cov, samples.transpose(0, 2, 1))
        sample_means = samples.mean(1)

        # obtain true parameters
        true_parameters = map(self._get_normal_parameters, path)
        true_means, true_precisions, true_covariances = unzip(true_parameters)

        # compute the divergence. this might be easier to understand if I wrote it as a loop
        traces = np.array([np.trace(np.dot(true, sample))
                           for true, sample in zip(true_precisions, sample_covariances)])
        kernels = np.array([gaussian_kernel(true, sample, precision)
                            for true, sample, precision in zip(true_means, sample_means, true_precisions)])
        determinants = np.array([np.linalg.slogdet(true)[1] - np.linalg.slogdet(sample)[1]
                                 for true, sample in zip(true_covariances, sample_covariances)])
        kl_divergence = 0.5 * (traces + kernels - self.mean1.shape[0] + determinants)

        return kl_divergence

    def plot_diagnostics(self, output, path):
        """ plots diagnostics for the normal path sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        path: list of tuples
            parameters used at each sampling step. first value is a mixing parameter in [0,1]
            second value is inverse temperature parameter in (0,1]
        """

        samples, log_weights, ess = output
        plt.rcParams['figure.figsize'] = 10, 6

        # plot ess over iterations
        plt.subplot(121)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot kl divergence
        divergence = self._compute_kl_divergence(samples, path)
        plt.subplot(122)
        plt.plot(divergence)
        plt.title('KL divergence')
        plt.xlabel('iteration')


class MeanFieldIsingSampler(Sampler):

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

        Sampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

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

        samples, log_weights, ess = output
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
        plt.plot(ess)
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
