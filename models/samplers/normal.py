import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils import load_stan_model, stan_model_wrapper, plot_density, gaussian_kernel, unzip, generate_covariance_matrix
from smc_samplers import SMCSampler


class NormalSampler(SMCSampler):

    def __init__(self, mean, covariance, load=True):
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
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.mean = mean
        self.covariance = covariance
        self.precision = np.linalg.inv(self.covariance)

        # load the stan model
        self.stan_model = load_stan_model(
            'stan',
            'normal_model.stan',
            'normal_model.txt',
            self._stan_text_model(),
            load
        )

        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

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


class MultimodalNormalSampler(SMCSampler):

    def __init__(self, means, scales, probability=0.5, dimension=2, load=True):
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
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='multimodal_normal_model.stan',
            model_code_file='multimodal_normal_model.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )

        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

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


class NormalPathSampler(SMCSampler):

    def __init__(self, mean1, mean2, covariance1, covariance2, load=True):
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
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='normal_path_model.stan',
            model_code_file='normal_path_model.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )

        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

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
        mean, precision, covariance = self._get_normal_parameters(params)

        samples = stats.multivariate_normal(mean=mean, cov=covariance).rvs(N)

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


class RegressionSelectionSampler(SMCSampler):

    def __init__(self, n_observations, n_dimensions, prior_sd, design_correlation=0.3,  coefficient_sd=1.0, seed=1337):
        """ Sequential monte carlo sampler for comparing a linear model with one set of features
        to a linear model with another set of features. Uses a tempered geometric mixture.
        Markov kernels are done via independent sampling

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
        """
        if seed is not None:
            np.random.seed(seed)

        # generate the data
        X, Y, true_coefficients, design_covariance = self._generate_data(n_observations, n_dimensions,
                                                                         design_correlation, coefficient_sd)
        self.X = X
        self.Y = Y
        self.true_coefficients = true_coefficients
        self.design_covariance = design_covariance
        self.gamma1 = np.diag(np.array([0, 1]*(n_dimensions/2)) == 1)
        self.gamma0 = np.diag(np.array([0, 1]*(n_dimensions/2)) == 0)
        self.n_dimensions = n_dimensions
        self.prior_sd = prior_sd

        #
        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    @staticmethod
    def _generate_data(n_observations, n_dimensions, design_correlation, coefficient_sd):
        """ Generates data from the model.  First, a design matrix is drawn with correlation according to design
        correlation. True covariates are chosen so that each model contains half the true covariates. Coefficients
        are then drawn randomly using a normal distribution. Observations are created by multiplying the coefficients
        by the true covariates and adding some noise


        parameters
        ----------
        n_observations: int
            number of observations to generate
        n_dimensions: even int
            number of covariate dimensions to use
        design_correlation: float in (-1.0, 0.5)
            roughly speaking, this is the correlation between adjacent columns in the design matrix
        coefficient_sd: float > 0
            standard deviation to use when sampling the coefficients

        returns
        -------
        np.array
            (n_observations, n_dimensions) array of covariates
        np.array
            (n_observations, ) array of observations
        np.array
            (n_coefficients, ) array of true coefficients. Half of them are exactly 0.
        np.array:
            (n_coefficients, n_coefficients) design matrix correlation

        """

        # create the true parameters
        true_indices = np.ones(n_dimensions)
        true_indices[: n_dimensions/2][::2] = 0
        true_indices[n_dimensions/2:][1::2] = 0

        coefficients = np.zeros(n_dimensions)
        coefficients[true_indices == 1] = stats.norm().rvs(n_dimensions/2)*coefficient_sd

        # generate the data
        covariance = generate_covariance_matrix(n_dimensions, 1.0, -design_correlation)
        X = stats.multivariate_normal(mean=np.zeros(n_dimensions), cov=covariance).rvs(n_observations)
        Y = np.dot(X, coefficients) + stats.norm().rvs(n_observations)

        return X, Y, coefficients, covariance

    def _log_pdf(self, samples, params):
        """ log_pdf of the tempered geometric mixture distribution

        parameters
        ----------
        samples: np.array
            (N, n_dimensions) array of samples with shape
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta, temperature = params

        # compute the log pdf for each distribution
        residuals0 = self.Y[None, :] - np.dot(samples, np.dot(self.X, self.gamma0).T)
        log_density0 = -0.5*(residuals0**2).sum(1)

        residuals1 = self.Y[None, :] - np.dot(samples, np.dot(self.X, self.gamma1).T)
        log_density1 = -0.5 * (residuals1 ** 2).sum(1)

        log_prior = (samples**2).sum(1)/self.prior_sd**2

        # combine them
        log_density = temperature*((1.0-beta)*log_density0 + beta*log_density1 + log_prior)

        return log_density

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

        beta, temperature = params

        # could pre compute this stuff but I like to see it done explicity
        precision0 = np.dot(np.dot(self.X, self.gamma0).T, np.dot(self.X, self.gamma0))
        precision1 = np.dot(np.dot(self.X, self.gamma1).T, np.dot(self.X, self.gamma1))

        mean0 = np.dot(np.dot(self.X, self.gamma0).T, self.Y)
        mean1 = np.dot(np.dot(self.X, self.gamma1).T, self.Y)

        # compute the quantities of interest
        precision = temperature*((1.0-beta)*precision0 + beta*precision1 + np.eye(self.n_dimensions)/self.prior_sd**2)
        covariance = np.linalg.inv(precision)
        mean = temperature*((1.0-beta)*mean0 + beta*mean1)
        mean = np.dot(covariance, mean)

        return mean, precision, covariance

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a model consisting of the even indices

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]


        returns
        -------
        np.array
            (N, n_dimensions) matrix of normally distributed samples
        """

        # init some things
        mean, precision, covariance = self._get_normal_parameters(params)
        samples = stats.multivariate_normal(mean=mean, cov=covariance).rvs(N)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal posterior distribution. Uses an independence sampler targeting
        the exact distribution

        parameters
        ----------
        samples: np.array
            (N, n_dimensions) array of samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]
        kernel_steps: int >= 2
            Not used due to independent sampler.

        return
        ------
        np.array
            (N, n_dimension) array of updated samples
        """

        # draw independently
        mean, precision, covariance = self._get_normal_parameters(params)
        new_samples = stats.multivariate_normal(mean=mean, cov=covariance).rvs(samples.shape[0])

        return new_samples

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
        kl_divergence = 0.5 * (traces + kernels - self.n_dimensions + determinants)

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
        plt.subplot(221)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot kl divergence
        divergence = self._compute_kl_divergence(samples, path)
        plt.subplot(222)
        plt.plot(divergence)
        plt.title('KL divergence')
        plt.xlabel('Iteration')

        # plot the log-likelihood of the mode
        log_pdf_list = []
        for params in path:
            mean, precision, covariance = self._get_normal_parameters(params)
            log_pdf_list.append(stats.multivariate_normal(mean=mean, cov=covariance).logpdf(mean))

        plt.subplot(223)
        plt.plot(log_pdf_list)
        plt.title('Log density of the mode')
        plt.xlabel('Iteration')

        # plot the covariance matrix
        plt.subplot(224)
        plt.pcolor(self.design_covariance[::-1], vmin=-1, vmax=1)
        plt.title('Design covariance matrix')
        plt.colorbar()

        plt.tight_layout()
