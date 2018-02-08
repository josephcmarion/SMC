import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import expit

from utils import load_stan_model, stan_model_wrapper, gaussian_kernel
from smc_samplers import SMCSampler


class LogisticRegressionSampler(SMCSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance, load=True):
        """ SMC sampler for logistic regression. Moves from the prior to model of interest via a
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
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.X = X
        self.Y = Y
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.prior_precision = np.linalg.inv(prior_covariance)

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='logistic_regression.stan',
            model_code_file='logistic_regression.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )
        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _log_pdf(self, samples, params):
        """ log_pdf of the geometric tempered logistic model

        parameters
        ----------
        samples: np.array
            (N, D) array of sample coefficients
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta, inverse_temperature = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0-10**-6
        pi[pi == 0.0] = 10**-6

        log_likelihood = self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)

        # log density
        log_density = inverse_temperature*(beta*log_likelihood.sum(1) + log_prior)

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
        samples = stats.multivariate_normal(mean=self.prior_mean, cov=self.prior_covariance).rvs(N)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of coefficient samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """
        beta, inverse_temperature = params

        # init some things
        data = {
            'D': self.prior_mean.shape[0],
            'N': self.X.shape[0],
            'X': self.X,
            'Y': self.Y,
            'prior_mean': self.prior_mean,
            'prior_covariance': self.prior_covariance,
            'beta': beta,
            'inverse_temperature': inverse_temperature
        }
        stan_kwargs = {'pars': 'coefficients'}
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
            int<lower = 1> D;
            int<lower = 1> N;

            matrix[N, D] X;
            vector[N] Y;

            vector[D] prior_mean;
            matrix[D,D] prior_covariance;

            real<lower=0, upper=1> beta;
            real<lower=0, upper=1> inverse_temperature;
        }

        parameters {
            vector[D] coefficients;
        }

        transformed parameters {
            vector[N] mu;
            vector[N] pi;

            mu = X*coefficients;
            pi = inv_logit(mu);
        }


        model {
            target += inverse_temperature*beta*(Y .* log(pi) + (1-Y) .* log(1-pi));
            target += inverse_temperature*multi_normal_lpdf(coefficients | prior_mean, prior_covariance);
        }
        """

        return model_code

    @staticmethod
    def plot_diagnostics(output, true_coefficients):
        """ plots diagnostics for the mean field ising sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        true_coefficients: np.array
            true coefficient values, possibly obtained through high quality monte carlo run
        """

        samples, log_weights, ess = output
        plt.rcParams['figure.figsize'] = 10, 6

        # plot ess over iterations
        plt.subplot(121)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot parameter estimates
        samples.mean(0)
        plt.subplot(122)
        plt.boxplot(samples)
        for i, coefficient in enumerate(true_coefficients):
            plt.plot([i + 0.6, i + 1.4], [coefficient] * 2, color='purple', ls=':', lw=4)
        plt.title('Coefficient estimates')
        plt.xlabel('coefficient number')

        plt.tight_layout()
        plt.show()


class LogisticPriorPathSampler(SMCSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance, load=True):
        """ SMC sampler for logistic regression. Uses likelihood tempering in addition to a geometric mixture from a
        posterior approximation to the prior of interest. Posterior is approximated by a gaussian, whose mean and
        covariance are determines by maximum likelihood estimation. Markov kernels are done via nuts in STAN

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
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.X = X
        self.Y = Y

        # prior parameters
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.prior_precision = np.linalg.inv(prior_covariance)

        # posterior approximation parameters through logistic regression
        from statsmodels.api import Logit as LogisticRegression

        lr = LogisticRegression(Y, X)
        fit = lr.fit()

        self.posterior_mean = fit.params
        self.posterior_covariance = fit.cov_params()
        self.posterior_precision = np.linalg.inv(self.posterior_covariance)

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='logistic_regression.stan',
            model_code_file='logistic_regression.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )
        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _get_normal_parameters(self, params):
        """ returns the mean, precision and covariance of the pseudo prior for the specified params

        parameters
        ----------
        params: tuple
            likelihood inverse temperature in [0,1] and prior mixture parameter in [0,1]

        returns
        -------
        np.array
            (D, ) mean vector
        np.array
            (D, D) precision matrix
        np.array
            (D, D) covariance matrix
        """

        beta, alpha = params

        pseudo_precision = alpha * self.prior_precision + (1-alpha)*self.posterior_precision
        pseudo_covariance = np.linalg.inv(pseudo_precision)

        pseudo_mean = alpha*np.dot(self.prior_precision, self.prior_mean) + \
                      (1 - alpha) * np.dot(self.posterior_precision, self.posterior_mean)
        pseudo_mean = np.dot(pseudo_covariance, pseudo_mean)

        return pseudo_mean, pseudo_precision, pseudo_covariance

    def _log_pdf(self, samples, params):
        """ log_pdf of the geometric tempered logistic model

        parameters
        ----------
        samples: np.array
            (N, D) array of sample coefficients
        params: tuple
            likelihood inverse temperature in [0,1] and prior mixture parameter in [0,1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        pseudo_mean, pseudo_precision, pseudo_covariance = self._get_normal_parameters(params)
        beta, alpha = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0-10**-6
        pi[pi == 0.0] = 10**-6

        log_likelihood = self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)

        # log prior
        log_prior = -0.5 * gaussian_kernel(samples, pseudo_mean, pseudo_precision)

        # log density
        log_density = beta*log_likelihood.sum(1) + log_prior

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
        pseudo_mean, pseudo_precision, pseudo_covariance = self._get_normal_parameters(params)
        samples = stats.multivariate_normal(mean=pseudo_mean, cov=pseudo_covariance).rvs(N)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of coefficient samples
        params: tuple
            first value is a mixing parameter in [0,1] second value is inverse temperature parameter in (0,1]
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """

        pseudo_mean, pseudo_precision, pseudo_covariance = self._get_normal_parameters(params)
        beta, alpha = params

        # init some things
        data = {
            'D': self.prior_mean.shape[0],
            'N': self.X.shape[0],
            'X': self.X,
            'Y': self.Y,
            'prior_mean': pseudo_mean,
            'prior_covariance': pseudo_covariance,
            'beta': beta,
            'inverse_temperature': 1.0
        }
        stan_kwargs = {'pars': 'coefficients'}
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
            int<lower = 1> D;
            int<lower = 1> N;

            matrix[N, D] X;
            vector[N] Y;

            vector[D] prior_mean;
            matrix[D,D] prior_covariance;

            real<lower=0, upper=1> beta;
            real<lower=0, upper=1> inverse_temperature;
        }

        parameters {
            vector[D] coefficients;
        }

        transformed parameters {
            vector[N] mu;
            vector[N] pi;

            mu = X*coefficients;
            pi = inv_logit(mu);
        }


        model {
            target += inverse_temperature*beta*(Y .* log(pi) + (1-Y) .* log(1-pi));
            target += inverse_temperature*multi_normal_lpdf(coefficients | prior_mean, prior_covariance);
        }
        """

        return model_code

    @staticmethod
    def plot_diagnostics(output, true_coefficients):
        """ plots diagnostics for the mean field ising sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        true_coefficients: np.array
            true coefficient values, possibly obtained through high quality monte carlo run
        """

        samples, log_weights, ess = output
        plt.rcParams['figure.figsize'] = 10, 6

        # plot ess over iterations
        plt.subplot(121)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot parameter estimates
        samples.mean(0)
        plt.subplot(122)
        plt.boxplot(samples)
        for i, coefficient in enumerate(true_coefficients):
            plt.plot([i + 0.6, i + 1.4], [coefficient] * 2, color='purple', ls=':', lw=4)
        plt.title('Coefficient estimates')
        plt.xlabel('coefficient number')

        plt.tight_layout()
        plt.show()


class GeometricTemperedLogisticSampler(SMCSampler):

    def __init__(self, X, Y, prior_mean, prior_covariance, load=True):
        """ SMC sampler for logistic regression. Uses geometric tempering to move from a gaussian centered at the MLE
        to the posterior of intersest.   Markov kernels are done via nuts in STAN

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
        load:
            If true, attempts to load the model. If false, compiles the model
        """

        self.X = X
        self.Y = Y

        # prior parameters
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.prior_precision = np.linalg.inv(prior_covariance)

        # posterior approximation parameters through logistic regression
        from statsmodels.api import Logit as LogisticRegression

        lr = LogisticRegression(Y, X)
        fit = lr.fit(method='bfgs')

        self.initial_mean = fit.params
        self.initial_covariance = fit.cov_params()
        self.initial_precision = np.linalg.inv(self.initial_covariance)

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='logistic_regression.stan',
            model_code_file='logistic_regression.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )
        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _get_normal_parameters(self, params):
        """ returns the mean, precision and covariance of the pseudo prior for the specified params

        parameters
        ----------
        params: tuple
            geometric mixing parameter in [0,1] and inverse temperature in (0, 1]

        returns
        -------
        np.array
            (D, ) mean vector
        np.array
            (D, D) precision matrix
        np.array
            (D, D) covariance matrix
        """

        beta, temperature = params

        precision = temperature*(beta * self.prior_precision + (1-beta)*self.initial_precision)
        covariance = np.linalg.inv(precision)

        mean = temperature*(beta*np.dot(self.prior_precision, self.prior_mean) +
                            (1 - beta) * np.dot(self.initial_precision, self.initial_mean))
        mean = np.dot(covariance, mean)

        return mean, precision, covariance

    def _log_pdf(self, samples, params):
        """ log_pdf of the geometric tempered logistic model

        parameters
        ----------
        samples: np.array
            (N, D) array of sample coefficients
        params: tuple
            geometric mixing parameter in [0,1] and inverse temperature in (0, 1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta, temperature = params

        # log likelihood
        mu = np.dot(samples, self.X.T)
        pi = expit(mu)

        # fix some numerical issues
        pi[pi == 1.0] = 1.0-10**-6
        pi[pi == 0.0] = 10**-6

        log_likelihood = self.Y * np.log(pi) + (1.0 - self.Y) * np.log(1.0 - pi)
        log_likelihood = temperature * beta * log_likelihood.sum(1)

        # log prior
        log_prior = temperature * beta * -0.5 * gaussian_kernel(samples, self.prior_mean, self.prior_precision)

        # log initial
        log_initial = temperature * (1-beta) * -0.5 *\
            gaussian_kernel(samples, self.initial_mean, self.initial_precision)

        # log density
        log_density = log_initial + log_prior + log_likelihood

        return log_density

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a flattened normal distribution

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            geometric mixing parameter in [0,1] and inverse temperature in (0, 1]

        returns
        -------
        np.array
            (N, D) matrix of normally distributed samples
        """

        # init some things
        pseudo_mean, pseudo_precision, pseudo_covariance = self._get_normal_parameters(params)
        samples = stats.multivariate_normal(mean=pseudo_mean, cov=pseudo_covariance).rvs(N)

        return samples

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of coefficient samples
        params: tuple
            geometric mixing parameter in [0,1] and inverse temperature in (0, 1]
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """

        pseudo_mean, pseudo_precision, pseudo_covariance = self._get_normal_parameters(params)
        beta, temperature = params

        # init some things
        data = {
            'D': self.prior_mean.shape[0],
            'N': self.X.shape[0],
            'X': self.X,
            'Y': self.Y,
            'prior_mean': pseudo_mean,
            'prior_covariance': pseudo_covariance,
            'beta': beta*temperature,  # this looks crazy but it's correct given how I've specified the model
            'inverse_temperature': 1.0  # same with this line
        }
        stan_kwargs = {'pars': 'coefficients'}
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
            int<lower = 1> D;
            int<lower = 1> N;

            matrix[N, D] X;
            vector[N] Y;

            vector[D] prior_mean;
            matrix[D,D] prior_covariance;

            real<lower=0, upper=1> beta;
            real<lower=0, upper=1> inverse_temperature;
        }

        parameters {
            vector[D] coefficients;
        }

        transformed parameters {
            vector[N] mu;
            vector[N] pi;

            mu = X*coefficients;
            pi = inv_logit(mu);
        }


        model {
            target += inverse_temperature*beta*(Y .* log(pi) + (1-Y) .* log(1-pi));
            target += inverse_temperature*multi_normal_lpdf(coefficients | prior_mean, prior_covariance);
        }
        """

        return model_code

    @staticmethod
    def plot_diagnostics(output, true_coefficients):
        """ plots diagnostics for the mean field ising sampler.

        parameters
        ----------
        output: np.array
            result of self.sampling, a tuple with samples from the full run, final weights, and ess at each step
            sampling must be run with save_all_samples=True
        true_coefficients: np.array
            true coefficient values, possibly obtained through high quality monte carlo run
        """

        samples, log_weights, ess = output
        plt.rcParams['figure.figsize'] = 10, 6

        # plot ess over iterations
        plt.subplot(121)
        plt.plot(ess)
        plt.ylim(-0.05, 1.05)
        plt.title('Effective sample size')
        plt.xlabel('Iteration')

        # plot parameter estimates
        samples.mean(0)
        plt.subplot(122)
        plt.boxplot(samples)
        for i, coefficient in enumerate(true_coefficients):
            plt.plot([i + 0.6, i + 1.4], [coefficient] * 2, color='purple', ls=':', lw=4)
        plt.title('Coefficient estimates')
        plt.xlabel('coefficient number')

        plt.tight_layout()
        plt.show()
