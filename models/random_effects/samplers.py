# general imports
import numpy as np
import scipy.stats as stats
from utils import load_stan_model, stan_model_wrapper
from smc_samplers import SMCSampler


class RandomEffectsSampler(SMCSampler):

    def __init__(self, n_observations, n_groups, alpha, sigma, tau, path_type, load=True, seed=1337):
        """ SMC sampler for a random effects model using simulated data.
        Supports a large number of possible path types (I hope).
        Markov kernels are done via nuts in STAN

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
        path_type: str
            specifies the path type. Options include 'geometric', 'geometric-tempered'
        seed: int?
            used to set the seed for data generation

        parameters
        ----------
        load:
            If true, attempts to load the model. If false, compiles the model
        """
        self.valid_types = ['geometric', 'geometric-tempered', 'prior-relaxing']
        self._check_valid_path_type(path_type)

        # save all of this model nonsense
        observations, groups, random_effects = self._generate_data(n_observations, n_groups, alpha, sigma, tau, seed)
        self.n_observations = n_observations
        self.n_groups = n_groups
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.seed = seed
        self.observations = observations
        self.groups = groups
        self.random_effects = random_effects
        self.path_type = path_type

        # load the stan model
        self.stan_model = load_stan_model(
            directory='stan',
            stan_file='random_effects.stan',
            model_code_file='random_effects.txt',
            model_code_text=self._stan_text_model(),
            load=load
        )

        SMCSampler.__init__(self, self._log_pdf, self._initial_distribution, self._markov_kernel)

    def _params_to_params(self, params):
        """ converts the path parameters to a full set of parameters

        parameters
        ----------
        params: tuple
            a tuple corresponding to the type chosen by path_type

        returns
        -------
        tuple
            beta, temperature, prior_tightening

        """

        if self.path_type == 'geometric':
            new_params = (params[0], 1.0, 1.0)
        elif self.path_type == 'geometric-tempered':
            new_params = (params[0], params[1], 1.0)
        elif self.path_type == 'prior-relaxing':
            new_params = (params[0], 1.0, params[1])

        return new_params

    def _initial_distribution(self, N, params):
        """ samples from the initial distribution, a conjugate normal model

        parameters
        ----------
        N: int > 0
            number of samples to draw
        params: tuple
            contains geometric mixture parameter in [0,1]

        returns
        -------
        np.array
            (N, D) matrix of samples
        """

        # init some things
        beta, temperature, relax = self._params_to_params(params)

        # scale parameters
        sigmas = stats.invgamma(
            a=((self.n_observations+2)*temperature-3)/2,
            scale=self.observations.var()*(self.n_observations-1)*temperature/2
        ).rvs((N, 1))**0.5
        taus = stats.invgamma(a=2, scale=2).rvs((N, 1))**0.5

        # location parameters
        alphas = self.observations.mean()+stats.norm().rvs((N, 1))*sigmas/(self.n_observations*temperature)**0.5
        random_effects = stats.norm().rvs((N, self.n_groups))*taus/relax**0.5

        return np.hstack([alphas, random_effects, taus, sigmas])

    def _log_pdf(self, samples, params):
        """ log_pdf of the geometric tempered logistic model. Right now the prior doesn't change,
        so generating the log pdf is easy. That being said, this will need to be changed in the future.

        parameters
        ----------
        samples: np.array
            (N, D) array of sample coefficients
        params: tuple
            contains geometric mixture parameter in [0,1]

        return
        ------
        np.array
            (N, ) vector of log_densities
        """

        beta, temperature, relax = self._params_to_params(params)
        alphas, random_effects, taus, sigmas = self._unpackage_samples(samples)

        # for the initial distribution
        deltas = (self.observations[None, :] - alphas[:, None])/sigmas[:, None]
        log_pdf_initial = temperature*(1.0-beta)*(-0.5)*(deltas**2).sum(1)

        # for the target distribution
        deltas = (self.observations[None, :] - alphas[:, None] - random_effects[:, self.groups])/sigmas[:, None]
        log_pdf_target = temperature*beta * (-0.5) * (deltas ** 2).sum(1)

        # prior distribution
        deltas = (random_effects/taus[:, None])
        log_pdf_re_prior = -0.5*(deltas**2).sum()*relax

        return log_pdf_initial + log_pdf_target + log_pdf_re_prior

    def _markov_kernel(self, samples, params, kernel_steps=10):
        """ markov kernel targeting the normal distribution, implemented in STAN

        parameters
        ----------
        samples: np.array
            (N, D) array of coefficient samples
        params: tuple
            contains geometric mixture parameter in [0,1]
        kernel_steps: int >= 2
            number of hamiltonian transitions to run

        return
        ------
        np.array
            (N, D) array of updated samples
        """

        beta, temperature, relax = self._params_to_params(params)

        data = {
            'n_observations': self.n_observations,
            'n_groups': self.n_groups,
            'groups': self.groups + 1,
            'observations': self.observations,
            'beta': beta,
            'temperature': temperature,
            'relax': relax
        }

        stan_kwargs = {'pars': ['alpha', 'random_effects', 'tau', 'sigma']}

        alphas, random_effects, taus, sigmas = self._unpackage_samples(samples)
        sample_list = [
            {'alpha': alpha, 'random_effects': random_effect, 'tau2': tau ** 2, 'sigma2': sigma ** 2}
            for alpha, random_effect, tau, sigma
            in zip(alphas, random_effects, taus, sigmas)
        ]

        # do the sampling
        new_samples, fit = stan_model_wrapper(sample_list, data, self.stan_model, kernel_steps, stan_kwargs)

        return new_samples

    def _unpackage_samples(self, samples):
        """ takes a matrix of samples and splits them into different parameters

        parameters
        ----------
        samples: np.array
            (N, D) sample values, possibly from markov_kernel
        n_groups: int
            number of random effect groups

        returns
        -------
        np.array
            (N, ) array of alphas
        np.array
            (N, n_groups) array of group random effects
        np.array
            (D, ) array of random effect standard deviations
        np.array
            (D, ) array of observation noise standard deviations
        """

        alpha = samples[:, 0]
        random_effects = samples[:, range(1, self.n_groups + 1)]
        tau = samples[:, -2]
        sigma = samples[:, -1]

        return alpha, random_effects, tau, sigma

    @staticmethod
    def _generate_data(n_observations, n_groups, alpha, sigma, tau, seed=1337):
        """ generates data from a simple model with an intercept and a group level random effect

        parameters
        ----------
        n_observations: int
            number of data points to generate
        n_groups: int
            number of groups to use
        alpha: float
            the baseline intercept
        sigma: float>0
            variance of the observation noise
        tau: float>0
            standard deviation of the group level effects
        seed: float?
            sets the seed of the random number generator for reproducibility

        returns
        -------
        np.array
            (n_observations, ) response vector
        np.array
            (n_observations, ) group label vector
        np.array
            (n_groups, ) group level intercepts


        """
        np.random.seed(seed)
        groups = np.random.choice(n_groups, n_observations)
        random_effects = stats.norm(scale=tau).rvs(n_groups)

        observations = alpha + random_effects[groups] + stats.norm().rvs(n_observations) * sigma

        return observations, groups, random_effects

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
            int<lower=1> n_observations;
            int<lower=1> n_groups;

            int<lower=0> groups[n_observations];
            vector[n_observations] observations;

            real<lower=0, upper=1> beta;
            real<lower=0, upper=1> temperature;
            real<lower=0, upper=1> relax;
        }

        parameters {
            real alpha;
            vector[n_groups] random_effects;

            real<lower=0> tau2;
            real<lower=0> sigma2;
        }

        transformed parameters {
            vector[n_observations] mu;

            real<lower=0> tau;
            real<lower=0> sigma;

            mu = alpha + random_effects[groups];
            tau = sqrt(tau2);
            sigma = sqrt(sigma2);
        }

        model {
            // mean priors
            target += normal_lpdf(random_effects | 0.0, tau)*relax;

            // scale priors
            target += inv_gamma_lpdf(tau2 | 2, 2);

            // likelihood
            target += temperature*beta*normal_lpdf(observations | mu, sigma);
            target += temperature*(1-beta)*normal_lpdf(observations | alpha, sigma);
        }
        """
        return model_code

    def _check_valid_path_type(self, path_type):
        """ raises an error if path_type is not in the supported types

        parameters
        ----------
        path_type: str
            the type to check
        """

        if path_type not in self.valid_types:
            err = 'Invalid path_type. Must be either' + \
                  ' '.join([''' '{}',''']*len(self.valid_types)).format(*self.valid_types)
            raise ValueError(err)
